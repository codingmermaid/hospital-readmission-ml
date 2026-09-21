"""
Contract tests between the shipped model artifact and the code that serves it.

These are the checks that would have caught the bugs this PR fixes. They need
only the committed model, not the dataset, so they run in CI.
"""

import numpy as np
import pytest

from src.calibration import calibrated_probability, model_scale_pos_weight
from src.features import (
    CATEGORICAL_COLS,
    FEATURE_ORDER,
    TRAINING_LEVELS,
    derive_features,
    to_model_frame,
)


class TestFeatureContract:
    def test_feature_order_matches_the_trained_model_exactly(self, model):
        """Order is positional in LightGBM: a permutation here is silent corruption."""
        assert list(model.feature_name_) == FEATURE_ORDER

    def test_feature_count_matches(self, model):
        assert model.n_features_in_ == len(FEATURE_ORDER)

    def test_declared_categoricals_match_the_model(self, model):
        """CATEGORICAL_COLS must line up with what the booster recorded."""
        recorded = dict(zip(CATEGORICAL_COLS, model.booster_.pandas_categorical, strict=True))
        assert set(recorded) == set(CATEGORICAL_COLS)

    def test_training_levels_match_the_booster(self, model):
        """TRAINING_LEVELS is a hand-copy of booster state; keep it honest.

        This is what documents that max_glu_serum and A1Cresult have no "None"
        level -- the CSV round-trip bug described in src/io.py.
        """
        for col, levels in zip(CATEGORICAL_COLS, model.booster_.pandas_categorical, strict=True):
            assert sorted(TRAINING_LEVELS[col]) == sorted(levels), col

    def test_none_is_absent_from_the_test_result_columns(self, model):
        """Pins the known data bug so a retrain that fixes it is visible.

        When the pipeline is rerun with src/io.read_pipeline_csv, "None" becomes
        a real level and this test will fail -- deliberately. Update
        TRAINING_LEVELS at that point; see the README.
        """
        levels = dict(zip(CATEGORICAL_COLS, model.booster_.pandas_categorical, strict=True))
        assert "None" not in levels["max_glu_serum"]
        assert "None" not in levels["A1Cresult"]


class TestFrameBuilding:
    def test_builds_columns_in_model_order(self, encounter):
        frame = to_model_frame(derive_features(encounter))
        assert list(frame.columns) == FEATURE_ORDER

    def test_produces_a_single_row(self, encounter):
        assert len(to_model_frame(derive_features(encounter))) == 1

    def test_categoricals_carry_full_training_levels(self, encounter):
        """Not just the one value in this row -- see to_model_frame's docstring."""
        frame = to_model_frame(derive_features(encounter))
        for col in CATEGORICAL_COLS:
            assert list(frame[col].cat.categories) == TRAINING_LEVELS[col], col

    def test_out_of_vocabulary_value_becomes_nan_visibly(self, encounter):
        """An unknown level must surface as NaN here, not vanish inside LightGBM."""
        encounter["race"] = "NotARealCategory"
        frame = to_model_frame(derive_features(encounter))
        assert frame["race"].isna().all()

    def test_nan_test_result_is_preserved(self, encounter):
        """The app sends NaN for "not tested"; it must survive frame building."""
        encounter["A1Cresult"] = np.nan
        frame = to_model_frame(derive_features(encounter))
        assert frame["A1Cresult"].isna().all()


class TestDerivedFeatures:
    def test_total_prior_visits_sums_the_three_counts(self, encounter):
        encounter.update(number_inpatient=2, number_emergency=3, number_outpatient=4)
        assert derive_features(encounter)["total_prior_visits"] == 9

    @pytest.mark.parametrize("inpatient,expected", [(0, 0), (1, 1), (5, 1)])
    def test_had_prior_inpatient_is_a_flag(self, encounter, inpatient, expected):
        encounter["number_inpatient"] = inpatient
        assert derive_features(encounter)["had_prior_inpatient"] == expected

    @pytest.mark.parametrize("n,expected", [(3, 3), (10, 10), (16, 10)])
    def test_diagnoses_are_capped(self, encounter, n, expected):
        encounter["number_diagnoses"] = n
        assert derive_features(encounter)["number_diagnoses_capped"] == expected

    def test_raw_number_diagnoses_is_consumed(self, encounter):
        """The uncapped column is not a model feature and must not leak through."""
        assert "number_diagnoses" not in derive_features(encounter)

    def test_med_changed_tracks_the_change_column(self, encounter):
        encounter["change"] = "Ch"
        assert derive_features(encounter)["med_changed"] == 1
        encounter["change"] = "No"
        assert derive_features(encounter)["med_changed"] == 0

    def test_does_not_mutate_its_input(self, encounter):
        before = dict(encounter)
        derive_features(encounter)
        assert encounter == before


class TestEndToEndPrediction:
    def test_predicts_a_valid_probability(self, model, encounter):
        raw = model.predict_proba(to_model_frame(derive_features(encounter)))[:, 1][0]
        assert 0.0 <= raw <= 1.0

    def test_calibrated_risk_is_plausible_for_a_low_risk_patient(self, model, encounter):
        """The headline fix.

        A 65-year-old with no prior visits is below-average risk. The raw score
        reads ~39%; the calibrated value must land near the ~11% base rate
        instead of several times above it.
        """
        frame = to_model_frame(derive_features(encounter))
        raw = model.predict_proba(frame)[:, 1][0]
        risk = calibrated_probability(raw, model_scale_pos_weight(model), None)

        assert raw > 0.25, "raw score is inflated, as expected pre-calibration"
        assert 0.02 < risk < 0.15, f"calibrated risk {risk:.3f} outside plausible range"
        assert risk < raw / 2, "calibration should substantially deflate the raw score"

    def test_prior_inpatient_visits_increase_risk(self, model, encounter):
        """Monotone in the strongest predictor the EDA identified."""
        risks = []
        for visits in [0, 1, 3, 8]:
            encounter.update(number_inpatient=visits)
            frame = to_model_frame(derive_features(encounter))
            raw = model.predict_proba(frame)[:, 1][0]
            risks.append(calibrated_probability(raw, model_scale_pos_weight(model), None))
        assert risks == sorted(risks), f"risk not monotone in prior visits: {risks}"

    def test_calibration_preserves_ranking(self, model, encounter):
        """Calibration must not reorder patients -- that would change ROC-AUC."""
        raws = []
        for visits in [0, 1, 2, 5, 9]:
            encounter.update(number_inpatient=visits)
            frame = to_model_frame(derive_features(encounter))
            raws.append(model.predict_proba(frame)[:, 1][0])

        weight = model_scale_pos_weight(model)
        calibrated = [calibrated_probability(r, weight, None) for r in raws]
        assert np.argsort(raws).tolist() == np.argsort(calibrated).tolist()
