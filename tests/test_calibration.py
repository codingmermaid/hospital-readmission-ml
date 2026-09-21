"""Tests for the probability calibration in src/calibration.py."""

import numpy as np
import pytest

from src.calibration import calibrated_probability, model_scale_pos_weight, prior_correct


class TestPriorCorrect:
    def test_unit_weight_is_identity(self):
        """An unweighted model needs no correction."""
        for p in [0.0, 0.1, 0.5, 0.9, 1.0]:
            assert prior_correct(p, 1.0) == pytest.approx(p)

    def test_fixed_points_are_preserved(self):
        """0 and 1 are fixed under the transform for any weight."""
        assert prior_correct(0.0, 7.74) == pytest.approx(0.0)
        assert prior_correct(1.0, 7.74) == pytest.approx(1.0)

    def test_deflates_scores_when_positives_were_upweighted(self):
        """Upweighting positives inflates scores, so correcting must reduce them."""
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert prior_correct(p, 7.74) < p

    def test_inverts_the_weighting_exactly(self):
        """prior_correct is the exact inverse of the reweighting it undoes.

        If p is the true probability, a model trained with weight w estimates
        p_w = w*p / (w*p + 1 - p). Feeding that back through prior_correct must
        return p.
        """
        w = 7.742690701253022
        for p in [0.01, 0.05, 0.112, 0.25, 0.5, 0.8, 0.99]:
            p_weighted = w * p / (w * p + (1 - p))
            assert prior_correct(p_weighted, w) == pytest.approx(p, abs=1e-12)

    def test_is_strictly_monotonic(self):
        """Ranking must be preserved, so ROC-AUC is unchanged by calibration."""
        raw = np.linspace(0.001, 0.999, 500)
        corrected = prior_correct(raw, 7.74)
        assert np.all(np.diff(corrected) > 0)

    def test_maps_the_decision_threshold_onto_the_base_rate(self):
        """The regression this PR fixes, pinned as a test.

        The saved threshold of ~0.502 was chosen for ~54% recall, not as a 50/50
        cutoff. Displayed raw it reads as "50% risk"; corrected it lands near the
        ~11% base rate, which is what it actually represents.
        """
        corrected = prior_correct(0.5019540849522942, 7.742690701253022)
        assert corrected == pytest.approx(0.115, abs=0.005)

    def test_accepts_arrays_and_preserves_shape(self):
        raw = np.array([0.2, 0.4, 0.6])
        out = prior_correct(raw, 7.74)
        assert isinstance(out, np.ndarray)
        assert out.shape == raw.shape

    def test_scalar_in_scalar_out(self):
        assert isinstance(prior_correct(0.4, 7.74), float)

    @pytest.mark.parametrize("bad_weight", [0.0, -1.0])
    def test_rejects_non_positive_weight(self, bad_weight):
        with pytest.raises(ValueError, match="scale_pos_weight"):
            prior_correct(0.5, bad_weight)

    @pytest.mark.parametrize("bad_p", [-0.01, 1.01])
    def test_rejects_out_of_range_probability(self, bad_p):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            prior_correct(bad_p, 7.74)


class TestModelScalePosWeight:
    def test_reads_the_weight_off_the_shipped_model(self, model):
        """The shipped model is weighted; if it stops being, calibration must change."""
        assert model_scale_pos_weight(model) == pytest.approx(7.74, abs=0.01)

    def test_defaults_to_one_when_unweighted(self):
        class Unweighted:
            def get_params(self):
                return {"scale_pos_weight": None}

        assert model_scale_pos_weight(Unweighted()) == 1.0


class TestCalibratedProbability:
    def test_falls_back_to_prior_correction_without_a_calibrator(self):
        assert calibrated_probability(0.4, 7.74, None) == pytest.approx(prior_correct(0.4, 7.74))

    def test_prefers_a_supplied_calibrator(self):
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit([0.0, 0.5, 1.0], [0.0, 0.05, 0.2])

        out = calibrated_probability(0.5, 7.74, iso)
        assert out == pytest.approx(0.05, abs=1e-9)
        # ...and is genuinely not the analytic path.
        assert out != pytest.approx(prior_correct(0.5, 7.74))
