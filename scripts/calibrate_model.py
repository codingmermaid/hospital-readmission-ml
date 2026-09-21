"""
Fit and evaluate a probability calibrator for the LightGBM model.

Run after `lightgbm_model.py`:

    python scripts/calibrate_model.py

Requires `data/diabetic_data_features.csv` (produced by `feature_engineering.py`),
which is not committed -- see the README for how to rebuild it.

Method
------
The model is *not* refitted. We reuse the exact same patient-grouped split it was
trained with (`GroupShuffleSplit`, `random_state=42`, `test_size=0.2`), then cut
the held-out portion in half -- again by patient, so no patient appears on both
sides:

    train (80%)  -> already used to fit the model, untouched here
    calib (10%)  -> fit the isotonic regression
    eval  (10%)  -> report honest calibration metrics

Fitting the calibrator on data the model trained on would produce a calibrator
that looks excellent and generalises badly, which is the usual way this goes
wrong.

Outputs
-------
    model/calibrator.joblib   isotonic calibrator consumed by app.py
    charts/calibration_curve.png   reliability diagram, before vs after
"""

import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration import model_scale_pos_weight, prior_correct
from src.features import CATEGORICAL_COLS, FEATURE_ORDER, GROUP_COL, TARGET
from src.io import read_pipeline_csv

FEATURES_CSV = "data/diabetic_data_features.csv"
MODEL_PATH = "model/lightgbm_model.joblib"
CALIBRATOR_PATH = "model/calibrator.joblib"
CHART_PATH = "charts/calibration_curve.png"

RANDOM_STATE = 42


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Average gap between predicted and observed rates, weighted by bin size.

    Complements the Brier score, which mixes calibration and discrimination
    together; ECE isolates the calibration part.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, edges[1:-1], right=True)
    error = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        error += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(error)


def main() -> int:
    if not os.path.exists(FEATURES_CSV):
        print(f"ERROR: {FEATURES_CSV} not found.")
        print("The dataset is not committed to this repo. Rebuild it with:")
        print("  python scripts/clean.py && python scripts/feature_engineering.py")
        return 1

    df = read_pipeline_csv(FEATURES_CSV)
    model = joblib.load(MODEL_PATH)
    weight = model_scale_pos_weight(model)
    print(f"Model scale_pos_weight: {weight:.4f}")

    X = df[FEATURE_ORDER].copy()
    for col in CATEGORICAL_COLS:
        X[col] = X[col].astype("category")
    y = df[TARGET].to_numpy()
    groups = df[GROUP_COL].to_numpy()

    # Reproduce the training split exactly, so `holdout` is genuinely unseen.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    _, holdout_idx = next(splitter.split(X, y, groups=groups))

    # Split the holdout in half, again grouped by patient.
    inner = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=RANDOM_STATE)
    rel_calib, rel_eval = next(inner.split(X.iloc[holdout_idx], y[holdout_idx], groups=groups[holdout_idx]))
    calib_idx = holdout_idx[rel_calib]
    eval_idx = holdout_idx[rel_eval]

    assert not set(groups[calib_idx]) & set(groups[eval_idx]), "patient leaked across split"
    print(f"Calibration rows: {len(calib_idx)} | Evaluation rows: {len(eval_idx)}")
    print(f"Observed base rate (eval): {y[eval_idx].mean():.4f}")

    raw_calib = model.predict_proba(X.iloc[calib_idx])[:, 1]
    raw_eval = model.predict_proba(X.iloc[eval_idx])[:, 1]

    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(raw_calib, y[calib_idx])

    variants = {
        "raw (scale_pos_weight inflated)": raw_eval,
        "analytic prior correction": prior_correct(raw_eval, weight),
        "isotonic calibration": isotonic.predict(raw_eval),
    }

    print(f"\n{'variant':<34} {'mean pred':>10} {'Brier':>9} {'ECE':>8} {'ROC-AUC':>9}")
    print("-" * 74)
    for name, probs in variants.items():
        print(
            f"{name:<34} {probs.mean():>10.4f} "
            f"{brier_score_loss(y[eval_idx], probs):>9.4f} "
            f"{expected_calibration_error(y[eval_idx], probs):>8.4f} "
            f"{roc_auc_score(y[eval_idx], probs):>9.4f}"
        )
    print(
        "\nROC-AUC is identical across all three: both corrections are monotonic, "
        "so they change the numbers without reordering any patients."
    )

    os.makedirs(os.path.dirname(CALIBRATOR_PATH), exist_ok=True)
    joblib.dump(isotonic, CALIBRATOR_PATH)
    print(f"\nSaved calibrator to {CALIBRATOR_PATH}")

    _plot_reliability(y[eval_idx], variants)
    return 0


def _plot_reliability(y_true, variants) -> None:
    os.makedirs(os.path.dirname(CHART_PATH), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")

    for name, probs in variants.items():
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", lw=1.6, label=name)

    ax.axhline(y_true.mean(), color="grey", lw=0.8, alpha=0.6)
    ax.annotate(
        f"observed base rate {y_true.mean():.1%}",
        xy=(0.55, y_true.mean()),
        xytext=(0.55, y_true.mean() + 0.04),
        fontsize=9,
        color="grey",
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed readmission rate")
    ax.set_title("Reliability diagram (held-out patients)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=150)
    print(f"Saved reliability diagram to {CHART_PATH}")


if __name__ == "__main__":
    raise SystemExit(main())
