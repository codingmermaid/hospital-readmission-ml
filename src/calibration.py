"""
Turning the model's score into something that can honestly be called a risk.

The LightGBM model is trained with ``scale_pos_weight = n_neg / n_pos = 7.74``.
That is a reasonable choice for learning under a ~11% positive rate, but it has a
consequence that is easy to miss: the numbers coming out of ``predict_proba`` are
no longer probabilities of readmission. They are probabilities under a reweighted
world in which readmission is roughly 8x more common than it really is.

Concretely, before this module existed the app showed a 65-year-old with no prior
inpatient visits an "Estimated 30-day readmission risk" of **39.3%**. The observed
base rate in the data is about 11%, and that patient is lower risk than average.
The number was inflated by about 5x.

Undoing it is a closed form. Training with weight ``w`` on the positive class
means the fitted model estimates

    p_w = w * p / (w * p + (1 - p))

where ``p`` is the true conditional probability. Solving for ``p``:

    p = p_w / (p_w + w * (1 - p_w))

That is :func:`prior_correct`. It needs no data and no refit, so it works against
the already-trained artifact committed to this repo.

Prior correction fixes the *level*. It does not fix the *shape* -- a boosted tree
can still be over-confident in the tails even after the base rate is right. For
that, :mod:`scripts.calibrate_model` fits an isotonic regression on a held-out
calibration split and saves it to ``model/calibrator.joblib``. When that file is
present :func:`calibrated_probability` prefers it; otherwise it falls back to the
analytic correction.
"""

from __future__ import annotations

import os

import numpy as np

#: Path of the optional isotonic calibrator produced by scripts/calibrate_model.py.
CALIBRATOR_PATH = os.path.join("model", "calibrator.joblib")


def prior_correct(p_weighted, scale_pos_weight: float):
    """Undo the odds inflation introduced by ``scale_pos_weight``.

    Args:
        p_weighted: Score(s) from ``predict_proba`` on a model trained with
            ``scale_pos_weight``. Scalar or array-like.
        scale_pos_weight: The weight applied to the positive class at fit time.

    Returns:
        The corresponding true-prior probability, same shape as the input.

    The transform is monotonic, so it never reorders patients and leaves ROC-AUC
    unchanged. It moves the *values*, which is exactly what a displayed
    percentage depends on.
    """
    if scale_pos_weight <= 0:
        raise ValueError(f"scale_pos_weight must be positive, got {scale_pos_weight}")

    p = np.asarray(p_weighted, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p_weighted must lie in [0, 1]")

    corrected = p / (p + scale_pos_weight * (1.0 - p))
    return float(corrected) if np.isscalar(p_weighted) or p.ndim == 0 else corrected


def model_scale_pos_weight(model) -> float:
    """Read ``scale_pos_weight`` back off a fitted LightGBM estimator.

    Returns 1.0 when the model was trained unweighted, which makes
    :func:`prior_correct` a no-op.
    """
    weight = model.get_params().get("scale_pos_weight")
    if weight is None:
        return 1.0
    return float(weight)


def load_calibrator(path: str = CALIBRATOR_PATH):
    """Load the isotonic calibrator if one has been fitted, else ``None``.

    Absence is a normal state, not an error: the calibrator requires the raw
    dataset, which is not committed to this repo.
    """
    if not os.path.exists(path):
        return None
    import joblib

    return joblib.load(path)


def calibrated_probability(
    p_weighted,
    scale_pos_weight: float,
    calibrator: object | None = None,
):
    """Best available estimate of the true readmission probability.

    Uses the fitted isotonic calibrator when one is supplied, and the analytic
    prior correction otherwise. Both are monotonic in the raw score, so the
    ranking of patients -- and therefore ROC-AUC -- is identical either way.
    """
    if calibrator is not None:
        p = np.asarray(p_weighted, dtype=float).reshape(-1)
        out = calibrator.predict(p)
        return float(out[0]) if np.isscalar(p_weighted) or np.ndim(p_weighted) == 0 else out
    return prior_correct(p_weighted, scale_pos_weight)
