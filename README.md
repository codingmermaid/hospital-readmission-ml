# Hospital Readmission Risk Prediction

An end-to-end machine learning project predicting whether a hospital patient will be
readmitted within 30 days of discharge, built on the UCI Diabetes 130-US Hospitals
dataset (~100,000 encounters, 1999-2008).

This is a portfolio / educational project. It is **not** a validated clinical tool and
should not be used for real patient care decisions.

## What's in this repo

- `app.py` — Streamlit app that takes patient/encounter details and returns a
  calibrated 30-day readmission risk.
- `src/` — shared library code: the feature contract (`features.py`), probability
  calibration (`calibration.py`), and CSV reading that preserves categorical
  levels (`io.py`).
- `model/` — the trained LightGBM model, its categorical feature list, and the
  chosen decision threshold, saved with `joblib`.
- `scripts/` — the full pipeline used to build the model: data cleaning, feature
  engineering, baseline logistic regression, LightGBM training, calibration,
  evaluation (ROC/PR curves, the accuracy-vs-recall comparison), and SHAP
  interpretability.
- `charts/` — all visuals generated during the project (EDA, evaluation, SHAP).
- `tests/` — pytest suite covering the calibration maths and the contract between
  the saved model and the code that serves it.

## Pipeline summary

1. **Data cleaning** — dropped columns that were mostly missing (`weight`, `payer_code`),
   removed hospice/death discharges, binarized the original 3-class target into a
   30-day readmission flag.
2. **EDA** — found that prior inpatient visits was the strongest single predictor of
   readmission risk, with a wide class imbalance (~11% positive rate).
3. **Feature engineering** — combined prior-visit counts, collapsed diagnosis codes into
   broad categories, added medication-change signals.
4. **Baseline model** — logistic regression with balanced class weights, used as a
   reference point before adding model complexity.
5. **LightGBM model** — trained with native categorical handling and
   `scale_pos_weight` for the class imbalance; outperformed the baseline modestly
   across ROC-AUC, precision, and recall.
6. **Evaluation** — used ROC-AUC, precision, recall, and precision-recall curves
   instead of accuracy, since a naive "always predict no readmission" model scores
   88.8% accuracy while catching zero at-risk patients.
7. **Calibration** — converted the model's reweighted scores back into probabilities
   that can honestly be shown as a percentage (see below).
8. **Interpretability** — used SHAP to confirm and explain what drives the model's
   predictions.
9. **Deployment (app only, not yet hosted)** — packaged as a Streamlit app.

## Probability calibration

The model is trained with `scale_pos_weight = n_neg / n_pos ≈ 7.74`, which is a
reasonable way to learn under an ~11% positive rate. It has a consequence that is
easy to miss: `predict_proba` no longer returns a probability of readmission. It
returns a probability under a reweighted world where readmission is about 8x more
common than it really is.

That matters as soon as a number is shown to a human. A 65-year-old with no prior
inpatient visits — a below-average-risk patient — scored **39.3%**, displayed as
"Estimated 30-day readmission risk". The observed base rate is about 11%.

Undoing the reweighting is a closed form. If a model is fitted with weight `w` on
the positive class, it estimates `p_w = w·p / (w·p + 1 − p)`, so the true
probability is:

```
p = p_w / (p_w + w · (1 − p_w))
```

Applying that maps the example above to **7.7%**, and maps the saved decision
threshold of 0.502 to **11.5%** — right at the base rate, which is what a
threshold chosen for ~54% recall should look like.

The correction is monotonic, so it changes the numbers without reordering any
patients: **ROC-AUC, precision and recall are all unchanged.** Only the displayed
probability moves.

`scripts/calibrate_model.py` additionally fits an isotonic regression on a
held-out calibration split (split by patient, so no patient appears on both
sides) and writes `model/calibrator.joblib`. When that file is present the app
prefers it; otherwise it falls back to the analytic correction above, so the app
works correctly straight from a clone.

```bash
python scripts/calibrate_model.py
```

## A note on `None` and `pd.read_csv`

`A1Cresult` and `max_glu_serum` both use the literal string `"None"` to mean "this
test was not performed" — about 95% of the rows in each column.

`pd.read_csv` treats `"None"` as a missing value by default; it sits in
`pandas._libs.parsers.STR_NA_VALUES` alongside `"NA"` and `"NULL"`. Because the
pipeline passes data between stages as CSV, that level was silently converted to
`NaN` before the model ever saw it. You can confirm it from the saved artifact:

```python
>>> model.booster_.pandas_categorical[3:5]
[['>200', '>300', 'Norm'], ['>7', '>8', 'Norm']]   # no 'None'
```

For LightGBM the damage is limited — it handles `NaN` natively, so "not tested"
was still learnable, just as missingness rather than as a labelled level, and
entirely by accident. For the logistic-regression baseline it is worse: that
pipeline one-hot encodes, so the level vanished from the design matrix entirely
and the baseline could not represent "not tested" at all. Some of the reported
LightGBM-over-baseline gap is this bug rather than a real modelling difference.

`src/io.read_pipeline_csv` fixes the reader. The committed model was trained
before the fix, so the app deliberately still sends `NaN` for "Not tested" to
match the artifact it actually ships.

### Retraining after the `None` fix

Rerunning the pipeline now produces a genuinely different feature space, so it
needs a matching update to the code:

```bash
python scripts/clean.py
python scripts/feature_engineering.py
python scripts/lightgbm_model.py
python scripts/calibrate_model.py
```

`tests/test_model_contract.py::test_none_is_absent_from_the_test_result_columns`
will then fail on purpose. At that point add `"None"` to `TRAINING_LEVELS` in
`src/features.py`, and change `app.py` to send the string `"None"` instead of
`NaN` for "Not tested". Until then the test documents the current state.

## Running the app locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Running the tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest tests/ -v
```

The tests need only the committed model artifact, not the dataset, so they run on
a fresh clone and in CI. They cover the calibration maths, the feature-order and
categorical-level contract between `src/features.py` and the saved model, and the
guards that stop the app offering an option the model never saw.

## Data

The dataset used is the [Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
dataset from the UCI Machine Learning Repository. It is not included in this repo;
download it separately if you want to rerun the pipeline scripts from scratch.

## Model performance

| Metric | Logistic Regression | LightGBM |
|---|---|---|
| ROC-AUC | 0.646 | 0.660 |
| Precision | 0.172 | 0.178 |
| Recall | 0.543 | 0.540 |

Both models were evaluated on a held-out test set split by patient (not by encounter),
so that no patient's data appears in both training and test sets.

These figures are unaffected by the calibration work above — the correction is
monotonic, so it cannot change any ranking-based or threshold-based metric once the
threshold is mapped through it too.
