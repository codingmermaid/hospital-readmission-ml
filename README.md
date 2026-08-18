# Hospital Readmission Risk Prediction

An end-to-end machine learning project predicting whether a hospital patient will be
readmitted within 30 days of discharge, built on the UCI Diabetes 130-US Hospitals
dataset (~100,000 encounters, 1999-2008).

This is a portfolio / educational project. It is **not** a validated clinical tool and
should not be used for real patient care decisions.

## What's in this repo

- `app.py` — Streamlit app that takes patient/encounter details and returns an
  estimated 30-day readmission risk score.
- `model/` — the trained LightGBM model, its categorical feature list, and the
  chosen decision threshold, saved with `joblib`.
- `scripts/` — the full pipeline used to build the model: data cleaning, feature
  engineering, baseline logistic regression, LightGBM training, evaluation
  (ROC/PR curves, the accuracy-vs-recall comparison), and SHAP interpretability.
- `charts/` — all visuals generated during the project (EDA, evaluation, SHAP).

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
7. **Interpretability** — used SHAP to confirm and explain what drives the model's
   predictions.
8. **Deployment (app only, not yet hosted)** — packaged as a Streamlit app.

## Running the app locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

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
