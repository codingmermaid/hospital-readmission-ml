import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

from src.features import CATEGORICAL_COLS, DROP_FOR_MODEL, FEATURE_ORDER, GROUP_COL, MED_COLS, TARGET
from src.io import read_pipeline_csv

MODEL_DIR = 'model'

df = read_pipeline_csv('data/diabetic_data_features.csv')

# Shared with baseline_model.py via src/features.py.
drop_for_model = list(DROP_FOR_MODEL)
med_cols = [c for c in MED_COLS if c in df.columns]
drop_for_model += med_cols

X = df.drop(columns=drop_for_model + [TARGET])
y = df[TARGET]
groups = df[GROUP_COL]

# Pin the column order and the categorical set rather than inferring them from
# dtypes. LightGBM matches features positionally at predict time, so if this
# order ever differs from what app.py builds, every prediction is silently
# wrong instead of raising. tests/test_model_contract.py asserts they agree.
X = X[FEATURE_ORDER]
categorical_cols = list(CATEGORICAL_COLS)
for col in categorical_cols:
    X[col] = X[col].astype('category')

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

model = lgb.LGBMClassifier(
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=30,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    verbose=-1
)

# Note: tried early stopping on AUC first, but AUC improves so gradually here
# (0.653 -> 0.666 over 100 rounds, still climbing) that early stopping kept
# cutting training short. Using a fixed, generous round count instead.
model.fit(
    X_train, y_train,
    categorical_feature=categorical_cols,
    eval_set=[(X_test, y_test)],
    eval_metric='auc'
)

y_proba = model.predict_proba(X_test)[:, 1]

# Fix: pick a threshold deliberately instead of using the default 0.5.
# We match the baseline's recall level (~0.54) for an apples-to-apples comparison,
# since in healthcare the operating point (recall target) matters more than the
# default cutoff -- a hospital would choose this threshold based on capacity to
# act on flagged patients, not by accepting whatever 0.5 happens to produce.
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
target_recall = 0.54
idx = np.argmin(np.abs(recalls[:-1] - target_recall))
chosen_threshold = thresholds[idx]
print(f"\nChosen threshold for ~{target_recall} recall: {chosen_threshold:.4f}")

y_pred = (y_proba >= chosen_threshold).astype(int)

print("\n=== LightGBM Results (threshold tuned to match baseline recall) ===")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1: {f1_score(y_test, y_pred):.4f}")
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nFull classification report:")
print(classification_report(y_test, y_pred, target_names=['No 30d readmit', '30d readmit']))

print("\n=== Feature importance (top 15) ===")
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importance.head(15))

# These used to be written to the repo root while app.py loaded them from
# model/, so a retrain appeared to succeed without changing what the app served.
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, os.path.join(MODEL_DIR, 'lightgbm_model.joblib'))
joblib.dump(categorical_cols, os.path.join(MODEL_DIR, 'categorical_cols.joblib'))
joblib.dump(chosen_threshold, os.path.join(MODEL_DIR, 'lightgbm_threshold.joblib'))
print(f"\nSaved LightGBM model + threshold to {MODEL_DIR}/")
print("Next: python scripts/calibrate_model.py to fit the probability calibrator.")
