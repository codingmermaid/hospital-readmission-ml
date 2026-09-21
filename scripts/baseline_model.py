import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
                               classification_report,
                               confusion_matrix,
                               f1_score,
                               precision_score,
                               recall_score,
                               roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import DROP_FOR_MODEL, GROUP_COL, MED_COLS, TARGET
from src.io import read_pipeline_csv

df = read_pipeline_csv('data/diabetic_data_features.csv')

# Feature selection now comes from src/features.py, which is the single place
# these lists are defined; they used to be duplicated verbatim here and in
# lightgbm_model.py, with nothing checking the two stayed in sync.
drop_for_model = list(DROP_FOR_MODEL)
med_cols = [c for c in MED_COLS if c in df.columns]
drop_for_model += med_cols

X = df.drop(columns=drop_for_model + [TARGET])
y = df[TARGET]
groups = df[GROUP_COL]

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"Categorical features ({len(categorical_cols)}):", categorical_cols)
print(f"Numeric features ({len(numeric_cols)}):", numeric_cols)

# --- Group-based split (no patient leakage) ---
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")
print(f"Train positive rate: {y_train.mean():.3f} | Test positive rate: {y_test.mean():.3f}")

# --- Preprocessing pipeline ---
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_cols)
])

# --- Baseline: Logistic Regression with class weighting for imbalance ---
baseline_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])

baseline_pipeline.fit(X_train, y_train)

y_pred = baseline_pipeline.predict(X_test)
y_proba = baseline_pipeline.predict_proba(X_test)[:, 1]

print("\n=== Baseline Logistic Regression Results ===")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1: {f1_score(y_test, y_pred):.4f}")
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nFull classification report:")
print(classification_report(y_test, y_pred, target_names=['No 30d readmit', '30d readmit']))

# Save for later comparison + deployment
joblib.dump(baseline_pipeline, 'baseline_logreg.joblib')
np.save('data/X_test_idx.npy', test_idx)
np.save('data/X_train_idx.npy', train_idx)
X.to_csv('data/X_features.csv', index=False)
y.to_csv('data/y_target.csv', index=False)
print("\nSaved model + train/test indices for next steps")
