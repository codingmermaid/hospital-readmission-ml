import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score,
                               confusion_matrix, classification_report, roc_curve)
import joblib

df = pd.read_csv('data/diabetic_data_features.csv')

# --- Select features for modeling ---
# Drop identifiers, leakage-prone/redundant columns, and raw versions replaced by engineered features
drop_for_model = [
    'encounter_id', 'patient_nbr', 'readmitted',  # identifiers + original 3-class target
    'age',  # replaced by age_midpoint
    'diag_1', 'diag_2', 'diag_3',  # replaced by diag_1_category (diag_2/3 dropped for now, high cardinality)
    'number_diagnoses',  # replaced by capped version
]
med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
            'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
            'metformin-rosiglitazone', 'metformin-pioglitazone', 'examide', 'citoglipton']
med_cols = [c for c in med_cols if c in df.columns]
# individual med columns are summarized by num_med_changes; drop raw versions to avoid 20+ sparse dummies
drop_for_model += med_cols

X = df.drop(columns=drop_for_model + ['readmitted_30d'])
y = df['readmitted_30d']
groups = df['patient_nbr']

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
