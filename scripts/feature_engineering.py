import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.features import AGE_MIDPOINTS, MAX_DIAGNOSES_CAP, MED_COLS, categorize_diag
from src.io import read_pipeline_csv

df = read_pipeline_csv('data/diabetic_data_clean.csv')

# --- 1. Combined prior utilization: inpatient + emergency + outpatient visits ---
# Each showed individual signal in EDA; combining captures overall "high healthcare utilizer" risk
df['total_prior_visits'] = df['number_inpatient'] + df['number_emergency'] + df['number_outpatient']

# --- 2. Flag for any prior inpatient visit (binary is often more robust than raw count at the tail) ---
df['had_prior_inpatient'] = (df['number_inpatient'] > 0).astype(int)

# --- 3. Age as ordinal midpoint instead of string bucket (lets models use it numerically) ---
df['age_midpoint'] = df['age'].map(AGE_MIDPOINTS)

# --- 4. Number of diagnoses, capped to reduce noise from rare high-tail values seen in EDA ---
df['number_diagnoses_capped'] = df['number_diagnoses'].clip(upper=MAX_DIAGNOSES_CAP)

# --- 5. Medication change flag: was diabetic medication changed during encounter? ---
# 'change' column is Ch/No; captures whether care plan was actively adjusted
df['med_changed'] = (df['change'] == 'Ch').astype(int)

# --- 6. Any diabetic medication prescribed at all ---
df['on_diabetes_med'] = (df['diabetesMed'] == 'Yes').astype(int)

# --- 7. Count of medications that were actually changed (up/down) across the 23 drug columns ---
med_cols = [c for c in MED_COLS if c in df.columns]
df['num_med_changes'] = (df[med_cols].isin(['Up', 'Down'])).sum(axis=1)

# --- 8. Simplify diag_1 (primary diagnosis) into broad ICD-9 categories ---
# categorize_diag lives in src/features.py so the app and the tests use the
# exact same bucketing rather than a second copy that can drift.
df['diag_1_category'] = df['diag_1'].apply(categorize_diag)

print("New features created:")
new_feats = ['total_prior_visits', 'had_prior_inpatient', 'age_midpoint',
             'number_diagnoses_capped', 'med_changed', 'on_diabetes_med',
             'num_med_changes', 'diag_1_category']
print(df[new_feats].describe(include='all').T[['count', 'unique', 'top', 'freq']].fillna(''))

print("\nReadmission rate by diag_1 category:")
print(df.groupby('diag_1_category')['readmitted_30d'].agg(['mean', 'count']).sort_values('mean', ascending=False))

print("\nReadmission rate by num_med_changes:")
print(df.groupby('num_med_changes')['readmitted_30d'].mean())

df.to_csv('data/diabetic_data_features.csv', index=False)
print("\nSaved to data/diabetic_data_features.csv, shape:", df.shape)
