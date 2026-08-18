import pandas as pd
import numpy as np

df = pd.read_csv('data/diabetic_data_clean.csv')

# --- 1. Combined prior utilization: inpatient + emergency + outpatient visits ---
# Each showed individual signal in EDA; combining captures overall "high healthcare utilizer" risk
df['total_prior_visits'] = df['number_inpatient'] + df['number_emergency'] + df['number_outpatient']

# --- 2. Flag for any prior inpatient visit (binary is often more robust than raw count at the tail) ---
df['had_prior_inpatient'] = (df['number_inpatient'] > 0).astype(int)

# --- 3. Age as ordinal midpoint instead of string bucket (lets models use it numerically) ---
age_map = {
    '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45,
    '[50-60)': 55, '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, '[90-100)': 95
}
df['age_midpoint'] = df['age'].map(age_map)

# --- 4. Number of diagnoses, capped to reduce noise from rare high-tail values seen in EDA ---
df['number_diagnoses_capped'] = df['number_diagnoses'].clip(upper=10)

# --- 5. Medication change flag: was diabetic medication changed during encounter? ---
# 'change' column is Ch/No; captures whether care plan was actively adjusted
df['med_changed'] = (df['change'] == 'Ch').astype(int)

# --- 6. Any diabetic medication prescribed at all ---
df['on_diabetes_med'] = (df['diabetesMed'] == 'Yes').astype(int)

# --- 7. Count of medications that were actually changed (up/down) across the 23 drug columns ---
med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
            'insulin', 'glyburide-metformin', 'glipizide-metformin',
            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
med_cols = [c for c in med_cols if c in df.columns]
df['num_med_changes'] = (df[med_cols].isin(['Up', 'Down'])).sum(axis=1)

# --- 8. Simplify diag_1 (primary diagnosis) into broad ICD-9 categories ---
# Full ICD-9 codes have 700+ levels; group into clinically meaningful buckets
def categorize_diag(code):
    if code == 'Unknown' or pd.isna(code):
        return 'Unknown'
    try:
        code_num = float(code)
    except ValueError:
        return 'Other'  # V or E codes (external causes/supplemental)
    if 390 <= code_num <= 459 or code_num == 785:
        return 'Circulatory'
    elif 460 <= code_num <= 519 or code_num == 786:
        return 'Respiratory'
    elif 520 <= code_num <= 579 or code_num == 787:
        return 'Digestive'
    elif code_num == 250 or (250 <= code_num < 251):
        return 'Diabetes'
    elif 800 <= code_num <= 999:
        return 'Injury'
    elif 710 <= code_num <= 739:
        return 'Musculoskeletal'
    elif 580 <= code_num <= 629 or code_num == 788:
        return 'Genitourinary'
    elif 140 <= code_num <= 239:
        return 'Neoplasms'
    else:
        return 'Other'

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
