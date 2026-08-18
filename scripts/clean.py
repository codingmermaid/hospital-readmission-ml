import pandas as pd
import numpy as np

df = pd.read_csv('data/diabetic_data.csv')
print("Starting shape:", df.shape)

# Replace '?' with NaN across the board for proper missing-value handling
df = df.replace('?', np.nan)

# --- Drop encounters where discharge was to hospice or patient died ---
# These can't be "readmitted" in any meaningful sense; including them skews the target
expired_hospice_codes = [11, 13, 14, 19, 20, 21]
before = len(df)
df = df[~df['discharge_disposition_id'].isin(expired_hospice_codes)]
print(f"Dropped {before - len(df)} rows for death/hospice discharge")

# --- Drop unusable / near-constant columns ---
drop_cols = ['weight', 'payer_code']  # weight 97% missing, payer_code not clinically predictive + 40% missing
df = df.drop(columns=drop_cols)

# --- medical_specialty: 49% missing, keep but bucket rare categories + missing as its own category ---
df['medical_specialty'] = df['medical_specialty'].fillna('Missing')
top_specialties = df['medical_specialty'].value_counts().nlargest(10).index
df['medical_specialty'] = df['medical_specialty'].where(
    df['medical_specialty'].isin(top_specialties), 'Other'
)

# --- race: fill missing with 'Unknown' rather than dropping ---
df['race'] = df['race'].fillna('Unknown')

# --- diag_1/2/3: small number missing, fill with 'Unknown' ---
for col in ['diag_1', 'diag_2', 'diag_3']:
    df[col] = df[col].fillna('Unknown')

# --- Binarize target: 30-day readmission is the clinically/financially meaningful outcome ---
df['readmitted_30d'] = (df['readmitted'] == '<30').astype(int)

print("\nFinal shape:", df.shape)
print("\nNew target distribution:")
print(df['readmitted_30d'].value_counts(normalize=True))

df.to_csv('data/diabetic_data_clean.csv', index=False)
print("\nSaved cleaned data to data/diabetic_data_clean.csv")
