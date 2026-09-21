import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import GroupShuffleSplit

from src.io import read_pipeline_csv

df = read_pipeline_csv('data/diabetic_data_features.csv')

drop_for_model = ['encounter_id', 'patient_nbr', 'readmitted', 'age', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses']
med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
            'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
            'metformin-rosiglitazone', 'metformin-pioglitazone', 'examide', 'citoglipton']
med_cols = [c for c in med_cols if c in df.columns]
drop_for_model += med_cols

X = df.drop(columns=drop_for_model + ['readmitted_30d'])
y = df['readmitted_30d']
groups = df['patient_nbr']

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))
X_test = X.iloc[test_idx].copy()

lgb_model = joblib.load('lightgbm_model.joblib')
categorical_cols = joblib.load('categorical_cols.joblib')
for col in categorical_cols:
    X_test[col] = X_test[col].astype('category')

# Use a sample for speed -- SHAP on the full 20K test set is slow, 2000 is plenty for stable plots
sample = X_test.sample(n=2000, random_state=42)

explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(sample)

# TreeExplainer returns a list for binary classification in some SHAP versions; normalize
if isinstance(shap_values, list):
    shap_values = shap_values[1]

LILAC = '#A78BFA'
TURQUOISE = '#2DD4BF'
BG = '#0F0B1E'
TEXT = '#F5F3FF'

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.facecolor'] = BG
plt.rcParams['text.color'] = TEXT

# --- Summary bar plot: mean |SHAP value| per feature (global importance) ---
fig = plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, sample, plot_type='bar', show=False, color=TURQUOISE, max_display=15)
ax = plt.gca()
ax.set_facecolor(BG)
fig.patch.set_facecolor(BG)
ax.tick_params(colors=TEXT)
ax.xaxis.label.set_color(TEXT)
ax.title.set_color(TEXT)
for text in ax.get_yticklabels() + ax.get_xticklabels():
    text.set_color(TEXT)
plt.title('What drives the model\'s readmission risk score?', color=TEXT, fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/shap_importance.png', dpi=150, facecolor=BG, bbox_inches='tight')
plt.close()

# --- Beeswarm plot: shows direction of effect, not just magnitude ---
fig = plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, sample, show=False, max_display=15)
ax = plt.gca()
ax.set_facecolor(BG)
fig.patch.set_facecolor(BG)
ax.tick_params(colors=TEXT)
ax.xaxis.label.set_color(TEXT)
ax.title.set_color(TEXT)
for text in ax.get_yticklabels() + ax.get_xticklabels():
    text.set_color(TEXT)
plt.title('How each feature pushes risk up or down', color=TEXT, fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/shap_beeswarm.png', dpi=150, facecolor=BG, bbox_inches='tight')
plt.close()

print("Saved SHAP importance + beeswarm plots")

# Print top features by mean absolute SHAP value for the writeup
mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=sample.columns).sort_values(ascending=False)
print("\nTop 10 features by mean |SHAP value|:")
print(mean_abs_shap.head(10))
