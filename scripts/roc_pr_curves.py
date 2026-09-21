import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_curve
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
X_test_raw = X.iloc[test_idx]
y_test = y.iloc[test_idx]

# --- Logistic regression predictions (pipeline handles its own preprocessing) ---
logreg_pipeline = joblib.load('baseline_logreg.joblib')
y_proba_lr = logreg_pipeline.predict_proba(X_test_raw)[:, 1]

# --- LightGBM predictions (needs categorical dtype) ---
lgb_model = joblib.load('lightgbm_model.joblib')
categorical_cols = joblib.load('categorical_cols.joblib')
X_test_lgb = X_test_raw.copy()
for col in categorical_cols:
    X_test_lgb[col] = X_test_lgb[col].astype('category')
y_proba_lgb = lgb_model.predict_proba(X_test_lgb)[:, 1]

# --- Plot styling ---
LILAC = '#A78BFA'
TURQUOISE = '#2DD4BF'
BG = '#0F0B1E'
TEXT = '#F5F3FF'

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.facecolor'] = BG
plt.rcParams['axes.facecolor'] = BG
plt.rcParams['text.color'] = TEXT
plt.rcParams['axes.labelcolor'] = TEXT
plt.rcParams['xtick.color'] = TEXT
plt.rcParams['ytick.color'] = TEXT
plt.rcParams['axes.edgecolor'] = '#4B4066'

def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(color='#2A2340', linewidth=0.6)
    ax.set_axisbelow(True)

# --- ROC curve ---
fig, ax = plt.subplots(figsize=(7, 6))
for y_proba, label, color in [(y_proba_lr, 'Logistic Regression', LILAC), (y_proba_lgb, 'LightGBM', TURQUOISE)]:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, linewidth=2.2, label=f'{label} (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='#4B4066', linewidth=1.5, linestyle='--', label='Random guess (AUC = 0.5)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate (Recall)')
ax.set_title('ROC Curve: Logistic Regression vs LightGBM', fontsize=13, fontweight='bold', color=TEXT)
ax.legend(loc='lower right', frameon=False, labelcolor=TEXT, fontsize=9)
style_ax(ax)
plt.tight_layout()
plt.savefig('charts/roc_curve_comparison.png', dpi=150, facecolor=BG)
plt.close()

# --- Precision-Recall curve (more informative than ROC on imbalanced data) ---
fig, ax = plt.subplots(figsize=(7, 6))
for y_proba, label, color in [(y_proba_lr, 'Logistic Regression', LILAC), (y_proba_lgb, 'LightGBM', TURQUOISE)]:
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    ax.plot(recall, precision, color=color, linewidth=2.2, label=f'{label} (AUC = {pr_auc:.3f})')
baseline_rate = y_test.mean()
ax.axhline(baseline_rate, color='#4B4066', linewidth=1.5, linestyle='--',
           label=f'Random guess (baseline rate = {baseline_rate:.3f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve\n(more informative than ROC on imbalanced data)', fontsize=13, fontweight='bold', color=TEXT)
ax.legend(loc='upper right', frameon=False, labelcolor=TEXT, fontsize=9)
style_ax(ax)
plt.tight_layout()
plt.savefig('charts/pr_curve_comparison.png', dpi=150, facecolor=BG)
plt.close()

print("Saved ROC and PR curves")
print(f"\nLogistic Regression - ROC-AUC: {auc(*roc_curve(y_test, y_proba_lr)[:2]):.4f}")
print(f"LightGBM - ROC-AUC: {auc(*roc_curve(y_test, y_proba_lgb)[:2]):.4f}")
