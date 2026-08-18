import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

df = pd.read_csv('data/diabetic_data_clean.csv')

# Brand palette
LILAC = '#A78BFA'
TURQUOISE = '#2DD4BF'
BG = '#0F0B1E'  # violet-black
TEXT = '#F5F3FF'

plt.rcParams['font.family'] = 'DejaVu Sans'  # fallback since Syne/DM Sans may not be installed
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
    ax.grid(axis='y', color='#2A2340', linewidth=0.6)
    ax.set_axisbelow(True)

# --- Chart 1: Readmission rate by prior inpatient visits ---
fig, ax = plt.subplots(figsize=(8, 5))
data = df.groupby('number_inpatient')['readmitted_30d'].mean().head(9) * 100
ax.bar(data.index.astype(str), data.values, color=LILAC, edgecolor=TURQUOISE, linewidth=1.2)
ax.set_xlabel('Prior inpatient visits (past year)')
ax.set_ylabel('30-day readmission rate (%)')
ax.set_title('Prior hospitalizations predict future readmission', fontsize=13, fontweight='bold', color=TEXT)
style_ax(ax)
plt.tight_layout()
plt.savefig('charts/readmission_by_prior_visits.png', dpi=150, facecolor=BG)
plt.close()

# --- Chart 2: Readmission rate by age ---
fig, ax = plt.subplots(figsize=(8, 5))
data = df.groupby('age')['readmitted_30d'].mean() * 100
ax.plot(data.index, data.values, marker='o', color=TURQUOISE, linewidth=2, markersize=7,
        markerfacecolor=LILAC, markeredgecolor=TURQUOISE)
ax.set_xlabel('Age group')
ax.set_ylabel('30-day readmission rate (%)')
ax.set_title('Readmission risk by age group', fontsize=13, fontweight='bold', color=TEXT)
plt.xticks(rotation=45, ha='right')
style_ax(ax)
plt.tight_layout()
plt.savefig('charts/readmission_by_age.png', dpi=150, facecolor=BG)
plt.close()

# --- Chart 3: Feature correlation with target ---
numeric_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
corr = df[numeric_cols + ['readmitted_30d']].corr()['readmitted_30d'].drop('readmitted_30d').sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
colors = [TURQUOISE if v < 0 else LILAC for v in corr.values]
ax.barh(corr.index, corr.values, color=colors, edgecolor='#4B4066', linewidth=1)
ax.set_xlabel('Correlation with 30-day readmission')
ax.set_title('Which features correlate most with readmission?', fontsize=13, fontweight='bold', color=TEXT)
style_ax(ax)
ax.grid(axis='x', color='#2A2340', linewidth=0.6)
plt.tight_layout()
plt.savefig('charts/feature_correlation.png', dpi=150, facecolor=BG)
plt.close()

# --- Chart 4: Target class imbalance ---
fig, ax = plt.subplots(figsize=(6, 5))
counts = df['readmitted_30d'].value_counts().sort_index()
labels = ['Not readmitted\nwithin 30 days', 'Readmitted\nwithin 30 days']
ax.bar(labels, counts.values, color=[LILAC, TURQUOISE], edgecolor='#4B4066', linewidth=1.2, width=0.5)
for i, v in enumerate(counts.values):
    ax.text(i, v + 800, f'{v:,}\n({v/len(df)*100:.1f}%)', ha='center', color=TEXT, fontsize=10)
ax.set_ylabel('Number of encounters')
ax.set_title('Class imbalance: only ~11% are 30-day readmissions', fontsize=13, fontweight='bold', color=TEXT)
style_ax(ax)
plt.tight_layout()
plt.savefig('charts/class_imbalance.png', dpi=150, facecolor=BG)
plt.close()

print("Saved 4 charts to charts/")
