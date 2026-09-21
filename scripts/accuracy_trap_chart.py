import matplotlib.pyplot as plt
import numpy as np

LILAC = '#A78BFA'
TURQUOISE = '#2DD4BF'
BG = '#0F0B1E'
TEXT = '#F5F3FF'
RED = '#F87171'  # for the "recall = 0" callout on the naive model

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.facecolor'] = BG
plt.rcParams['axes.facecolor'] = BG
plt.rcParams['text.color'] = TEXT
plt.rcParams['axes.labelcolor'] = TEXT
plt.rcParams['xtick.color'] = TEXT
plt.rcParams['ytick.color'] = TEXT
plt.rcParams['axes.edgecolor'] = '#4B4066'

models = ['Naive\n("always predict\nno readmission")', 'Logistic\nRegression', 'LightGBM']
accuracy = [88.8, 66.0, 67.0]
recall = [0.0, 54.3, 54.0]

x = np.arange(len(models))
width = 0.32

fig, ax = plt.subplots(figsize=(8, 5.5))
bars1 = ax.bar(x - width/2, accuracy, width, label='Accuracy (%)', color=LILAC, edgecolor='#4B4066', linewidth=1)
bars2 = ax.bar(x + width/2, recall, width, label='Recall on at-risk patients (%)', color=TURQUOISE, edgecolor='#4B4066', linewidth=1)

for bar, val in zip(bars1, accuracy, strict=True):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.5, f'{val:.1f}%', ha='center', color=TEXT, fontsize=10)
for bar, val, color in zip(bars2, recall, [RED, TEXT, TEXT], strict=True):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.5, f'{val:.1f}%', ha='center', color=color, fontsize=10,
            fontweight='bold' if val == 0 else 'normal')

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel('%')
ax.set_ylim(0, 100)
ax.set_title('The naive model wins on accuracy\nbut catches zero at-risk patients', fontsize=13, fontweight='bold', color=TEXT)
ax.legend(loc='upper right', frameon=False, labelcolor=TEXT)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', color='#2A2340', linewidth=0.6)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('charts/accuracy_trap.png', dpi=150, facecolor=BG)
plt.close()
print("Saved charts/accuracy_trap.png")
