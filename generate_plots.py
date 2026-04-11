import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set global aesthetics using modern styling
sns.set_theme(style="whitegrid", rc={"axes.spines.top": False, "axes.spines.right": False})

plot_dir = r"c:\Users\xxerp\Desktop\Career\Proyectos\GitHub\LITE\plots"
os.makedirs(plot_dir, exist_ok=True)

# ---------------------------------------------------------
# 1. Ensemble Size Effect
# ---------------------------------------------------------
df_ens = pd.read_csv("results_ensemble_study.csv")
ens_cols = [c for c in df_ens.columns if c.startswith('LITETime-')]
ens_means = df_ens[ens_cols].mean()
n_members = np.arange(1, 11)

plt.figure(figsize=(8, 5))
sns.lineplot(x=n_members, y=ens_means.values, marker='o', color='#2b5c8f', linewidth=2.5, markersize=8)
plt.title("Effect of Ensemble Size on Mean Accuracy (128 Datasets)", fontsize=14, pad=15)
plt.xlabel("Ensemble Size (N)", fontsize=12)
plt.ylabel("Mean Accuracy", fontsize=12)
plt.xticks(n_members)
plt.axvline(x=5, color='gray', linestyle='--', alpha=0.7)
plt.text(5.2, ens_means.values.min() + 0.002, "Default N=5\n(Diminishing Returns)", color='gray', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "ensemble_size_effect.png"), dpi=300)
plt.close()


# ---------------------------------------------------------
# 2. Multivariate Methods Comparison
# ---------------------------------------------------------
df_mv = pd.read_csv("results_multivariate.csv")
mv_means = df_mv.drop(columns=['dataset']).mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
colors = ['#1f77b4' if m == 'ConvTran' else '#ff7f0e' if 'LITE' in m else '#aec7e8' for m in mv_means.index]
ax = sns.barplot(x=mv_means.values, y=mv_means.index, palette=colors)
plt.title("Mean Accuracy on 30 Multivariate Datasets", fontsize=14, pad=15)
plt.xlabel("Mean Accuracy", fontsize=12)
plt.ylabel("")
plt.xlim(0.65, 0.76)

# Add value labels
for i, v in enumerate(mv_means.values):
    ax.text(v + 0.001, i, f"{v:.4f}", va='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "multivariate_comparison.png"), dpi=300)
plt.close()


# ---------------------------------------------------------
# 3. Univariate Benchmarks Boxplot
# ---------------------------------------------------------
df_uni = pd.read_csv("results.csv")
# Melt for seaborn boxplot
df_uni_melted = df_uni.melt(id_vars=['dataset'], var_name='Method', value_name='Accuracy')

# Calculate means to sort the boxplot
uni_means = df_uni.drop(columns=['dataset']).mean().sort_values(ascending=False)
df_uni_melted['Method'] = pd.Categorical(df_uni_melted['Method'], categories=uni_means.index, ordered=True)

plt.figure(figsize=(10, 6))
sns.boxplot(x='Accuracy', y='Method', data=df_uni_melted, palette="Set2", showfliers=False)
plt.title("Accuracy Distribution across 128 UCR Datasets", fontsize=14, pad=15)
plt.xlabel("Accuracy", fontsize=12)
plt.ylabel("")
plt.xlim(0.5, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "univariate_boxplot.png"), dpi=300)
plt.close()


# ---------------------------------------------------------
# 4. Improvement Experiments Delta (Replacing accuracy_comparison.png)
# ---------------------------------------------------------
configs = ['Global min-max', 'Batch 128', 'Baseline', 'Batch 32', 'AdamW', 'Global z-norm', 'Differenced channel']
deltas = [-1.61, -0.07, 0.0, 0.18, 0.23, 0.88, 1.40]

plt.figure(figsize=(9, 5))
c_map = ['#e74c3c' if d < 0 else '#95a5a6' if d == 0 else '#2ecc71' for d in deltas]
ax = sns.barplot(x=deltas, y=configs, palette=c_map)

plt.title("Performance Delta vs. Baseline (31-Dataset Fast Subset)", fontsize=14, pad=15)
plt.xlabel("Accuracy Delta (%)", fontsize=12)
plt.ylabel("")
plt.axvline(x=0, color='black', linewidth=1.5)

for i, v in enumerate(deltas):
    if v < 0:
        ax.text(v - 0.05, i, f"{v}%", va='center', ha='right', fontsize=10, color='#c0392b', fontweight='bold')
    elif v > 0:
        ax.text(v + 0.05, i, f"+{v}%", va='center', ha='left', fontsize=10, color='#27ae60', fontweight='bold')
    else:
        ax.text(v + 0.05, i, f"Baseline", va='center', ha='left', fontsize=10, color='gray', style='italic')

plt.xlim(-2.0, 1.8)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "accuracy_comparison.png"), dpi=300)
plt.close()

print("Plots generated successfully in the 'plots' directory.")
