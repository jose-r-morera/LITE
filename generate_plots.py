import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob

# Set global aesthetics using modern styling
sns.set_theme(style="whitegrid", rc={"axes.spines.top": False, "axes.spines.right": False})

plot_dir = r"c:\Users\xxerp\Desktop\Career\Proyectos\GitHub\LITE\plots"
os.makedirs(plot_dir, exist_ok=True)

exp_dir = r"c:\Users\xxerp\Desktop\Career\Proyectos\GitHub\LITE\experimentation_results"

# Mapping human-readable names to file names
configs = {
    'Baseline': 'fast_subset_baseline.csv',
    'AdamW': 'adamw_results.csv',
    'Batch 32': 'batch_32.csv',
    'Batch 128': 'batch_128.csv',
    'Global min-max': 'global_min_max.csv',
    'Global z-norm': 'global_znormalization.csv',
    'Differenced channel*': 'derived_input_channel.csv'
}

data = {}
for name, file in configs.items():
    df = pd.read_csv(os.path.join(exp_dir, file))
    df['dataset_clean'] = df['dataset'].astype(str).str.strip().str.upper()
    avg_row = df[df['dataset_clean'] == 'AVERAGE']
    if not avg_row.empty:
        lite_mean = avg_row['LITE-mean'].values[0]
        litetime = avg_row['LITETime'].values[0]
        time_mean = avg_row['Training Time (s)-mean'].values[0]
        data[name] = {'LITE': lite_mean, 'LITETime': litetime, 'Time': time_mean, 'df': df}

# Extract baseline metrics
baseline_lite = data['Baseline']['LITE']
baseline_litetime = data['Baseline']['LITETime']

print("--- TABLE DATA FOR LATEX ---")
for name in configs.keys():
    delta_lite = (data[name]['LITE'] - baseline_lite) * 100
    delta_litetime = (data[name]['LITETime'] - baseline_litetime) * 100
    delta_str_lite = f"+{delta_lite:.2f}\\%" if delta_lite > 0 else f"{delta_lite:.2f}\\%"
    delta_str_litetime = f"+{delta_litetime:.2f}\\%" if delta_litetime > 0 else f"{delta_litetime:.2f}\\%"
    if name == 'Baseline':
        delta_str_lite = "---"
        delta_str_litetime = "---"
        
    print(f"{name} & {data[name]['LITE']:.4f} & {delta_str_lite} & {data[name]['LITETime']:.4f} & {delta_str_litetime} & {data[name]['Time']:.1f} \\\\")

# ---------------------------------------------------------
# 1. Performance Delta Plot (accuracy_comparison.png)
# ---------------------------------------------------------
deltas = []
config_names = []
for name in configs.keys():
    if name == 'Baseline': continue
    diff = (data[name]['LITE'] - baseline_lite) * 100
    deltas.append(diff)
    config_names.append(name)

plt.figure(figsize=(9, 5))
c_map = ['#e74c3c' if d < 0 else '#2ecc71' for d in deltas]
ax = sns.barplot(x=deltas, y=config_names, palette=c_map)

plt.title("Performance Delta vs. Baseline (31-Dataset Fast Subset)", fontsize=14, pad=15)
plt.xlabel("Accuracy Delta (%)", fontsize=12)
plt.ylabel("")
plt.axvline(x=0, color='black', linewidth=1.5)

for i, v in enumerate(deltas):
    if v < 0:
        ax.text(v - 0.05, i, f"{v:.2f}%", va='center', ha='right', fontsize=10, color='#c0392b', fontweight='bold')
    else:
        ax.text(v + 0.05, i, f"+{v:.2f}%", va='center', ha='left', fontsize=10, color='#27ae60', fontweight='bold')

plt.xlim(min(deltas) - 0.5, max(deltas) + 0.5)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "accuracy_comparison.png"), dpi=300)
plt.close()

# ---------------------------------------------------------
# 2. Time vs Accuracy Plot (time_vs_accuracy.png)
# ---------------------------------------------------------
plt.figure(figsize=(9, 6))

times = [data[name]['Time'] for name in configs.keys()]
accuracies = [data[name]['LITE'] for name in configs.keys()]

sns.scatterplot(x=times, y=accuracies, s=150, color='#3498db', edgecolor='black', zorder=5)

for name in configs.keys():
    plt.annotate(
        name,
        (data[name]['Time'], data[name]['LITE']),
        xytext=(8, -5),
        textcoords='offset points',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5, alpha=0.8)
    )

plt.title("Accuracy vs. Training Time Trade-off", fontsize=14, pad=15)
plt.xlabel("Mean Training Time per Dataset (s)", fontsize=12)
plt.ylabel("Mean Accuracy (LITE)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6, zorder=0)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "time_vs_accuracy.png"), dpi=300)
plt.close()

# ---------------------------------------------------------
# 3. Dataset Improvement Plot (dataset_improvement.png)
# ---------------------------------------------------------
df_base = data['Baseline']['df']
df_adamw = data['AdamW']['df']

# exclude AVERAGE row
df_base = df_base[df_base['dataset'] != 'AVERAGE']
df_adamw = df_adamw[df_adamw['dataset'] != 'AVERAGE']

df_joined = pd.merge(df_base, df_adamw, on='dataset', suffixes=('_base', '_adamw'))
df_joined['Improvement'] = (df_joined['LITE-mean_adamw'] - df_joined['LITE-mean_base']) * 100
df_joined = df_joined.sort_values(by='Improvement', ascending=False)

plt.figure(figsize=(12, 6))
c_map_dataset = ['#2ecc71' if x > 0 else ('#e74c3c' if x < 0 else '#95a5a6') for x in df_joined['Improvement']]
sns.barplot(x='dataset', y='Improvement', data=df_joined, palette=c_map_dataset)
plt.title("Per-dataset Accuracy Improvement of AdamW vs. Baseline", fontsize=14, pad=15)
plt.xlabel("", fontsize=12)
plt.ylabel("Accuracy Improvement (%)", fontsize=12)
plt.xticks(rotation=90, fontsize=9)
plt.axhline(y=0, color='black', linewidth=1)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "dataset_improvement.png"), dpi=300)
plt.close()

print("Dynamic plots generated successfully!")
