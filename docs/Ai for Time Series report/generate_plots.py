"""
Generate updated plots for the LITE research report,
including the new AdamW + Global Z-Normalization experiment
AND the full 128-dataset UCR results from the main branch.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['figure.dpi'] = 150

# ── Load all experiment CSVs ──────────────────────────────────────
base_dir = "../../experimentation_results/"

baseline   = pd.read_csv(base_dir + "fast_subset_baseline.csv")
adamw      = pd.read_csv(base_dir + "adamw_results.csv")
znorm      = pd.read_csv(base_dir + "global_znormalization.csv")
batch32    = pd.read_csv(base_dir + "batch_32.csv")
batch128   = pd.read_csv(base_dir + "batch_128.csv")
minmax     = pd.read_csv(base_dir + "global_min_max.csv")
diff_chan  = pd.read_csv(base_dir + "globalz_and_differentiated_channel.csv")
learn_filt = pd.read_csv(base_dir + "gz_diff_and_learnable_filters.csv")
adamw_diff = pd.read_csv(base_dir + "gz_diff_learn_adamw.csv")
adamw_znorm = pd.read_csv(base_dir + "adamw_znorm.csv")

# Full 128-dataset UCR results
full_baseline = pd.read_csv(base_dir + "full_ucr/no_change_reproduced_results_ucr.csv")
full_improved = pd.read_csv(base_dir + "full_ucr/gz_adamw_diff.csv")

# Drop AVERAGE/SUM rows
for df in [baseline, adamw, znorm, batch32, batch128, minmax, diff_chan,
           learn_filt, adamw_diff, adamw_znorm, full_baseline, full_improved]:
    mask = df['dataset'].str.upper().isin(['AVERAGE', 'AVG', 'SUM'])
    df.drop(df[mask].index, inplace=True)

# ── Fast subset mean accuracies ───────────────────────────────────
configs = {
    'Baseline':           baseline['LITE-mean'].mean(),
    'AdamW':              adamw['LITE-mean'].mean(),
    'Batch 32':           batch32['LITE-mean'].mean(),
    'Batch 128':          batch128['LITE-mean'].mean(),
    'Global min-max':     minmax['LITE-mean'].mean(),
    'Global z-norm':      znorm['LITE-mean'].mean(),
    'Diff. channel':      diff_chan['LITE-mean'].mean(),
    'Learnable filters':  learn_filt['LITE-mean'].mean(),
    'AdamW + Diff.':      adamw_diff['LITE-mean'].mean(),
    'AdamW + Z-norm':     adamw_znorm['LITE-mean'].mean(),
}

times = {
    'Baseline':           baseline['Training Time (s)-mean'].mean(),
    'AdamW':              adamw['Training Time (s)-mean'].mean(),
    'Batch 32':           batch32['Training Time (s)-mean'].mean(),
    'Batch 128':          batch128['Training Time (s)-mean'].mean(),
    'Global min-max':     minmax['Training Time (s)-mean'].mean(),
    'Global z-norm':      znorm['Training Time (s)-mean'].mean(),
    'Diff. channel':      diff_chan['Training Time (s)-mean'].mean(),
    'Learnable filters':  learn_filt['Training Time (s)-mean'].mean(),
    'AdamW + Diff.':      adamw_diff['Training Time (s)-mean'].mean(),
    'AdamW + Z-norm':     adamw_znorm['Training Time (s)-mean'].mean(),
}

baseline_acc = configs['Baseline']

# ═══════════════════════════════════════════════════════════════════
# PLOT 1: Performance Delta vs. Baseline (horizontal bar chart)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

labels_delta = ['AdamW', 'Batch 32', 'Batch 128', 'Global min-max',
                'Global z-norm', 'Diff. channel', 'Learnable filters',
                'AdamW + Diff.', 'AdamW + Z-norm']
deltas = [(configs[k] - baseline_acc) * 100 for k in labels_delta]
colors = ['#2ecc71' if d >= 0 else '#e74c3c' for d in deltas]

y_pos = np.arange(len(labels_delta))
bars = ax.barh(y_pos, deltas, color=colors, height=0.55, edgecolor='white')
ax.set_yticks(y_pos)
ax.set_yticklabels(labels_delta, fontsize=11)
ax.invert_yaxis()
ax.axvline(0, color='black', linewidth=1.2)
ax.set_xlabel('Accuracy Delta (%)', fontsize=12)
ax.set_title('Performance Delta vs. Baseline (30-Dataset Fast Subset)', fontsize=13, fontweight='bold')

for bar, d in zip(bars, deltas):
    sign = '+' if d >= 0 else ''
    color = '#27ae60' if d >= 0 else '#c0392b'
    x_offset = 0.05 if d >= 0 else -0.05
    ha = 'left' if d >= 0 else 'right'
    ax.text(bar.get_width() + x_offset, bar.get_y() + bar.get_height()/2,
            f'{sign}{d:.2f}%', va='center', ha=ha, fontweight='bold', color=color, fontsize=10)

ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/accuracy_comparison.png', dpi=200, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════
# PLOT 2: Accuracy vs Training Time Trade-off (scatter)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))

scatter_configs = ['Baseline', 'AdamW', 'Batch 32', 'Batch 128',
                   'Global min-max', 'Global z-norm', 'Diff. channel',
                   'Learnable filters', 'AdamW + Diff.', 'AdamW + Z-norm']
scatter_accs = [configs[k] for k in scatter_configs]
scatter_times = [times[k] for k in scatter_configs]

scatter_colors = ['#3498db'] * len(scatter_configs)
scatter_colors[-1] = '#e74c3c'
scatter_sizes = [100] * len(scatter_configs)
scatter_sizes[-1] = 180

ax.scatter(scatter_times, scatter_accs, c=scatter_colors, s=scatter_sizes, zorder=5, edgecolors='white', linewidth=1.5)

for i, label in enumerate(scatter_configs):
    offset_x, offset_y = 5, 0.001
    if label == 'AdamW + Z-norm':
        offset_x = 5
        offset_y = -0.002
    ax.annotate(label, (scatter_times[i], scatter_accs[i]),
                textcoords="offset points", xytext=(offset_x, offset_y * 1000),
                fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

ax.set_xlabel('Mean Training Time per Dataset (s)', fontsize=12)
ax.set_ylabel('Mean Accuracy (LITE)', fontsize=12)
ax.set_title('Accuracy vs. Training Time Trade-off', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('plots/time_vs_accuracy.png', dpi=200, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════
# PLOT 3: Per-dataset accuracy improvement of Diff. channel vs Baseline
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5))

merged = baseline.merge(diff_chan, on='dataset', suffixes=('_base', '_new'))
merged['delta'] = (merged['LITE-mean_new'] - merged['LITE-mean_base']) * 100
merged = merged.sort_values('delta', ascending=False)

wins = (merged['delta'] > 0).sum()
losses = (merged['delta'] < 0).sum()
ties = (merged['delta'] == 0).sum()

colors_per = ['#2ecc71' if d >= 0 else '#e74c3c' for d in merged['delta']]
ax.bar(range(len(merged)), merged['delta'], color=colors_per, edgecolor='white', width=0.7)
ax.set_xticks(range(len(merged)))
ax.set_xticklabels(merged['dataset'], rotation=45, ha='right', fontsize=8)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('Accuracy Improvement (%)', fontsize=11)
ax.set_title('Per-dataset Accuracy: Diff. Channel + Z-norm vs. Baseline (Best Config)', fontsize=13, fontweight='bold')

ax.annotate(f'Improved: {wins} | Unchanged: {ties} | Decreased: {losses}\nMean Δ: {merged["delta"].mean():+.2f}%',
            xy=(0.02, 0.95), xycoords='axes fraction', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray'),
            verticalalignment='top')

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/dataset_improvement.png', dpi=200, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════
# PLOT 4: Training Time Comparison (bar chart)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

time_labels = ['Baseline', 'AdamW', 'Batch 32', 'Batch 128', 'Global z-norm', 'Diff. channel',
               'Learnable filters', 'AdamW + Diff.', 'AdamW + Z-norm']
time_vals = [times[k] for k in time_labels]
bar_colors = ['#3498db'] * len(time_labels)
bar_colors[-1] = '#e74c3c'

bars = ax.bar(range(len(time_labels)), time_vals, color=bar_colors, edgecolor='white', width=0.6)
ax.set_xticks(range(len(time_labels)))
ax.set_xticklabels(time_labels, rotation=25, ha='right', fontsize=9)
ax.set_ylabel('Mean Training Time per Dataset (s)', fontsize=11)
ax.set_title('Training Time Comparison Across Configurations', fontsize=13, fontweight='bold')

for bar, val in zip(bars, time_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            f'{val:.0f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/training_time_comparison.png', dpi=200, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════
# PLOT 5: Box plot comparing accuracy distributions (fast subset)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))

box_data = [
    baseline['LITE-mean'].values,
    adamw['LITE-mean'].values,
    znorm['LITE-mean'].values,
    diff_chan['LITE-mean'].values,
    learn_filt['LITE-mean'].values,
    adamw_diff['LITE-mean'].values,
    adamw_znorm['LITE-mean'].values,
]
box_labels = ['Baseline', 'AdamW', 'Global z-norm', 'Diff. channel',
              'Learnable\nfilters', 'AdamW +\nDiff.', 'AdamW +\nZ-norm']
box_colors = ['#95a5a6', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e74c3c']

bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, widths=0.5,
                medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Accuracy Distribution Across Configurations (30 Datasets)', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/univariate_boxplot.png', dpi=200, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════
# PLOT 6: FULL UCR - Top/Bottom 10 improvements (readable at column width)
# ═══════════════════════════════════════════════════════════════════
full_merged = full_baseline.merge(full_improved, on='dataset', suffixes=('_base', '_imp'))
full_merged['delta'] = (full_merged['LITE-mean_imp'] - full_merged['LITE-mean_base']) * 100
full_merged = full_merged.sort_values('delta', ascending=False)

wins_full = (full_merged['delta'] > 0).sum()
losses_full = (full_merged['delta'] < 0).sum()
ties_full = (full_merged['delta'] == 0).sum()
mean_delta = full_merged['delta'].mean()

top10 = full_merged.head(10)
bot10 = full_merged.tail(10).iloc[::-1]  # reverse so worst is at bottom
show = pd.concat([top10, bot10])

fig, ax = plt.subplots(figsize=(8, 7))
y_pos = np.arange(len(show))
colors_h = ['#2ecc71' if d >= 0 else '#e74c3c' for d in show['delta']]

bars = ax.barh(y_pos, show['delta'], color=colors_h, height=0.6, edgecolor='white')
ax.set_yticks(y_pos)
ax.set_yticklabels(show['dataset'], fontsize=8)
ax.invert_yaxis()
ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel('Accuracy Improvement (%)', fontsize=10)
ax.set_title('Full UCR (128 Datasets): Top & Bottom 10', fontsize=12, fontweight='bold')

# Add a divider line between top10 and bot10
ax.axhline(9.5, color='gray', linewidth=0.8, linestyle='--')

# Value labels on bars
for bar, d in zip(bars, show['delta']):
    sign = '+' if d >= 0 else ''
    color = '#1a7a3a' if d >= 0 else '#8b1a1a'
    x_off = 0.3 if d >= 0 else -0.3
    ha = 'left' if d >= 0 else 'right'
    ax.text(bar.get_width() + x_off, bar.get_y() + bar.get_height()/2,
            f'{sign}{d:.1f}%', va='center', ha=ha, fontsize=7, fontweight='bold', color=color)

# Summary annotation
ax.annotate(f'Overall: {wins_full} improved, {ties_full} unchanged, {losses_full} decreased\n'
            f'Mean accuracy delta: {mean_delta:+.2f}%',
            xy=(0.98, 0.5), xycoords='axes fraction', fontsize=8, ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/full_ucr_improvement.png', dpi=200, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════
# PLOT 7: FULL UCR - Box plot comparing baseline vs improved
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

box_data_full = [
    full_baseline['LITE-mean'].values,
    full_improved['LITE-mean'].values,
    full_baseline['LITETime'].values,
    full_improved['LITETime'].values,
]
box_labels_full = ['LITE\n(Baseline)', 'LITE\n(Improved)', 'LITETime\n(Baseline)', 'LITETime\n(Improved)']
box_colors_full = ['#95a5a6', '#e74c3c', '#bdc3c7', '#c0392b']

bp = ax.boxplot(box_data_full, tick_labels=box_labels_full, patch_artist=True, widths=0.5,
                medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp['boxes'], box_colors_full):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Full UCR Archive: Baseline vs. Best Improvement (128 Datasets)', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add mean lines
for i, data in enumerate(box_data_full, 1):
    ax.scatter(i, data.mean(), color='gold', marker='D', s=60, zorder=5, edgecolors='black', linewidth=0.8)

ax.legend(['Mean'], loc='lower right')
plt.tight_layout()
plt.savefig('plots/full_ucr_boxplot.png', dpi=200, bbox_inches='tight')
plt.close()


# ═══════════════════════════════════════════════════════════════════
# Print summary
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FAST SUBSET SUMMARY (30 datasets)")
print("="*70)
print(f"{'Configuration':<25} {'Mean Accuracy':>15} {'Delta':>10}")
print("-"*50)
for k in ['Baseline', 'AdamW', 'Batch 32', 'Batch 128', 'Global min-max',
          'Global z-norm', 'Diff. channel', 'Learnable filters',
          'AdamW + Diff.', 'AdamW + Z-norm']:
    delta = (configs[k] - baseline_acc) * 100
    sign = '+' if delta >= 0 else ''
    print(f"{k:<25} {configs[k]:>15.4f} {sign}{delta:>9.2f}%")

print("\n" + "="*70)
print("FULL UCR SUMMARY (128 datasets)")
print("="*70)
base_lite = full_baseline['LITE-mean'].mean()
imp_lite = full_improved['LITE-mean'].mean()
base_litetime = full_baseline['LITETime'].mean()
imp_litetime = full_improved['LITETime'].mean()
print(f"Baseline LITE mean:    {base_lite:.4f}")
print(f"Improved LITE mean:    {imp_lite:.4f}  (delta: {(imp_lite - base_lite)*100:+.2f}%)")
print(f"Baseline LITETime mean:{base_litetime:.4f}")
print(f"Improved LITETime mean:{imp_litetime:.4f}  (delta: {(imp_litetime - base_litetime)*100:+.2f}%)")
print(f"Wins: {wins_full}, Ties: {ties_full}, Losses: {losses_full}")
print(f"Mean training time base: {full_baseline['Training Time (s)-mean'].mean():.1f}s")
print(f"Mean training time improved: {full_improved['Training Time (s)-mean'].mean():.1f}s")

print("\n\nAll plots saved to plots/ directory.")
