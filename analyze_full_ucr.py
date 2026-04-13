import pandas as pd
import numpy as np

base = pd.read_csv('experimentation_results/full_ucr/no_change_reproduced_results_ucr.csv')
improved = pd.read_csv('experimentation_results/full_ucr/gz_adamw_diff.csv')

# Remove AVERAGE/SUM rows
base = base[~base['dataset'].str.upper().isin(['AVERAGE','SUM'])]
improved = improved[~improved['dataset'].str.upper().isin(['AVERAGE','SUM'])]

print(f'Baseline datasets: {len(base)}')
print(f'Improved datasets: {len(improved)}')
print(f'Baseline mean LITE: {base["LITE-mean"].mean():.6f}')
print(f'Baseline mean LITETime: {base["LITETime"].mean():.6f}')
print(f'Improved mean LITE: {improved["LITE-mean"].mean():.6f}')
print(f'Improved mean LITETime: {improved["LITETime"].mean():.6f}')
print(f'LITE Delta: {(improved["LITE-mean"].mean() - base["LITE-mean"].mean()) * 100:.4f}%')
print(f'LITETime Delta: {(improved["LITETime"].mean() - base["LITETime"].mean()) * 100:.4f}%')

# Find common datasets
merged = base.merge(improved, on='dataset', suffixes=('_base','_imp'))
merged['delta'] = (merged['LITE-mean_imp'] - merged['LITE-mean_base']) * 100
wins = (merged['delta'] > 0).sum()
ties = (merged['delta'] == 0).sum()
losses = (merged['delta'] < 0).sum()
print(f'\nCommon datasets: {len(merged)}')
print(f'Wins: {wins}, Ties: {ties}, Losses: {losses}')
top5 = merged.nlargest(5, 'delta')[['dataset','LITE-mean_base','LITE-mean_imp','delta']]
bot5 = merged.nsmallest(5, 'delta')[['dataset','LITE-mean_base','LITE-mean_imp','delta']]
print(f'\nTop 5 improvements:\n{top5.to_string()}')
print(f'\nBottom 5:\n{bot5.to_string()}')
print(f'\nMean training time base: {base["Training Time (s)-mean"].mean():.1f}s')
print(f'Mean training time improved: {improved["Training Time (s)-mean"].mean():.1f}s')

# Datasets only in base
only_base = set(base['dataset']) - set(improved['dataset'])
only_imp = set(improved['dataset']) - set(base['dataset'])
print(f'\nOnly in baseline ({len(only_base)}): {only_base}')
print(f'Only in improved ({len(only_imp)}): {only_imp}')
