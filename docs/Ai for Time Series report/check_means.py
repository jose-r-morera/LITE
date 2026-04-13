import pandas as pd
import os

base_path = '../../experimentation_results/full_ucr/no_change_reproduced_results_ucr.csv'
imp_path = '../../experimentation_results/full_ucr/gz_adamw_diff.csv'

def get_mean(path):
    df = pd.read_csv(path)
    df = df[~df['dataset'].str.upper().isin(['AVERAGE', 'AVG', 'SUM'])]
    return df['LITE-mean'].mean(), df['LITETime'].mean(), df['Training Time (s)-mean'].mean()

b_lite, b_ltime, b_time = get_mean(base_path)
i_lite, i_ltime, i_time = get_mean(imp_path)

print(f"BASELINE: LITE={b_lite:.6f}, LITETime={b_ltime:.6f}, Time={b_time:.2f}")
print(f"IMPROVED: LITE={i_lite:.6f}, LITETime={i_ltime:.6f}, Time={i_time:.2f}")
print(f"LITE Delta: {(i_lite - b_lite)*100:.4f}%")
print(f"LITETime Delta: {(i_ltime - b_ltime)*100:.4f}%")
