# LITE Verification & Improvement

Project as part of the MSc in Advanced AI of UCD

Goal: Verify the results from the original paper and study improvement opportunities

CHECK [Original README](README_og.md)

## Running the modified benchmark
To keep the benchmarking fast, a reduced dataset of UCR is used, comprising 32 out of the 127 datasets (25%) of the original dataset. 

Also, verbose / logging that was in the original version is disable.

The recommended fast benchmark command is:

```bash
python3 main.py --verbose="False" --dataset="fast" --track-emissions="False" --runs=3
```

## Command Line Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--dataset` | Name of the dataset to run the experiment on. Special values: `all` (runs all univariate datasets), `fast` (runs a representative subset of ~30 datasets). | `Coffee` |
| `--classifier` | The architecture to use. Options: `LITE` (original univariate), `LITEMV` (optimized multivariate). | `LITE` |
| `--runs` | Number of experimental runs to perform per dataset for calculating mean/std results. | `5` |
| `--output-directory` | Parent directory where results, plots, and models will be saved. | `results/` |
| `--track-emissions` | Boolean flag to enable/disable carbon emission tracking via CodeCarbon. | `True` |
| `--verbose` | Boolean flag to show training logs and validation tracking for every epoch. | `False` |
| `--params` | **Query Flag**: If present, the script prints the classifier's parameter count (Total, Trainable, Non-trainable) for the dataset and exits immediately. | `False` |
| `--differentiate` | Boolean flag to toggle the first-order derivative (differencing) input channel. If `True`, the model uses a 2-channel input (Raw + Delta). If `False`, it uses only Raw. | `True` |

### Notes on Boolean Flags
For boolean flags like `--track-emissions`, `--verbose`, and `--differentiate`, you can pass truthy/falsy strings (e.g., `True`, `False`, `1`, `0`, `yes`, `no`).