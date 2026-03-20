# LITE Verification & Improvement

Project as part of the MSc in Advanced AI of UCD

Goal: Verify the results from the original paper and study improvement opportunities

CHECK [Original README](README_og.md)

## Running the modified benchmark
To keep the benchmarking fast, a reduced dataset of UCR is used, comprising 32 out of the 127 datasets (25%) of the original dataset. 

Also, verbose / logging that was in the original version is disable.

The recommended fast benchmark command is:

python3 main.py --verbose="False" --dataset="fast" --track-emissions="False" --runs=3