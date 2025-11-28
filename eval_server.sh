#!/bin/bash


source /miniconda3/etc/profile.d/conda.sh && \
conda activate py312

path="PATH TO THE TRAJ FOLDER"

current_time=$(date "+%Y%m%d_%H%M%S")

python get_all_runs.py --target-name all_runs_04_09

# remove the preds.json file if it exists
rm -f $path/preds.json

# 1. summarize all preds file in each subfolder into 1 file
python summarize_preds.py "$path"

python filter_patches.py "$path/preds.json" "$path/filtered_preds.json"

2. run the evaluation
sb-cli submit swe-bench_lite  test \
    --predictions_path $path/filtered_preds.json \
    --run_id all_runs_eval_rerun_${current_time}