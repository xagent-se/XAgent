#!/bin/bash


source /miniconda3/etc/profile.d/conda.sh && \
conda activate py312

path="PATH TO THE TRAJ FOLDER"

python get_all_runs.py

# remove the preds.json file if it exists
rm -f $path/preds.json

# 1. summarize all preds file in each subfolder into 1 file
python summarize_preds.py "$path"

# 2. run the evaluation
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path $path/preds.json \
    --max_workers 28 \
    --run_id all_runs_eval_rerun