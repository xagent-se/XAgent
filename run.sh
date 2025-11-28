source /miniconda3/etc/profile.d/conda.sh && \
conda activate py312 && \
sweagent run-batch \
  --config config/patch_generation.yaml \
  --instances.type swe_bench \
  --instances.subset lite \
  --instances.split test \
  --instances.filter "$(tr '\n' '|' < selected_issues.yml | sed 's/,$//')" \
  --num_workers 28
