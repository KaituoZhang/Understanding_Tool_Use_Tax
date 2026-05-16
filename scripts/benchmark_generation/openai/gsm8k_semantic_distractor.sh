#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/gsm8k_semantic_distractor/benchmark_generation/openai_run/}"

python pipeline/benchmark_generation/main.py \
  --exp_desc "gsm8k_semantic_distractor_benchmark_openai" \
  --pipeline_config_dir configs/pipeline_config/benchmark_generation/openai/gsm8k_semantic_distractor.json \
  --eval_config_dir configs/eval_config/gsm8k_semantic_distractor/default.json \
  --management_config_dir configs/management_config/default.json \
  --output_folder_dir "${OUTPUT_DIR}" \
  --overwrite allowed
