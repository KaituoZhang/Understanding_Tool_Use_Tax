#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/gsm8k_semantic_distractor/mechanism_ablation/openai_gate_run/}"

python pipeline/mechanism_ablation/main.py \
  --exp_desc "gsm8k_semantic_distractor_mechanism_openai_gate" \
  --pipeline_config_dir configs/pipeline_config/mechanism_ablation/openai/gsm8k_semantic_distractor_gate.json \
  --eval_config_dir configs/eval_config/gsm8k_semantic_distractor/default.json \
  --management_config_dir configs/management_config/default.json \
  --output_folder_dir "${OUTPUT_DIR}" \
  --overwrite allowed
