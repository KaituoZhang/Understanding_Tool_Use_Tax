#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_DIR="output/gsm8k_semantic_distractor/gate_artifacts/mock"
OUTPUT_DIR="${1:-output/gsm8k_semantic_distractor/mechanism_ablation/mock_gate_quick_start/}"

python -m pipeline.mechanism_ablation.gsm8k_semantic_distractor.gate.train_gate \
  --results_dir data/gsm8k_semantic_distractor/sample_gate_results \
  --out_dir "${ARTIFACT_DIR}" \
  --which g_step

python pipeline/mechanism_ablation/main.py \
  --exp_desc "gsm8k_semantic_distractor_mechanism_mock_gate_quick_start" \
  --pipeline_config_dir configs/pipeline_config/mechanism_ablation/mock/gsm8k_semantic_distractor_gate.json \
  --eval_config_dir configs/eval_config/gsm8k_semantic_distractor/default.json \
  --management_config_dir configs/management_config/default.json \
  --output_folder_dir "${OUTPUT_DIR}" \
  --overwrite allowed
