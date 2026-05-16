"""Shared constants for gate training and inference."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS_DIR = (
    PROJECT_ROOT / "output" / "gsm8k_semantic_distractor" / "mechanism_ablation"
)
DEFAULT_GATE_ARTIFACT_DIR = (
    PROJECT_ROOT / "output" / "gsm8k_semantic_distractor" / "gate_artifacts"
)

COMMIT_LABELS = ["submit", "one_more_turn", "fallback_cot"]
COMMIT_LABELS_DEPLOY = ["submit", "one_more_turn"]
COMMIT_BINARY_LABELS = ["submit", "one_more_turn"]
STEP_LABELS = ["continue", "commit"]

MAX_EXTRA_TURNS_GSTEP = 3
MAX_ONE_MORE_TURN_GLOBAL = 1

N_SPLITS_GROUP_KFOLD = 5
RANDOM_SEED = 42

FEATURE_NAMES = [
    "turn_index",
    "successful_calls",
    "error_count",
    "budget_remaining",
    "last_two_output_diff",
    "output_stagnation",
    "output_is_numeric",
    "last_output_magnitude",
    "pred_matches_last_success",
    "pred_matches_any_tool_output",
    "last_success_in_reasoning",
    "reasoning_lastnum_vs_last_success_diff",
    "pred_in_reasoning",
    "tool_output_in_reasoning_frac",
    "reasoning_length",
    "reasoning_step_count",
    "reasoning_has_uncertainty",
    "n_numbers_in_reasoning",
    "n_chunks_seen",
    "n_pred_evidence_ids",
    "evidence_coverage_ratio",
    "n_distinct_outputs",
    "all_outputs_same",
    "expression_complexity",
]
