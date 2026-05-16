"""
Feature engineering for protocol-internal gates.

All features are inference-safe and do not use gold answers.
"""

import math
import re
from typing import Any, Dict, List, Optional

from ..metrics import answers_equal, normalize_to_number
from .configs import FEATURE_NAMES


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_UNCERTAIN_RE = re.compile(
    r"\b(maybe|might|possibly|not sure|uncertain|guess|probably|approximately|around)\b",
    re.IGNORECASE,
)
_STEP_RE = re.compile(
    r"\b(step\s*\d+|then|next|finally|after that|first|second|third)\b",
    re.IGNORECASE,
)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        casted = float(value)
        if math.isfinite(casted):
            return casted
    except Exception:
        pass
    return default


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def last_success_output(tool_trace: List[Dict[str, Any]]) -> str:
    for step in reversed(tool_trace or []):
        output = _as_text(step.get("output", ""))
        if not output.startswith("Error") and not output.startswith("[CALCULATOR"):
            return output
    return ""


def successful_outputs(tool_trace: List[Dict[str, Any]]) -> List[str]:
    outputs = []
    for step in tool_trace or []:
        output = _as_text(step.get("output", ""))
        if not output.startswith("Error") and not output.startswith("[CALCULATOR"):
            outputs.append(output)
    return outputs


def _reasoning_last_number(reasoning: str) -> Optional[float]:
    numbers = _NUM_RE.findall(reasoning or "")
    if not numbers:
        return None
    try:
        return float(numbers[-1])
    except Exception:
        return None


def _text_contains_numeric(text: str, value_text: str) -> bool:
    if not text or not value_text:
        return False
    if value_text in text:
        return True
    numeric_value = normalize_to_number(value_text)
    if numeric_value is None:
        return False
    for match in _NUM_RE.findall(text):
        try:
            if abs(float(match) - numeric_value) < 1e-9:
                return True
        except Exception:
            continue
    return False


def _expression_complexity(expr: str) -> float:
    if not expr:
        return 0.0
    operators = sum(expr.count(op) for op in ["+", "-", "*", "/", "%", "^"])
    separators = expr.count(";") + expr.count("|")
    parens = expr.count("(") + expr.count(")")
    return float(operators + separators + parens)


def build_feature_dict(
    *,
    pred_answer: str,
    reasoning: str,
    tool_trace: List[Dict[str, Any]],
    n_chunks_seen: int = 0,
    pred_evidence_ids: Optional[List[int]] = None,
    expression: str = "",
    max_tool_calls_limit: int = 5,
) -> Dict[str, float]:
    pred_answer = _as_text(pred_answer)
    reasoning = _as_text(reasoning)
    pred_evidence_ids = pred_evidence_ids or []
    tool_trace = tool_trace or []

    successful = successful_outputs(tool_trace)
    last_success = last_success_output(tool_trace)
    last_success_num = normalize_to_number(last_success)

    tool_calls = len(tool_trace)
    successful_calls = len(successful)
    error_calls = sum(
        1 for step in tool_trace if _as_text(step.get("output", "")).startswith("Error")
    )

    if len(successful) >= 2:
        earlier, later = successful[-2], successful[-1]
        earlier_num, later_num = normalize_to_number(earlier), normalize_to_number(later)
        if earlier_num is not None and later_num is not None:
            last_two_diff = abs(earlier_num - later_num)
        else:
            last_two_diff = 0.0 if earlier == later else 1.0
        stagnation = 1.0 if earlier == later else 0.0
    else:
        last_two_diff = 0.0
        stagnation = 0.0

    pred_matches_last = 1.0 if (pred_answer and last_success and answers_equal(pred_answer, last_success)) else 0.0
    pred_matches_any = 1.0 if (pred_answer and any(answers_equal(pred_answer, output) for output in successful)) else 0.0
    last_in_reasoning = 1.0 if _text_contains_numeric(reasoning, last_success) else 0.0
    reasoning_last_num = _reasoning_last_number(reasoning)
    if reasoning_last_num is not None and last_success_num is not None:
        reasoning_last_diff = abs(reasoning_last_num - last_success_num)
    else:
        reasoning_last_diff = 0.0
    pred_in_reasoning = 1.0 if _text_contains_numeric(reasoning, pred_answer) else 0.0
    if successful:
        matches = sum(1 for output in successful if _text_contains_numeric(reasoning, output))
        output_in_reasoning_frac = _safe_div(float(matches), float(len(successful)))
    else:
        output_in_reasoning_frac = 0.0

    reasoning_words = len(reasoning.split()) if reasoning else 0
    reasoning_steps = len(_STEP_RE.findall(reasoning)) + reasoning.count(";")
    reasoning_uncertainty = 1.0 if _UNCERTAIN_RE.search(reasoning or "") else 0.0
    reasoning_num_count = len(_NUM_RE.findall(reasoning or ""))

    distinct_outputs = len(set(successful))
    all_outputs_same = 1.0 if (len(successful) > 0 and distinct_outputs == 1) else 0.0

    return {
        "turn_index": float(tool_calls),
        "successful_calls": float(successful_calls),
        "error_count": float(error_calls),
        "budget_remaining": float(max(0, max_tool_calls_limit - tool_calls)),
        "last_two_output_diff": float(last_two_diff),
        "output_stagnation": float(stagnation),
        "output_is_numeric": 1.0 if last_success_num is not None else 0.0,
        "last_output_magnitude": float(abs(last_success_num) if last_success_num is not None else 0.0),
        "pred_matches_last_success": float(pred_matches_last),
        "pred_matches_any_tool_output": float(pred_matches_any),
        "last_success_in_reasoning": float(last_in_reasoning),
        "reasoning_lastnum_vs_last_success_diff": float(reasoning_last_diff),
        "pred_in_reasoning": float(pred_in_reasoning),
        "tool_output_in_reasoning_frac": float(output_in_reasoning_frac),
        "reasoning_length": float(reasoning_words),
        "reasoning_step_count": float(reasoning_steps),
        "reasoning_has_uncertainty": float(reasoning_uncertainty),
        "n_numbers_in_reasoning": float(reasoning_num_count),
        "n_chunks_seen": float(n_chunks_seen),
        "n_pred_evidence_ids": float(len(pred_evidence_ids)),
        "evidence_coverage_ratio": float(
            _safe_div(float(len(pred_evidence_ids)), float(max(1, n_chunks_seen)))
        ),
        "n_distinct_outputs": float(distinct_outputs),
        "all_outputs_same": float(all_outputs_same),
        "expression_complexity": float(_expression_complexity(expression)),
    }


def feature_vector_from_dict(
    feat: Dict[str, float],
    feature_names: Optional[List[str]] = None,
) -> List[float]:
    names = feature_names or FEATURE_NAMES
    return [_to_float(feat.get(name, 0.0), 0.0) for name in names]


def build_commit_features_from_record(
    record: Dict[str, Any],
    max_tool_calls_limit: int = 5,
) -> List[float]:
    feat = build_feature_dict(
        pred_answer=_as_text(record.get("pred_answer", "")),
        reasoning=_as_text(record.get("reasoning", "")),
        tool_trace=record.get("tool_trace", []) or [],
        n_chunks_seen=int(record.get("n_chunks_seen", 0) or 0),
        pred_evidence_ids=record.get("pred_evidence_ids", []) or [],
        expression=_as_text(record.get("expression", "")),
        max_tool_calls_limit=max_tool_calls_limit,
    )
    return feature_vector_from_dict(feat)


def build_step_features_online(
    *,
    tool_trace: List[Dict[str, Any]],
    candidate_text: str,
    pred_answer: str = "",
    n_chunks_seen: int = 0,
    max_tool_calls_limit: int = 5,
) -> List[float]:
    feat = build_feature_dict(
        pred_answer=pred_answer,
        reasoning=_as_text(candidate_text),
        tool_trace=tool_trace or [],
        n_chunks_seen=n_chunks_seen,
        pred_evidence_ids=[],
        expression="",
        max_tool_calls_limit=max_tool_calls_limit,
    )
    return feature_vector_from_dict(feat)
