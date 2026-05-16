"""
Augmented feature builders for learned protocol gates.

These extend the numeric gate features with hashed text bins while keeping the
representation inference-safe.
"""

import hashlib
import math
import re
from typing import Any, Dict, List, Optional

from .feature_engineering import (
    build_commit_features_from_record,
    build_step_features_online,
    last_success_output,
    successful_outputs,
)


_TOK_RE = re.compile(r"[A-Za-z0-9_./%-]+")


DEFAULT_AUG_SPEC: Dict[str, Any] = {
    "type": "aug_v1",
    "step_text_bins": 64,
    "step_trace_bins": 32,
    "commit_reason_bins": 64,
    "commit_trace_bins": 32,
    "commit_expr_bins": 16,
}


def _stable_hash(token: str) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest, 16)


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [token.lower() for token in _TOK_RE.findall(text)]


def _hash_bow(text: str, n_bins: int) -> List[float]:
    n_bins = int(max(1, n_bins))
    vec = [0.0] * n_bins
    for token in _tokenize(text):
        vec[_stable_hash(token) % n_bins] += 1.0
    norm = math.sqrt(sum(value * value for value in vec))
    if norm > 0:
        vec = [value / norm for value in vec]
    return vec


def _spec_int(spec: Optional[Dict[str, Any]], key: str, default: int) -> int:
    if not spec:
        return default
    try:
        return int(spec.get(key, default))
    except Exception:
        return default


def build_step_features_aug_online(
    *,
    tool_trace: List[Dict[str, Any]],
    candidate_text: str,
    pred_answer: str,
    n_chunks_seen: int,
    max_tool_calls_limit: int,
    feature_spec: Optional[Dict[str, Any]] = None,
) -> List[float]:
    text_bins = _spec_int(feature_spec, "step_text_bins", DEFAULT_AUG_SPEC["step_text_bins"])
    trace_bins = _spec_int(feature_spec, "step_trace_bins", DEFAULT_AUG_SPEC["step_trace_bins"])

    base = build_step_features_online(
        tool_trace=tool_trace or [],
        candidate_text=candidate_text or "",
        pred_answer=pred_answer or "",
        n_chunks_seen=n_chunks_seen,
        max_tool_calls_limit=max_tool_calls_limit,
    )
    last_output = last_success_output(tool_trace or [])
    return base + _hash_bow(candidate_text or "", text_bins) + _hash_bow(last_output or "", trace_bins)


def build_commit_features_aug_online(
    *,
    pred_answer: str,
    reasoning: str,
    tool_trace: List[Dict[str, Any]],
    n_chunks_seen: int,
    pred_evidence_ids: Optional[List[int]],
    expression: str,
    max_tool_calls_limit: int,
    feature_spec: Optional[Dict[str, Any]] = None,
) -> List[float]:
    reason_bins = _spec_int(feature_spec, "commit_reason_bins", DEFAULT_AUG_SPEC["commit_reason_bins"])
    trace_bins = _spec_int(feature_spec, "commit_trace_bins", DEFAULT_AUG_SPEC["commit_trace_bins"])
    expr_bins = _spec_int(feature_spec, "commit_expr_bins", DEFAULT_AUG_SPEC["commit_expr_bins"])

    record = {
        "pred_answer": str(pred_answer or ""),
        "reasoning": str(reasoning or ""),
        "tool_trace": tool_trace or [],
        "n_chunks_seen": int(n_chunks_seen or 0),
        "pred_evidence_ids": pred_evidence_ids or [],
        "expression": str(expression or ""),
    }
    base = build_commit_features_from_record(
        record,
        max_tool_calls_limit=max_tool_calls_limit,
    )
    successful = successful_outputs(tool_trace or [])
    trace_text = " ; ".join(successful[-3:]) if successful else ""
    return (
        base
        + _hash_bow(record["reasoning"], reason_bins)
        + _hash_bow(trace_text, trace_bins)
        + _hash_bow(record["expression"], expr_bins)
    )
