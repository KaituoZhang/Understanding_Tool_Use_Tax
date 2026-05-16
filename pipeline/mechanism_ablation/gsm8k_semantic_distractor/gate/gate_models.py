"""Inference wrappers for protocol-internal gates."""

import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..metrics import answers_equal
from .configs import MAX_EXTRA_TURNS_GSTEP, MAX_ONE_MORE_TURN_GLOBAL
from .feature_engineering import (
    build_commit_features_from_record,
    build_step_features_online,
    last_success_output,
    successful_outputs,
)
from .learned_feature_builder import (
    build_commit_features_aug_online,
    build_step_features_aug_online,
)
from .simple_models import LinearGateModel


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _restore_model(model_payload: Any) -> Any:
    if isinstance(model_payload, dict):
        if model_payload.get("type") == "linear_gate_model_v1":
            return LinearGateModel.from_dict(model_payload)
    return model_payload


def _load_artifact(path: str) -> Dict[str, Any]:
    artifact_path = Path(path)
    if artifact_path.suffix.lower() == ".json":
        with artifact_path.open("r", encoding="utf-8") as handle:
            artifact = json.load(handle)
    else:
        with artifact_path.open("rb") as handle:
            artifact = pickle.load(handle)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("Invalid gate artifact format: {0}".format(path))
    artifact = dict(artifact)
    artifact["model"] = _restore_model(artifact.get("model"))
    return artifact


def _predict_proba(model: Any, x: List[float], positive_class: Any) -> float:
    row = [x]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row)[0]
        classes_ = getattr(model, "classes_", None)
        if classes_ is not None:
            for i, cls in enumerate(classes_):
                if str(cls) == str(positive_class):
                    return float(proba[i])
        return float(proba[1] if len(proba) > 1 else proba[0])
    if hasattr(model, "decision_function"):
        score = model.decision_function(row)
        if isinstance(score, list):
            score = score[0]
        if isinstance(score, (tuple, list)):
            score = score[0]
        return float(_sigmoid(float(score)))
    pred = model.predict(row)[0]
    return 1.0 if str(pred) == str(positive_class) else 0.0


def _extract_pred_from_candidate_text(text: str) -> str:
    if not text:
        return ""
    match = re.search(
        r'(?is)["\']?final[_\s]?answer["\']?\s*[:=]\s*["\']?([^"\',}\n]+)',
        text,
    )
    if match:
        return match.group(1).strip()

    match = re.search(
        r'(?is)<\s*(?:answer|result|final_answer)\s*>\s*([^<>]{1,80})\s*'
        r'<\s*/\s*(?:answer|result|final_answer)\s*>',
        text,
    )
    if match:
        return match.group(1).strip()

    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return numbers[-1] if numbers else ""


@dataclass
class StepDecision:
    action: str
    p_continue: float


@dataclass
class CommitDecision:
    action: str
    pred_answer: str
    p_one_more: float
    anchor_applied: bool
    last_success: str
    one_more_used: bool


class GStepGate:
    """Learned step gate: continue vs commit."""

    def __init__(
        self,
        model: Any,
        threshold: float = 0.05,
        max_extra_turns: int = MAX_EXTRA_TURNS_GSTEP,
        positive_class: Any = "continue",
        feature_spec: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.threshold = float(threshold)
        self.max_extra_turns = int(max_extra_turns)
        self.positive_class = positive_class
        self.feature_spec = feature_spec or {}

    @classmethod
    def from_artifact(
        cls,
        path: str,
        threshold: float = 0.05,
        max_extra_turns: int = MAX_EXTRA_TURNS_GSTEP,
    ) -> "GStepGate":
        artifact = _load_artifact(path)
        return cls(
            model=artifact["model"],
            threshold=threshold,
            max_extra_turns=max_extra_turns,
            positive_class=artifact.get("positive_class", "continue"),
            feature_spec=artifact.get("feature_spec", {}),
        )

    def decide(
        self,
        *,
        tool_trace: List[Dict[str, Any]],
        candidate_text: str,
        pred_answer: str = "",
        n_chunks_seen: int = 0,
        max_tool_calls_limit: int = 5,
        extra_turns_used: int = 0,
    ) -> StepDecision:
        if extra_turns_used >= self.max_extra_turns:
            return StepDecision(action="commit", p_continue=0.0)
        if len(tool_trace or []) >= max_tool_calls_limit:
            return StepDecision(action="commit", p_continue=0.0)
        if (not pred_answer) and candidate_text:
            pred_answer = _extract_pred_from_candidate_text(candidate_text)

        if str(self.feature_spec.get("type", "")) == "aug_v1":
            features = build_step_features_aug_online(
                tool_trace=tool_trace or [],
                candidate_text=candidate_text or "",
                pred_answer=pred_answer or "",
                n_chunks_seen=n_chunks_seen,
                max_tool_calls_limit=max_tool_calls_limit,
                feature_spec=self.feature_spec,
            )
        else:
            features = build_step_features_online(
                tool_trace=tool_trace or [],
                candidate_text=candidate_text or "",
                pred_answer=pred_answer or "",
                n_chunks_seen=n_chunks_seen,
                max_tool_calls_limit=max_tool_calls_limit,
            )
        probability = _predict_proba(self.model, features, self.positive_class)
        action = "continue" if probability >= self.threshold else "commit"
        return StepDecision(action=action, p_continue=float(probability))


class GCommitGate:
    """Commit gate with a learned one-more-turn classifier."""

    def __init__(
        self,
        model: Optional[Any] = None,
        threshold: float = 0.5,
        positive_class: Any = "one_more_turn",
        max_one_more_turn: int = MAX_ONE_MORE_TURN_GLOBAL,
        feature_spec: Optional[Dict[str, Any]] = None,
        enable_safe_filters: bool = True,
        max_tool_calls_for_one_more: int = 2,
    ):
        self.model = model
        self.threshold = float(threshold)
        self.positive_class = positive_class
        self.max_one_more_turn = int(max_one_more_turn)
        self.feature_spec = feature_spec or {}
        self.enable_safe_filters = bool(enable_safe_filters)
        self.max_tool_calls_for_one_more = int(max(0, max_tool_calls_for_one_more))

    @classmethod
    def from_artifact(cls, path: str, threshold: float = 0.5) -> "GCommitGate":
        artifact = _load_artifact(path)
        return cls(
            model=artifact.get("model"),
            threshold=threshold,
            positive_class=artifact.get("positive_class", "one_more_turn"),
            max_one_more_turn=int(artifact.get("max_one_more_turn", MAX_ONE_MORE_TURN_GLOBAL)),
            feature_spec=artifact.get("feature_spec", {}),
            enable_safe_filters=bool(artifact.get("enable_safe_filters", True)),
            max_tool_calls_for_one_more=int(artifact.get("max_tool_calls_for_one_more", 2)),
        )

    def decide(
        self,
        *,
        pred_answer: str,
        reasoning: str,
        tool_trace: List[Dict[str, Any]],
        n_chunks_seen: int = 0,
        pred_evidence_ids: Optional[List[int]] = None,
        expression: str = "",
        max_tool_calls_limit: int = 5,
        one_more_used: int = 0,
    ) -> CommitDecision:
        pred_answer = str(pred_answer or "").strip()
        reasoning = str(reasoning or "").strip()
        pred_evidence_ids = pred_evidence_ids or []
        tool_trace = tool_trace or []
        last_success = last_success_output(tool_trace or [])

        if one_more_used >= self.max_one_more_turn:
            return CommitDecision(
                action="submit",
                pred_answer=pred_answer,
                p_one_more=0.0,
                anchor_applied=False,
                last_success=last_success,
                one_more_used=True,
            )

        if self.model is None:
            return CommitDecision(
                action="submit",
                pred_answer=pred_answer,
                p_one_more=0.0,
                anchor_applied=False,
                last_success=last_success,
                one_more_used=bool(one_more_used),
            )

        if str(self.feature_spec.get("type", "")) == "aug_v1":
            features = build_commit_features_aug_online(
                pred_answer=pred_answer,
                reasoning=reasoning,
                tool_trace=tool_trace,
                n_chunks_seen=n_chunks_seen,
                pred_evidence_ids=pred_evidence_ids,
                expression=expression,
                max_tool_calls_limit=max_tool_calls_limit,
                feature_spec=self.feature_spec,
            )
        else:
            record = {
                "pred_answer": pred_answer,
                "reasoning": reasoning,
                "tool_trace": tool_trace,
                "n_chunks_seen": n_chunks_seen,
                "pred_evidence_ids": pred_evidence_ids,
                "expression": expression,
            }
            features = build_commit_features_from_record(
                record,
                max_tool_calls_limit=max_tool_calls_limit,
            )

        probability = _predict_proba(self.model, features, self.positive_class)
        action = "one_more_turn" if probability >= self.threshold else "submit"

        if action == "one_more_turn" and self.enable_safe_filters:
            successful = successful_outputs(tool_trace or [])
            tool_calls = len(tool_trace or [])
            has_tool_error = any(
                str(step.get("output", "")).startswith("Error")
                for step in (tool_trace or [])
            )
            pred_matches_last = bool(
                pred_answer and last_success and answers_equal(pred_answer, last_success)
            )
            stagnation_last2 = len(successful) >= 2 and (successful[-1] == successful[-2])
            stagnation_all = len(successful) >= 2 and (len(set(successful)) == 1)

            too_many_calls = (
                tool_calls > self.max_tool_calls_for_one_more
                and (not has_tool_error)
            )
            repeated_wrong_chain = (
                pred_matches_last
                and tool_calls >= 2
                and (stagnation_last2 or stagnation_all)
            )

            if too_many_calls or repeated_wrong_chain:
                action = "submit"

        return CommitDecision(
            action=action,
            pred_answer=pred_answer,
            p_one_more=float(probability),
            anchor_applied=False,
            last_success=last_success,
            one_more_used=bool(one_more_used),
        )
