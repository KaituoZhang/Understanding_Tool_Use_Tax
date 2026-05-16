"""
Label construction for protocol-internal gates.

Training-time labels may use hindsight signals, but inference-time features
remain gold-free. The paper-faithful G-STEP label path is:

1. baseline correct -> commit
2. baseline wrong + CoT correct -> continue
3. baseline wrong + tool_calls < 2 -> continue
4. otherwise -> commit
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .configs import DEFAULT_RESULTS_DIR, N_SPLITS_GROUP_KFOLD


def load_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


_MULTI_STEP_WORDS = re.compile(
    r"\b(then|next|step\s*[2-9]|second|third|finally|after\s+that|"
    r"first.*then|additionally|also\s+need)\b",
    re.IGNORECASE,
)


def _has_incompleteness_signals(record: Dict) -> bool:
    tool_calls = record.get("tool_calls", 0)
    reasoning = record.get("reasoning", "")
    tool_trace = record.get("tool_trace", [])

    if tool_calls <= 1 and _MULTI_STEP_WORDS.search(reasoning):
        return True

    error_count = sum(
        1 for step in tool_trace if str(step.get("output", "")).startswith("Error")
    )
    if error_count > 0:
        return True

    if tool_calls <= 1:
        return True

    return False


def _load_keyed(path: Path) -> Dict[Tuple[str, str], Dict]:
    out = {}
    for record in load_jsonl(path):
        out[(record["id"], record["variant"])] = record
    return out


def construct_gstep_labels_paper(
    baseline_path: Optional[Path] = None,
    cot_path: Optional[Path] = None,
) -> List[Dict]:
    if baseline_path is None:
        baseline_path = DEFAULT_RESULTS_DIR / "fc_baseline.jsonl"
    if cot_path is None:
        cot_path = DEFAULT_RESULTS_DIR / "fc_notool_cot.jsonl"

    records = load_jsonl(baseline_path)
    cot_by_key = _load_keyed(cot_path)
    labeled = []

    for record in records:
        key = (record["id"], record["variant"])
        cot_record = cot_by_key.get(key)
        if cot_record is None:
            raise KeyError(
                "Missing matching CoT record for id={0}, variant={1} in {2}".format(
                    record["id"],
                    record["variant"],
                    cot_path,
                )
            )

        if record.get("answer_correct", False):
            label = "commit"
            label_source = "baseline_correct"
            label_confidence = "strong"
        elif cot_record.get("answer_correct", False):
            label = "continue"
            label_source = "cot_fixable"
            label_confidence = "strong"
        elif int(record.get("tool_calls", 0) or 0) < 2:
            label = "continue"
            label_source = "undercompute_heuristic"
            label_confidence = "medium"
        else:
            label = "commit"
            label_source = "residual_commit"
            label_confidence = "weak"

        labeled.append(
            {
                "id": record["id"],
                "variant": record["variant"],
                "label": label,
                "label_source": label_source,
                "label_confidence": label_confidence,
                "baseline_answer_correct": bool(record.get("answer_correct", False)),
                "cot_answer_correct": bool(cot_record.get("answer_correct", False)),
                "record": record,
            }
        )

    return labeled


def construct_gcommit_labels(
    baseline_path: Optional[Path] = None,
) -> List[Dict]:
    if baseline_path is None:
        baseline_path = DEFAULT_RESULTS_DIR / "fc_baseline.jsonl"

    records = load_jsonl(baseline_path)
    labeled = []

    for record in records:
        correct = record.get("answer_correct", False)

        if correct:
            label_4class = "submit"
        elif _has_incompleteness_signals(record):
            label_4class = "one_more_turn"
        else:
            label_4class = "fallback_cot"

        label_binary = "submit" if label_4class == "submit" else "one_more_turn"

        labeled.append({
            "id": record["id"],
            "variant": record["variant"],
            "label_4class": label_4class,
            "label_binary": label_binary,
            "record": record,
        })

    return labeled


def construct_gstep_labels(
    baseline_path: Optional[Path] = None,
) -> List[Dict]:
    if baseline_path is None:
        baseline_path = DEFAULT_RESULTS_DIR / "fc_baseline.jsonl"

    records = load_jsonl(baseline_path)
    labeled = []

    for record in records:
        correct = record.get("answer_correct", False)
        tool_calls = record.get("tool_calls", 0)

        if correct:
            label = "commit"
        elif tool_calls < 2:
            label = "continue"
        else:
            label = "commit"

        labeled.append({
            "id": record["id"],
            "variant": record["variant"],
            "label": label,
            "record": record,
        })

    return labeled


def construct_gcommit_labels_mixed(
    baseline_path: Optional[Path] = None,
    gate_results_path: Optional[Path] = None,
    cot_path: Optional[Path] = None,
) -> List[Dict]:
    if baseline_path is None:
        baseline_path = DEFAULT_RESULTS_DIR / "fc_baseline.jsonl"

    records = load_jsonl(baseline_path)
    cot_by_key = _load_keyed(cot_path) if cot_path is not None else {}
    gate_by_key = _load_keyed(gate_results_path) if gate_results_path is not None else {}

    labeled = []
    for record in records:
        key = (record["id"], record["variant"])
        correct = record.get("answer_correct", False)

        cot_record = cot_by_key.get(key)
        gate_record = gate_by_key.get(key)

        cot_correct = cot_record.get("answer_correct", False) if cot_record else False
        gate_correct = gate_record.get("answer_correct", False) if gate_record else False
        gate_fired = False
        if gate_record:
            gate_fired = (
                gate_record.get("gate_one_more_used", 0) > 0
                or gate_record.get("gate_step_continue_count", 0) > 0
            )

        if correct:
            label_4class = "submit"
        elif cot_correct:
            label_4class = "one_more_turn"
        elif gate_fired and gate_correct:
            label_4class = "one_more_turn"
        elif gate_fired and not gate_correct:
            label_4class = "submit"
        elif _has_incompleteness_signals(record):
            label_4class = "one_more_turn"
        else:
            label_4class = "fallback_cot"

        label_binary = "submit" if label_4class == "submit" else "one_more_turn"

        labeled.append({
            "id": record["id"],
            "variant": record["variant"],
            "label_4class": label_4class,
            "label_binary": label_binary,
            "record": record,
        })

    return labeled


def construct_gstep_labels_mixed(
    baseline_path: Optional[Path] = None,
    gate_results_path: Optional[Path] = None,
    cot_path: Optional[Path] = None,
) -> List[Dict]:
    if baseline_path is None:
        baseline_path = DEFAULT_RESULTS_DIR / "fc_baseline.jsonl"

    records = load_jsonl(baseline_path)
    cot_by_key = _load_keyed(cot_path) if cot_path is not None else {}
    gate_by_key = _load_keyed(gate_results_path) if gate_results_path is not None else {}

    labeled = []
    for record in records:
        key = (record["id"], record["variant"])
        correct = record.get("answer_correct", False)
        tool_calls = record.get("tool_calls", 0)

        cot_record = cot_by_key.get(key)
        gate_record = gate_by_key.get(key)

        cot_correct = cot_record.get("answer_correct", False) if cot_record else False
        gate_correct = gate_record.get("answer_correct", False) if gate_record else False
        gate_fired = False
        if gate_record:
            gate_fired = gate_record.get("gate_step_continue_count", 0) > 0

        if correct:
            label = "commit"
        elif cot_correct:
            label = "continue"
        elif gate_fired and gate_correct:
            label = "continue"
        elif gate_fired and not gate_correct:
            label = "commit"
        elif tool_calls < 2:
            label = "continue"
        else:
            label = "commit"

        labeled.append({
            "id": record["id"],
            "variant": record["variant"],
            "label": label,
            "record": record,
        })

    return labeled


def build_group_kfold_splits(
    samples: List[Dict],
    n_splits: int = N_SPLITS_GROUP_KFOLD,
) -> List[Tuple[List[int], List[int]]]:
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if not samples:
        return []

    groups = [str(sample["id"]) for sample in samples]
    unique_groups = sorted(set(groups))
    n_splits = min(int(n_splits), len(unique_groups))
    if n_splits < 2:
        return [([], list(range(len(samples))))]

    try:
        from sklearn.model_selection import GroupKFold  # type: ignore

        gkf = GroupKFold(n_splits=n_splits)
        dummy = [0] * len(samples)
        out = []
        for train_idx, test_idx in gkf.split(dummy, groups=groups):
            out.append((list(train_idx), list(test_idx)))
        return out
    except Exception:
        pass

    group_to_indices: Dict[str, List[int]] = {}
    for index, group in enumerate(groups):
        group_to_indices.setdefault(group, []).append(index)

    fold_groups: List[List[str]] = [[] for _ in range(n_splits)]
    for group in unique_groups:
        hashed = int(hashlib.md5(group.encode("utf-8")).hexdigest(), 16) % n_splits
        fold_groups[hashed].append(group)

    splits = []
    all_indices = list(range(len(samples)))
    for fold in range(n_splits):
        test_set = set(fold_groups[fold])
        test_indices = [i for i, group in enumerate(groups) if group in test_set]
        test_index_set = set(test_indices)
        train_indices = [i for i in all_indices if i not in test_index_set]
        if test_indices:
            splits.append((train_indices, test_indices))
    return splits
