"""
Offline training for protocol-internal gates.

The paper-faithful default trains G-STEP from `fc_baseline.jsonl` and
`fc_notool_cot.jsonl` using:

- 120-dimensional inference-safe features
- `StandardScaler`
- 2-layer MLP (`128 -> 64`)
- 5-fold `GroupKFold`

Legacy `g_commit` training is kept as an optional non-paper extension.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .configs import N_SPLITS_GROUP_KFOLD, RANDOM_SEED
from .label_construction import (
    build_group_kfold_splits,
    construct_gcommit_labels,
    construct_gcommit_labels_mixed,
    construct_gstep_labels,
    construct_gstep_labels_mixed,
    construct_gstep_labels_paper,
)
from .learned_feature_builder import (
    DEFAULT_AUG_SPEC,
    build_commit_features_aug_online,
    build_step_features_aug_online,
)
from .simple_models import LinearGateModel


def _binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    tp = sum(1 for gold, pred in zip(y_true, y_pred) if gold == 1 and pred == 1)
    tn = sum(1 for gold, pred in zip(y_true, y_pred) if gold == 0 and pred == 0)
    fp = sum(1 for gold, pred in zip(y_true, y_pred) if gold == 0 and pred == 1)
    fn = sum(1 for gold, pred in zip(y_true, y_pred) if gold == 1 and pred == 0)

    total = max(1, len(y_true))
    accuracy = float(tp + tn) / float(total)
    precision = float(tp) / float(max(1, tp + fp))
    recall = float(tp) / float(max(1, tp + fn))
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "acc": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _try_import_mlp():
    try:
        import numpy as np  # type: ignore
        from sklearn.neural_network import MLPClassifier  # type: ignore
        from sklearn.pipeline import make_pipeline  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore

        return np, MLPClassifier, make_pipeline, StandardScaler
    except Exception:
        return None


def _train_test_split(rows: Sequence[Sequence[float]], indices: Sequence[int]) -> List[List[float]]:
    return [list(rows[index]) for index in indices]


def _labels_for_indices(labels: Sequence[int], indices: Sequence[int]) -> List[int]:
    return [int(labels[index]) for index in indices]


def _rep_count(confidence: str) -> int:
    conf = str(confidence or "").lower()
    if conf == "strong":
        return 3
    if conf == "medium":
        return 2
    return 1


def _artifact_to_jsonable(artifact: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    model = artifact.get("model")
    if not hasattr(model, "to_dict"):
        return None
    payload = dict(artifact)
    payload["model"] = model.to_dict()
    return payload


def _save_artifacts(artifact: Dict[str, Any], out_path: Path) -> Dict[str, str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(artifact, handle)

    saved = {"pickle": str(out_path)}
    jsonable = _artifact_to_jsonable(artifact)
    if jsonable is not None:
        json_path = out_path.with_suffix(".json")
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(jsonable, handle, indent=2, ensure_ascii=False)
        saved["json"] = str(json_path)
    return saved


def _fit_binary_model(
    X: Sequence[Sequence[float]],
    y: Sequence[int],
    *,
    hidden1: int,
    hidden2: int,
    max_iter: int,
    random_seed: int,
    model_family: str,
):
    if len(set(int(label) for label in y)) < 2:
        return LinearGateModel().fit(X, y), "linear_fallback"

    imported = _try_import_mlp()
    if imported is None:
        return LinearGateModel().fit(X, y), "linear_fallback"

    np, MLPClassifier, make_pipeline, StandardScaler = imported
    try:
        early_stopping = len(X) >= 20
        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(hidden1, hidden2),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size="auto",
                learning_rate_init=1e-3,
                max_iter=max_iter,
                early_stopping=early_stopping,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=random_seed,
            ),
        )
        model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
        return model, model_family
    except Exception:
        return LinearGateModel().fit(X, y), "linear_fallback"


def _cv_train_binary(
    *,
    X: List[List[float]],
    y: List[int],
    groups: List[str],
    task: str,
    positive_name: str,
    out_path: Path,
    feature_spec: Optional[Dict[str, Any]] = None,
    hidden1: int = 128,
    hidden2: int = 64,
    max_iter: int = 500,
    model_family: str = "paper_mlp",
) -> Dict[str, Any]:
    if not X:
        raise ValueError("No training samples available for {0}".format(task))

    dummy_samples = [{"id": group} for group in groups]
    splits = build_group_kfold_splits(dummy_samples, n_splits=N_SPLITS_GROUP_KFOLD)

    fold_rows = []
    backend_name = "linear_fallback"
    for fold_index, (train_idx, test_idx) in enumerate(splits):
        if (not train_idx) or (not test_idx):
            continue
        model, backend_name = _fit_binary_model(
            _train_test_split(X, train_idx),
            _labels_for_indices(y, train_idx),
            hidden1=hidden1,
            hidden2=hidden2,
            max_iter=max_iter,
            random_seed=RANDOM_SEED + fold_index,
            model_family=model_family,
        )
        predictions = model.predict(_train_test_split(X, test_idx))
        metrics = _binary_metrics(_labels_for_indices(y, test_idx), predictions)
        fold_rows.append(
            {
                "fold": fold_index,
                "acc": float(metrics["acc"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
        )

    final_model, backend_name = _fit_binary_model(
        X,
        y,
        hidden1=hidden1,
        hidden2=hidden2,
        max_iter=max_iter,
        random_seed=RANDOM_SEED,
        model_family=model_family,
    )
    artifact = {
        "model": final_model,
        "task": task,
        "positive_class": positive_name,
        "n_samples": int(len(y)),
        "n_positive": int(sum(y)),
        "backend": backend_name,
        "model_family": model_family if backend_name != "linear_fallback" else "linear_fallback",
        "feature_spec": feature_spec or {},
        "cv": fold_rows,
    }
    artifact_paths = _save_artifacts(artifact, out_path)
    artifact["artifact_paths"] = artifact_paths
    return artifact


def _as_qid(sample_id: str) -> str:
    return str(sample_id)


def _default_cot_path(results_dir: Path) -> Optional[Path]:
    candidate = results_dir / "fc_notool_cot.jsonl"
    return candidate if candidate.exists() else None


def _feature_dim(feature_spec: Dict[str, Any]) -> int:
    return 24 + int(feature_spec["step_text_bins"]) + int(feature_spec["step_trace_bins"])


def train_gcommit(
    results_dir: Path,
    out_dir: Path,
    label_mode: str = "offpolicy",
    cot_path: Optional[Path] = None,
    gate_results_path: Optional[Path] = None,
    hidden1: int = 128,
    hidden2: int = 64,
    max_iter: int = 500,
    feature_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    baseline_path = results_dir / "fc_baseline.jsonl"
    feature_spec = dict(feature_spec or DEFAULT_AUG_SPEC)

    if label_mode == "mixed":
        labeled = construct_gcommit_labels_mixed(
            baseline_path=baseline_path,
            gate_results_path=gate_results_path,
            cot_path=cot_path,
        )
    else:
        labeled = construct_gcommit_labels(baseline_path=baseline_path)

    X: List[List[float]] = []
    y: List[int] = []
    groups: List[str] = []
    for sample in labeled:
        record = sample["record"]
        features = build_commit_features_aug_online(
            pred_answer=str(record.get("pred_answer", "") or ""),
            reasoning=str(record.get("reasoning", "") or ""),
            tool_trace=record.get("tool_trace", []) or [],
            n_chunks_seen=int(record.get("n_chunks_seen", 0) or 0),
            pred_evidence_ids=record.get("pred_evidence_ids", []) or [],
            expression=str(record.get("expression", "") or ""),
            max_tool_calls_limit=int(record.get("max_tool_calls_limit", 5) or 5),
            feature_spec=feature_spec,
        )
        rep = _rep_count(sample.get("label_confidence", "weak"))
        target = 1 if sample["label_binary"] == "one_more_turn" else 0
        for _ in range(max(1, rep)):
            X.append(features)
            y.append(target)
            groups.append(_as_qid(record["id"]))

    artifact = _cv_train_binary(
        X=X,
        y=y,
        groups=groups,
        task="g_commit",
        positive_name="one_more_turn",
        out_path=out_dir / "g_commit.pkl",
        feature_spec=feature_spec,
        hidden1=hidden1,
        hidden2=hidden2,
        max_iter=max_iter,
        model_family="extension_mlp",
    )
    return {
        "task": artifact.get("task"),
        "positive_class": artifact.get("positive_class"),
        "n_samples": artifact.get("n_samples"),
        "n_positive": artifact.get("n_positive"),
        "label_mode": label_mode,
        "backend": artifact.get("backend"),
        "model_family": artifact.get("model_family"),
        "feature_spec": artifact.get("feature_spec"),
        "cv": artifact.get("cv", []),
        "artifact_paths": artifact.get("artifact_paths", {}),
        "paper_faithful": False,
    }


def train_gstep(
    results_dir: Path,
    out_dir: Path,
    label_mode: str = "paper",
    cot_path: Optional[Path] = None,
    gate_results_path: Optional[Path] = None,
    hidden1: int = 128,
    hidden2: int = 64,
    max_iter: int = 500,
    feature_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    baseline_path = results_dir / "fc_baseline.jsonl"
    feature_spec = dict(feature_spec or DEFAULT_AUG_SPEC)

    if label_mode == "paper":
        cot_path = cot_path or _default_cot_path(results_dir)
        if cot_path is None:
            raise ValueError(
                "Paper-faithful G-STEP training requires fc_notool_cot.jsonl. "
                "Provide --cot_path or place it under results_dir."
            )
        labeled = construct_gstep_labels_paper(
            baseline_path=baseline_path,
            cot_path=cot_path,
        )
    elif label_mode == "mixed":
        labeled = construct_gstep_labels_mixed(
            baseline_path=baseline_path,
            gate_results_path=gate_results_path,
            cot_path=cot_path,
        )
    else:
        labeled = construct_gstep_labels(baseline_path=baseline_path)

    X: List[List[float]] = []
    y: List[int] = []
    groups: List[str] = []
    label_source_counts: Dict[str, int] = {}
    for sample in labeled:
        record = sample["record"]
        features = build_step_features_aug_online(
            tool_trace=record.get("tool_trace", []) or [],
            candidate_text=str(record.get("reasoning", "") or ""),
            pred_answer=str(record.get("pred_answer", "") or ""),
            n_chunks_seen=int(record.get("n_chunks_seen", 0) or 0),
            max_tool_calls_limit=int(record.get("max_tool_calls_limit", 5) or 5),
            feature_spec=feature_spec,
        )
        rep = _rep_count(sample.get("label_confidence", "weak"))
        if sample.get("label_source") == "cot_fixable":
            rep += 1
        target = 1 if sample["label"] == "continue" else 0
        for _ in range(max(1, rep)):
            X.append(features)
            y.append(target)
            groups.append(_as_qid(record["id"]))
        source = str(sample.get("label_source", "unknown"))
        label_source_counts[source] = label_source_counts.get(source, 0) + 1

    artifact = _cv_train_binary(
        X=X,
        y=y,
        groups=groups,
        task="g_step",
        positive_name="continue",
        out_path=out_dir / "g_step.pkl",
        feature_spec=feature_spec,
        hidden1=hidden1,
        hidden2=hidden2,
        max_iter=max_iter,
        model_family="paper_mlp",
    )
    return {
        "task": artifact.get("task"),
        "positive_class": artifact.get("positive_class"),
        "n_samples": artifact.get("n_samples"),
        "n_positive": artifact.get("n_positive"),
        "label_mode": label_mode,
        "backend": artifact.get("backend"),
        "model_family": artifact.get("model_family"),
        "feature_spec": artifact.get("feature_spec"),
        "feature_dim": _feature_dim(feature_spec),
        "label_source_counts": label_source_counts,
        "cv": artifact.get("cv", []),
        "artifact_paths": artifact.get("artifact_paths", {}),
        "paper_faithful": artifact.get("backend") != "linear_fallback",
        "cot_path": str(cot_path) if cot_path is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Train gate models for GSM8K mechanism ablation.")
    parser.add_argument("--results_dir", required=True, help="Directory containing fc_baseline.jsonl")
    parser.add_argument("--out_dir", required=True, help="Directory to save trained gate artifacts")
    parser.add_argument("--which", default="g_step", choices=["all", "g_commit", "g_step"])
    parser.add_argument(
        "--label_mode",
        default="paper",
        choices=["paper", "offpolicy", "mixed"],
        help="paper: baseline+CoT labels for G-STEP. offpolicy/mixed: legacy extensions.",
    )
    parser.add_argument(
        "--cot_path",
        default=None,
        help="Path to fc_notool_cot.jsonl. Paper mode auto-detects results_dir/fc_notool_cot.jsonl.",
    )
    parser.add_argument(
        "--gate_results_path",
        default=None,
        help="Optional legacy gate results JSONL used by mixed-label extensions.",
    )
    parser.add_argument("--hidden1", type=int, default=128)
    parser.add_argument("--hidden2", type=int, default=64)
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--step_text_bins", type=int, default=DEFAULT_AUG_SPEC["step_text_bins"])
    parser.add_argument("--step_trace_bins", type=int, default=DEFAULT_AUG_SPEC["step_trace_bins"])
    parser.add_argument("--commit_reason_bins", type=int, default=DEFAULT_AUG_SPEC["commit_reason_bins"])
    parser.add_argument("--commit_trace_bins", type=int, default=DEFAULT_AUG_SPEC["commit_trace_bins"])
    parser.add_argument("--commit_expr_bins", type=int, default=DEFAULT_AUG_SPEC["commit_expr_bins"])
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cot_path = Path(args.cot_path) if args.cot_path else None
    gate_results_path = Path(args.gate_results_path) if args.gate_results_path else None
    feature_spec = {
        "type": "aug_v1",
        "step_text_bins": int(args.step_text_bins),
        "step_trace_bins": int(args.step_trace_bins),
        "commit_reason_bins": int(args.commit_reason_bins),
        "commit_trace_bins": int(args.commit_trace_bins),
        "commit_expr_bins": int(args.commit_expr_bins),
    }

    summary: Dict[str, Any] = {}
    if args.which in ("all", "g_step"):
        summary["g_step"] = train_gstep(
            results_dir,
            out_dir,
            label_mode=args.label_mode,
            cot_path=cot_path,
            gate_results_path=gate_results_path,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            max_iter=args.max_iter,
            feature_spec=feature_spec,
        )
        print("Saved: {0}".format(out_dir / "g_step.pkl"))

    if args.which in ("all", "g_commit"):
        gcommit_label_mode = args.label_mode
        if gcommit_label_mode == "paper":
            gcommit_label_mode = "mixed" if cot_path is not None else "offpolicy"
        summary["g_commit"] = train_gcommit(
            results_dir,
            out_dir,
            label_mode=gcommit_label_mode,
            cot_path=cot_path,
            gate_results_path=gate_results_path,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            max_iter=args.max_iter,
            feature_spec=feature_spec,
        )
        print("Saved: {0}".format(out_dir / "g_commit.pkl"))

    with (out_dir / "train_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print("Saved: {0}".format(out_dir / "train_summary.json"))


if __name__ == "__main__":
    main()
