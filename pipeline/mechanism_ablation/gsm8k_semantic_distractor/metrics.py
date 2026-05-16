import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set


NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def normalize_to_number(text: str) -> Optional[float]:
    if text is None:
        return None
    match = NUMBER_RE.search(str(text).strip().replace(",", ""))
    if not match:
        return None
    try:
        value = float(match.group(0))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def answers_equal(prediction: str, gold: str, tol: float = 1e-9) -> bool:
    prediction_number = normalize_to_number(prediction)
    gold_number = normalize_to_number(gold)
    if prediction_number is not None and gold_number is not None:
        return abs(prediction_number - gold_number) < tol
    normalize = lambda text: re.sub(r"\s+", "", str(text).strip().lower())
    return normalize(prediction) == normalize(gold)


def evidence_precision_recall_f1(pred_ids: List[int], gold_ids: List[int]) -> Dict[str, float]:
    pred_set: Set[int] = set(pred_ids)
    gold_set: Set[int] = set(gold_ids)
    if not gold_set:
        precision = 1.0 if not pred_set else 0.0
        return {"precision": precision, "recall": 1.0, "f1": precision}
    true_positive = len(pred_set & gold_set)
    precision = true_positive / len(pred_set) if pred_set else 0.0
    recall = true_positive / len(gold_set)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def evidence_exact_match(pred_ids: List[int], gold_ids: List[int]) -> bool:
    return set(pred_ids) == set(gold_ids)


def tool_stats(tool_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not tool_trace:
        return {"n_tool_calls": 0, "n_calc_errors": 0, "calc_used": False}
    n_tool_calls = len(tool_trace)
    n_calc_errors = sum(1 for step in tool_trace if str(step.get("output", "")).startswith("Error"))
    return {
        "n_tool_calls": n_tool_calls,
        "n_calc_errors": n_calc_errors,
        "calc_used": n_tool_calls > 0,
    }


def planning_accuracy(pred_evidence_correct: bool, answer_correct: bool, calc_used: bool) -> str:
    if answer_correct:
        return "correct"
    if not pred_evidence_correct:
        return "planning_error"
    return "expression_error" if calc_used else "compute_error"


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}
    buckets: Dict[str, Dict[str, List[Any]]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        key = f"{result['mode']}/{result['variant']}"
        buckets[key]["answer_correct"].append(result.get("answer_correct", False))
        buckets[key]["evidence_f1"].append(result.get("evidence_f1", 0.0))
        buckets[key]["evidence_em"].append(result.get("evidence_exact_match", False))
        if "tool_trace" in result:
            stats = tool_stats(result["tool_trace"])
            buckets[key]["calc_used"].append(stats["calc_used"])
            buckets[key]["n_tool_calls"].append(stats["n_tool_calls"])
            buckets[key]["planning_cat"].append(
                planning_accuracy(
                    result.get("evidence_exact_match", False),
                    result.get("answer_correct", False),
                    stats["calc_used"],
                )
            )

    report = {}
    for key, values in sorted(buckets.items()):
        n = len(values["answer_correct"])
        entry = {
            "n": n,
            "answer_acc": sum(values["answer_correct"]) / n,
            "evidence_f1_mean": sum(values["evidence_f1"]) / n,
            "evidence_em_acc": sum(values["evidence_em"]) / n,
        }
        if values.get("planning_cat"):
            total = len(values["planning_cat"])
            entry["planning_error_rate"] = sum(1 for value in values["planning_cat"] if value == "planning_error") / total
            entry["expression_error_rate"] = sum(1 for value in values["planning_cat"] if value == "expression_error") / total
            entry["compute_error_rate"] = sum(1 for value in values["planning_cat"] if value == "compute_error") / total
        if values.get("n_tool_calls"):
            entry["avg_tool_calls"] = sum(values["n_tool_calls"]) / len(values["n_tool_calls"])
        report[key] = entry
    return report


def print_report(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 110)
    print(f"{'Setting':<25} {'N':>5}  {'Ans%':>7}  {'EvF1':>7}  {'EvEM%':>7}  {'Plan%':>6}  {'Expr%':>6}  {'Comp%':>6}")
    print("-" * 110)
    for key in sorted(report):
        entry = report[key]
        plan = f"{100 * entry['planning_error_rate']:.1f}" if "planning_error_rate" in entry else "  -"
        expr = f"{100 * entry['expression_error_rate']:.1f}" if "expression_error_rate" in entry else "  -"
        comp = f"{100 * entry['compute_error_rate']:.1f}" if "compute_error_rate" in entry else "  -"
        print(
            f"{key:<25} {entry['n']:>5}  {100 * entry['answer_acc']:>6.2f}%  "
            f"{entry['evidence_f1_mean']:>6.4f}  {100 * entry['evidence_em_acc']:>6.2f}%  "
            f"{plan:>6}  {expr:>6}  {comp:>6}"
        )
    print("=" * 110)
