import re
from typing import Any, Dict, List

from eval.gsm8k_semantic_distractor.gsm8k_semantic_distractor_utils import (
    force_question_last,
    split_sentences,
)


RELATION_HINTS = [
    "each",
    "per",
    "ratio",
    "rate",
    "every",
    "half",
    "double",
    "twice",
    "remaining",
    "leftover",
    "together",
    "altogether",
    "in total",
    "sum",
    "total",
]

QUESTION_PREFIXES = [
    "how many",
    "how much",
    "what is",
    "what are",
    "find",
    "compute",
    "calculate",
    "determine",
]


def is_question_sentence(sentence: str) -> bool:
    lower = sentence.strip().lower()
    return lower.endswith("?") or any(lower.startswith(prefix) for prefix in QUESTION_PREFIXES)


def has_number(sentence: str) -> bool:
    return bool(re.search(r"\b\d+\b", sentence))


def has_relation_hint(sentence: str) -> bool:
    lower = sentence.lower()
    return any(hint in lower for hint in RELATION_HINTS)


def parse_first_expression_numbers(calc_chain: Dict[str, Any]) -> set:
    expressions = (calc_chain or {}).get("exprs") or []
    if not expressions:
        return set()
    return set(re.findall(r"\b\d+\b", expressions[0]))


def pick_evidence_indices(sentences: List[str], evidence_mode: str, calc_chain: Dict[str, Any]) -> List[int]:
    if evidence_mode == "all":
        return list(range(len(sentences)))

    non_question_indices = [index for index, sentence in enumerate(sentences) if not is_question_sentence(sentence)]
    if evidence_mode == "heuristic":
        selected = [
            index
            for index in non_question_indices
            if has_number(sentences[index]) or has_relation_hint(sentences[index])
        ]
        return selected or [0]

    if evidence_mode == "mse":
        needed_numbers = parse_first_expression_numbers(calc_chain)
        selected = []
        for index in non_question_indices:
            sentence_numbers = set(re.findall(r"\b\d+\b", sentences[index]))
            if sentence_numbers & needed_numbers:
                selected.append(index)
        if selected:
            return selected
        return [
            index
            for index in non_question_indices
            if has_number(sentences[index]) or has_relation_hint(sentences[index])
        ] or [0]

    raise ValueError(f"Unknown evidence_mode: {evidence_mode}")


def build_core_records(records: List[Dict[str, Any]], evidence_mode: str = "heuristic") -> List[Dict[str, Any]]:
    built_records = []
    for record in records:
        updated_record = dict(record)
        core_sentences = force_question_last(split_sentences(updated_record["question"]))
        calc_chain = (updated_record.get("schema") or {}).get("calc_chain") or {}
        updated_record["core_sentences"] = core_sentences
        updated_record["evidence_sentence_ids"] = pick_evidence_indices(
            core_sentences,
            evidence_mode,
            calc_chain,
        )
        built_records.append(updated_record)
    return built_records


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", dest="output_path", required=True)
    parser.add_argument("--evidence_mode", choices=["all", "heuristic", "mse"], default="heuristic")
    args = parser.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    built_records = build_core_records(records, args.evidence_mode)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        for record in built_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
