import re
from typing import Any, Dict, List

from eval.gsm8k_semantic_distractor.gsm8k_semantic_distractor_utils import parse_calc_chain


STOPWORD_LIKE_UNITS = {
    "time",
    "day",
    "week",
    "month",
    "number",
    "amount",
    "total",
    "sum",
    "value",
    "people",
}


def lemmatize_simple(word: str) -> str:
    token = word.lower()
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("sses") or token.endswith("xes"):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def extract_units(question: str) -> List[str]:
    units = set()
    for match in re.finditer(r"\b\d+\s+([A-Za-z]+s)\b", question):
        units.add(lemmatize_simple(match.group(1)))
    for match in re.finditer(r"\beach\s+([A-Za-z]+)\s+(has|having|contains)\s+\d+\s+([A-Za-z]+)s\b", question):
        units.add(lemmatize_simple(match.group(1)))
        units.add(lemmatize_simple(match.group(3)))
    for match in re.finditer(r"\bper\s+([A-Za-z]+)\b", question):
        units.add(lemmatize_simple(match.group(1)))
    return [unit for unit in sorted(units) if unit and unit not in STOPWORD_LIKE_UNITS]


def extract_forbidden_pairs(question: str) -> List[str]:
    pairs = []
    lower_question = question.lower()
    for left, _, right in re.findall(
        r"each\s+([a-z]+)\s+(has|having|contains)\s+\d+\s+([a-z]+)s",
        lower_question,
    ):
        pair = f"{right} per {left}"
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def parse_schema_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    parsed_records = []
    for record in records:
        updated_record = dict(record)
        updated_record["schema"] = {
            "units": extract_units(updated_record["question"]),
            "pairs": extract_forbidden_pairs(updated_record["question"]),
            "calc_chain": parse_calc_chain(updated_record["answer"]),
        }
        parsed_records.append(updated_record)
    return parsed_records


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", dest="output_path", required=True)
    args = parser.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    parsed_records = parse_schema_records(records)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        for record in parsed_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
