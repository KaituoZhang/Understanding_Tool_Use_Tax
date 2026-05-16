import json
import re
from typing import Any, Dict, List, Sequence

from eval.gsm8k_semantic_distractor.gsm8k_semantic_distractor_utils import (
    NOISY_VARIANTS,
    canonicalize_variant,
    derive_topic_hints,
    extract_core_words_and_numbers,
    extract_final_answer,
)


NEUTRAL_UNITS = {
    "books",
    "lights",
    "degrees",
    "kilometers",
    "tickets",
    "balloons",
    "pages",
    "bottles",
    "minutes",
    "kilograms",
    "meters",
}
DIFFERENCE_MARKERS = {
    "another",
    "different",
    "elsewhere",
    "yesterday",
    "unrelated",
    "separate",
    "nearby",
    "in a different",
    "someone else",
    "not this case",
}
HEDGING_MARKERS = {
    "claims",
    "claimed",
    "reported",
    "reportedly",
    "said",
    "unverified",
    "might",
    "possibly",
    "perhaps",
    "it is said",
    "it is possible",
    "some say",
    "around",
    "about",
    "likely",
    "not confirmed",
}
SOLVING_HINT_MARKERS = {
    "multiply",
    "divide",
    "sum",
    "total",
    "fraction",
    "percent",
    "per",
    "each",
    "equation",
    "formula",
    "calculate",
    "compute",
    "add",
    "subtract",
}
ANSWER_ASSERTION_MARKERS = {
    "the answer is",
    "correct answer",
    "choose option",
    "option x",
}
WORD_RE = re.compile(r"[A-Za-z]+")


def _contains_phrase(text: str, phrases: Sequence[str]) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in phrases)


def _numbers_in_text(text: str) -> List[str]:
    return re.findall(r"\b\d+(?:\.\d+)?\b", text)


def _parse_csvish(text: str) -> List[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def _meta_core_words(record: Dict[str, Any], variant: str) -> List[str]:
    meta = ((record.get("meta") or {}).get("gen_raw") or {}).get(variant, {})
    words = _parse_csvish(meta.get("core_words", ""))
    if words:
        return words
    extracted, _ = extract_core_words_and_numbers(record.get("question", ""))
    return _parse_csvish(extracted)


def _meta_core_numbers(record: Dict[str, Any], variant: str) -> List[str]:
    meta = ((record.get("meta") or {}).get("gen_raw") or {}).get(variant, {})
    numbers = _parse_csvish(meta.get("core_numbers", ""))
    if numbers:
        return numbers
    _, extracted = extract_core_words_and_numbers(record.get("question", ""))
    return _parse_csvish(extracted)


def _content_tokens(text: str) -> List[str]:
    return [token.lower() for token in WORD_RE.findall(text or "") if len(token) >= 3]


def _evidence_sentences(record: Dict[str, Any]) -> List[str]:
    sentences = record.get("question_sentences") or record.get("core_sentences") or []
    evidence_ids = record.get("evidence_sentence_ids") or []
    if not isinstance(sentences, list) or not isinstance(evidence_ids, list):
        return []
    out = []
    for evidence_id in evidence_ids:
        if isinstance(evidence_id, int) and 0 <= evidence_id < len(sentences):
            out.append(str(sentences[evidence_id]))
    return out


def _paraphrase_like(sentence: str, evidence_sentences: Sequence[str]) -> bool:
    sent_tokens = set(_content_tokens(sentence))
    if len(sent_tokens) < 2:
        return False
    for evidence in evidence_sentences:
        evidence_tokens = set(_content_tokens(evidence))
        if not evidence_tokens:
            continue
        overlap = sent_tokens & evidence_tokens
        if len(overlap) >= 2:
            return True
    return False


def _mentions_core_words(sentence: str, core_words: Sequence[str]) -> bool:
    lower = sentence.lower()
    for word in core_words:
        if re.search(rf"\b{re.escape(word.lower())}\b", lower):
            return True
    return False


def only_neutral_units(text: str, units: set) -> bool:
    lower = text.lower()
    for unit in units:
        if re.search(rf"\b{re.escape(str(unit).lower())}\b", lower):
            return False
    numbers = _numbers_in_text(lower)
    if numbers and not any(re.search(rf"\b{re.escape(unit)}\b", lower) for unit in NEUTRAL_UNITS):
        return False
    return True


def class_constraints_ok(
    variant: str,
    sentence: str,
    units: set,
    pairs: List[str],
    topic_hints: List[str],
    core_numbers: List[str],
    core_words: List[str],
    evidence_sentences: Sequence[str],
) -> bool:
    lower = sentence.lower()
    has_numbers = bool(_numbers_in_text(lower))
    has_diff = _contains_phrase(lower, DIFFERENCE_MARKERS)
    has_hedge = _contains_phrase(lower, HEDGING_MARKERS)
    has_solving_hint = _contains_phrase(lower, SOLVING_HINT_MARKERS)

    if variant == "TB":
        return (
            not has_numbers
            and not has_diff
            and not has_hedge
            and not has_solving_hint
            and not _mentions_core_words(sentence, core_words)
        )
    if variant == "PED":
        return (
            has_diff
            and not has_numbers
            and not has_hedge
            and not _mentions_core_words(sentence, core_words)
        )
    if variant == "HU":
        return (
            has_hedge
            and not has_diff
            and not has_numbers
            and not _contains_phrase(lower, ANSWER_ASSERTION_MARKERS)
        )
    if variant == "SP":
        sentence_numbers = set(_numbers_in_text(lower))
        allowed_numbers = set(core_numbers)
        return (
            sentence_numbers.issubset(allowed_numbers)
            and not has_diff
            and not has_hedge
            and not has_solving_hint
            and _paraphrase_like(sentence, evidence_sentences)
        )
    return False


def eval_safe_expression(expression: str) -> float:
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expression):
        raise ValueError("Unsafe expression")
    return eval(expression, {"__builtins__": {}}, {})


def recompute_from_chain(calc_chain: Dict[str, Any]) -> str:
    last_value = None
    for expression in calc_chain.get("exprs", []):
        last_value = eval_safe_expression(expression)
    return calc_chain.get("final") or (str(last_value) if last_value is not None else None)


def truth_invariant_ok(calc_chain: Dict[str, Any], official_answer: str) -> bool:
    recomputed = recompute_from_chain(calc_chain)
    if recomputed is None:
        return False
    try:
        return abs(float(recomputed) - float(official_answer)) < 1e-9
    except ValueError:
        return str(recomputed).strip() == str(official_answer).strip()


def leakage_check(noise_sentences: List[str], official_answer: str, pairs: List[str]) -> bool:
    text = " ".join(sentence.lower() for sentence in noise_sentences if sentence).strip()
    if not text:
        return False
    if official_answer and official_answer.lower() in text:
        return True
    return any(pair.lower() in text for pair in pairs)


def validate_noise_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    validated_records = []
    for record in records:
        updated_record = dict(record)
        schema = updated_record["schema"]
        units = set(schema["units"])
        pairs = schema["pairs"]
        calc_chain = schema["calc_chain"]
        official_answer = extract_final_answer(updated_record)
        topic_hints = derive_topic_hints(updated_record["question"], schema["units"], k=4)
        evidence_sentences = _evidence_sentences(updated_record)

        validated_noise = {}
        grouped_noise: Dict[str, List[str]] = {variant: [] for variant in NOISY_VARIANTS}
        for noise_record in updated_record.get("noise", []):
            variant = canonicalize_variant(noise_record.get("type"))
            if variant in grouped_noise and isinstance(noise_record.get("text"), str):
                grouped_noise[variant].append(noise_record["text"])

        for variant, sentences in grouped_noise.items():
            core_numbers = _meta_core_numbers(updated_record, variant)
            core_words = _meta_core_words(updated_record, variant)
            constraints_ok = all(
                class_constraints_ok(
                    variant,
                    sentence,
                    units,
                    pairs,
                    topic_hints,
                    core_numbers,
                    core_words,
                    evidence_sentences,
                )
                for sentence in sentences
            )
            validated_noise[variant] = {
                "class_constraints": constraints_ok,
                "no_leakage": not leakage_check(sentences, official_answer, pairs),
            }

        updated_record["validated_noises"] = validated_noise
        updated_record["meta_checks"] = {"truth_invariant": truth_invariant_ok(calc_chain, official_answer)}
        validated_records.append(updated_record)
    return validated_records


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", dest="output_path", required=True)
    args = parser.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    validated_records = validate_noise_records(records)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        for record in validated_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
