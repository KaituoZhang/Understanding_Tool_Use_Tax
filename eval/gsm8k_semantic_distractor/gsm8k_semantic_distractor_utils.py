import json
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PUBLIC_VARIANTS = ["base", "TB", "PED", "HU", "SP"]
NOISY_VARIANTS = ["TB", "PED", "HU", "SP"]

LEGACY_TO_PUBLIC = {
    "LX": "TB",
    "HN1": "PED",
    "HN2": "HU",
    "LEX": "SP",
}
PUBLIC_TO_LEGACY = {value: key for key, value in LEGACY_TO_PUBLIC.items()}

VARIANT_DESCRIPTIONS = {
    "TB": "Thematic Background",
    "PED": "Parallel Entity Distractor",
    "HU": "Hedged Uncertainty",
    "SP": "Semantic Paraphrase",
}

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
FINAL_ANSWER_RE = re.compile(r"####\s*([^\n\r]+)")
STOPWORDS = set(
    """
    a an the of for to in on at by from with without within and or but if then else when
    while as into over under is are was were be been being this that these those each per
    has have having had do did does doing how what why many much all more left after before
    """
    .split()
)


def canonicalize_variant(variant: Optional[str]) -> str:
    if not variant:
        return "base"
    return LEGACY_TO_PUBLIC.get(variant, variant)


def canonicalize_variants(variants: Optional[Iterable[str]]) -> Optional[List[str]]:
    if variants is None:
        return None
    seen = set()
    ordered: List[str] = []
    for variant in variants:
        canonical = canonicalize_variant(variant)
        if canonical not in seen:
            ordered.append(canonical)
            seen.add(canonical)
    return ordered


def split_sentences(text: str) -> List[str]:
    text = text.strip().replace("\r", "").replace("\n", " ")
    if not text:
        return []
    return [segment.strip() for segment in SENTENCE_SPLIT_RE.split(text) if segment.strip()]


def force_question_last(sentences: Sequence[str]) -> List[str]:
    ordered = [sentence.strip() for sentence in sentences if sentence and sentence.strip()]
    if not ordered:
        return []
    if ordered[-1].endswith("?"):
        return ordered
    for index in range(len(ordered) - 1, -1, -1):
        if ordered[index].endswith("?"):
            question = ordered.pop(index)
            ordered.append(question)
            break
    return ordered


def parse_calc_chain(answer_text: str) -> Dict[str, Any]:
    expressions = [match.group(1) for match in re.finditer(r"<<([^<>]+?)>>", answer_text)]
    final_matches = FINAL_ANSWER_RE.findall(answer_text)
    final_answer = final_matches[-1].strip() if final_matches else None
    if not final_answer:
        for expression in reversed(expressions):
            if "=" in expression:
                final_answer = expression.split("=")[-1].strip()
                break

    safe_expressions = []
    for expression in expressions:
        clean_expression = expression.split("=", 1)[0].strip()
        if re.fullmatch(r"[0-9+\-*/().\s]+", clean_expression):
            safe_expressions.append(clean_expression)
    return {"exprs": safe_expressions, "final": final_answer}


def extract_final_answer(example: Dict[str, Any]) -> str:
    chain = (example.get("schema") or {}).get("calc_chain") or {}
    final_answer = chain.get("final")
    if final_answer is not None and str(final_answer).strip():
        return str(final_answer).strip()
    matches = FINAL_ANSWER_RE.findall(example.get("answer", ""))
    if matches:
        return matches[-1].strip()
    return str(example.get("answer", "")).strip()


def extract_core_words_and_numbers(text: str) -> Tuple[str, str]:
    numbers = sorted(set(re.findall(r"\b\d+(?:\.\d+)?\b", text)))
    words = sorted(
        word
        for word in set(re.findall(r"\b[A-Z][a-z]+\b", text))
        if word not in {"The", "A", "An", "If", "When", "Then", "It", "She", "He", "They", "How"}
    )
    return ", ".join(words), ", ".join(numbers)


def derive_topic_hints(question: str, units: Sequence[str], k: int = 4) -> List[str]:
    hints = list(dict.fromkeys(unit.lower() for unit in units if isinstance(unit, str) and unit.strip()))
    if len(hints) >= k:
        return hints[:k]

    candidates = []
    seen = set(hints)
    for token in re.findall(r"[A-Za-z]+", question):
        token_lower = token.lower()
        if token_lower in STOPWORDS or len(token_lower) < 3 or token_lower in seen:
            continue
        candidates.append(token_lower)
        seen.add(token_lower)

    for word, _ in Counter(candidates).most_common():
        hints.append(word)
        if len(hints) >= k:
            break

    fallback = ["item", "group", "container", "participant", "object"]
    for word in fallback:
        if len(hints) >= k:
            break
        if word not in hints:
            hints.append(word)
    return hints[:k]


def build_core_view(sentences: Sequence[str], evidence_ids: Sequence[int]) -> str:
    evidence_id_set = set(evidence_ids)
    lines = []
    for index, sentence in enumerate(sentences):
        if index == len(sentences) - 1:
            prefix = "[Q]"
        elif index in evidence_id_set:
            prefix = "[EVID]"
        else:
            prefix = "[CTX]"
        lines.append(f"{prefix} {sentence.strip()}")
    return "\n".join(lines)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dump_jsonl(records: Sequence[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def to_numeric(text: str) -> Optional[float]:
    try:
        return float(str(text).replace(",", ""))
    except (TypeError, ValueError):
        return None
