import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from eval.gsm8k_semantic_distractor.gsm8k_semantic_distractor_utils import (
    NOISY_VARIANTS,
    build_core_view,
    canonicalize_variant,
    extract_core_words_and_numbers,
    force_question_last,
    split_sentences,
)


HEDGING_MARKERS = [
    "reportedly",
    "some say",
    "it is said",
    "it is possible",
    "possibly",
    "might",
]

DIFFERENCE_MARKERS = [
    "another",
    "elsewhere",
    "different",
    "nearby",
    "separate",
]

OPENAI_PROMPTS = {
    "TB": """
You are generating IN-DOMAIN background filler for a math word problem.
These sentences should SOUND relevant to the same general topic/domain, but MUST NOT help solve the problem.

Return STRICT JSON ONLY:
{
  "topic": "<topic>",
  "before": [<exactly {before_n} short sentences>],
  "after": [<exactly {after_n} short sentences>]
}

HARD RULES (must follow):
- Insert-only: write NEW sentences only; do NOT rewrite or paraphrase the evidence.
- Keep ONE coherent in-domain topic (same domain as the problem).
- NO numbers anywhere (no digits, no written numbers, no fractions).
- No math/solving strategy hints (forbidden: multiply, divide, sum, total, fraction, percent, per, each, equation, formula, calculate, compute).
- Do NOT use difference markers (forbidden: another, elsewhere, different, nearby, unrelated, separate, yesterday).
- Do NOT use hedging markers (forbidden: reportedly, claimed, might, possibly, about, around, likely, unverified).
- Do NOT mention the specific entities (names, places, dates) listed in: [{core_words}]. However, you MAY use general domain terms (e.g. "clips", "store", "selling") to keep the topic relevant.
- Do NOT copy core numbers: [{core_numbers}]
- Each sentence must be short, factual, and self-contained.

Problem core (evidence-centric view):
{core_view}
""".strip(),
    "PED": """
You are generating TOPIC-RELATED distractors that must be LOGICALLY EXCLUDED from the problem.
They should share surface-level domain words, but clearly refer to a different person/time/place/event.

Return STRICT JSON ONLY:
{
  "topic": "<topic>",
  "before": [<exactly {before_n} short sentences>],
  "after": [<exactly {after_n} short sentences>]
}

HARD RULES (must follow):
- Insert-only: write NEW sentences only; do NOT rewrite/paraphrase the evidence.
- Keep ONE coherent topic in the same domain.
- EACH sentence MUST include at least ONE difference marker: another / elsewhere / different / nearby / unrelated / separate / in a different / someone else / not this case.
- NO numbers anywhere.
- Must include 1-2 domain/topic hint words (same domain), but DO NOT restate evidence facts/relations.
- MUST NOT use hedging markers (forbidden: reportedly, claimed, might, possibly, about, around, likely).
- Use specific INVENTED names (e.g., "Alice", "Mr. Smith", "The neighbor") instead of generic terms like "someone" or "another person" to make it sound natural.
- Do NOT copy core numbers: [{core_numbers}]
- Each sentence short and self-contained.

Problem core (evidence-centric view):
{core_view}
""".strip(),
    "HU": """
You are generating TOPIC-RELATED statements that sound like evidence but are epistemically UNCERTAIN.
These should not create contradictions that make the problem unsolvable.

Return STRICT JSON ONLY:
{
  "topic": "<topic>",
  "before": [<exactly {before_n} short sentences>],
  "after": [<exactly {after_n} short sentences>]
}

HARD RULES (must follow):
- Insert-only: write NEW sentences only; do NOT rewrite/paraphrase the evidence.
- Keep ONE coherent topic in the same domain.
- MUST NOT include difference markers (forbidden: another, elsewhere, different, nearby, unrelated, separate, yesterday).
- NO numbers anywhere (avoid creating alternative exact facts).
- Do NOT assert the answer or options (forbidden: "the answer is", "correct answer", "choose option", "option X").
- You MAY mention the main entities (e.g. "{core_words}") to make the distractor sound like a valid premise, but the hedging marker must keep it uncertain.
- Do NOT copy core numbers: [{core_numbers}]
- Each sentence short and self-contained.
- EACH sentence MUST include at least ONE hedging marker: reportedly / claimed / it is said / some say / might / possibly / perhaps / around / about / likely / unverified / not confirmed.
- The tone should be speculative, vague, or rumor-based, distinct from the factual tone of the evidence.

Problem core (evidence-centric view):
{core_view}
""".strip(),
    "SP": """
You are generating SEMANTICS-PRESERVING PARAPHRASES of the EVIDENCE sentences in a word problem.
Your output will be inserted BEFORE and AFTER the evidence as paraphrase fillers.

Return STRICT JSON ONLY:
{
  "topic": "<topic>",
  "before": [<exactly {before_n} short sentences>],
  "after": [<exactly {after_n} short sentences>]
}

HARD RULES (must follow):
- One coherent topic: all sentences must be about the SAME scenario as the evidence.
- Paraphrase-only: every generated sentence MUST be a paraphrase of some [EVID] sentence content.
- Do NOT paraphrase [Q] (the question sentence). Do NOT add new facts, new events, or new entities.
- Syntactic Variation: You MUST change the sentence structure significantly (e.g., switch between active/passive voice, change word order). Do NOT simply replace one word; make the sentence LOOK different but MEAN the same.
- Numbers/values: you may mention numbers ONLY if they already appear in the evidence. Do NOT introduce any new numbers, ranges, approximations, or near-miss numbers. Keep exact values unchanged (no "about", "around", "roughly", "approximately").
- Keep relations unchanged: Preserve relational operators such as half / twice / remaining / difference / total. Do NOT change who did what, when, or the direction of a relation.
- No markers: Do NOT use difference markers (another / elsewhere / yesterday / different / nearby / unrelated / separate).
- No hedging markers: Do NOT use hedging markers (reportedly / claimed / might / possibly / about / around / likely / unverified).
- No solving hints: Do NOT include step-by-step solution language (multiply / divide / add / subtract / compute / calculate / equation / formula).
- Each sentence must be short, grammatical, and self-contained.

Problem core (evidence-centric view):
{core_view}
""".strip(),
}


def normalize_example(record: Dict[str, Any]) -> Tuple[List[str], List[int]]:
    if isinstance(record.get("question_sentences"), list) and record["question_sentences"]:
        question_sentences = [str(sentence).strip() for sentence in record["question_sentences"] if str(sentence).strip()]
    elif isinstance(record.get("core_sentences"), list) and record["core_sentences"]:
        question_sentences = [str(sentence).strip() for sentence in record["core_sentences"] if str(sentence).strip()]
    else:
        question_sentences = split_sentences(record["question"])
    question_sentences = force_question_last(question_sentences)

    evidence_ids = record.get("evidence_sentence_ids", [])
    if not isinstance(evidence_ids, list):
        evidence_ids = []
    evidence_ids = sorted(
        set(
            int(index)
            for index in evidence_ids
            if isinstance(index, int) and 0 <= index < max(len(question_sentences) - 1, 0)
        )
    )
    if not evidence_ids and len(question_sentences) >= 2:
        evidence_ids = [0]
    return question_sentences, evidence_ids


def _default_noise_plan() -> Dict[str, Any]:
    return {
        "defaults": {"before": 2, "after": 2},
        "per_variant": {},
        "max_total_per_variant": 8,
    }


def counts_for_variant(noise_plan: Dict[str, Any], variant: str) -> Tuple[int, int]:
    plan = noise_plan or _default_noise_plan()
    defaults = plan.get("defaults", {"before": 2, "after": 2})
    per_variant = (plan.get("per_variant") or {}).get(variant, {})
    before = int(per_variant.get("before", defaults.get("before", 2)))
    after = int(per_variant.get("after", defaults.get("after", 2)))
    cap = int(plan.get("max_total_per_variant", 8))
    if before + after > cap:
        after = max(0, cap - before)
    return max(0, before), max(0, after)


def _make_before_after(sentences: Sequence[str], before_n: int, after_n: int) -> Tuple[List[str], List[str]]:
    return list(sentences[:before_n]), list(sentences[before_n: before_n + after_n])


def _mock_tb(question_sentences: Sequence[str], before_n: int, after_n: int) -> Tuple[str, List[str], List[str]]:
    topic = "bookmark sales" if "bookmark" in " ".join(question_sentences).lower() else "school supplies"
    bank = [
        f"People often keep notes on {topic}.",
        f"General trends in {topic} can change over time.",
        f"Background details on {topic} do not determine the answer by themselves.",
        f"Related reports on {topic} describe the setting without affecting the math.",
    ]
    before, after = _make_before_after(bank, before_n, after_n)
    return topic, before, after


def _mock_ped(question_sentences: Sequence[str], before_n: int, after_n: int) -> Tuple[str, List[str], List[str]]:
    topic = "parallel entity"
    bank = [
        "In another case, Mason handled a separate set of items.",
        "Elsewhere, Nora worked on a different task for another group.",
        "In a nearby setting, Liam prepared supplies for a separate event.",
        "Different people often completed unrelated versions of the same activity.",
    ]
    before, after = _make_before_after(bank, before_n, after_n)
    return topic, before, after


def _mock_hu(question_sentences: Sequence[str], before_n: int, after_n: int) -> Tuple[str, List[str], List[str]]:
    topic = "hedged note"
    bank = [
        "It is said that the earlier notes might have been reviewed later.",
        "Some say the situation was possibly discussed after the fact.",
        "Reportedly, the record may have been checked more than once.",
        "It is possible that someone mentioned the details again afterward.",
    ]
    before, after = _make_before_after(bank, before_n, after_n)
    return topic, before, after


def _mock_sp(question_sentences: Sequence[str], before_n: int, after_n: int) -> Tuple[str, List[str], List[str]]:
    evidence_sentences = [sentence for sentence in question_sentences[:-1] if sentence]
    paraphrases = []
    for sentence in evidence_sentences:
        paraphrase = sentence.replace("Then ", "After that, ")
        paraphrase = paraphrase.replace("How many", "Determine how many")
        paraphrases.append(paraphrase)
    while len(paraphrases) < before_n + after_n:
        paraphrases.extend(paraphrases or evidence_sentences)
    before, after = _make_before_after(paraphrases, before_n, after_n)
    return "semantic paraphrase", before, after


def generate_mock_noise(
    variant: str,
    question_sentences: Sequence[str],
    before_n: int,
    after_n: int,
) -> Tuple[str, List[str], List[str], Optional[str], Optional[str]]:
    generators = {
        "TB": _mock_tb,
        "PED": _mock_ped,
        "HU": _mock_hu,
        "SP": _mock_sp,
    }
    topic, before, after = generators[variant](question_sentences, before_n, after_n)
    raw_payload = json.dumps({"topic": topic, "before": before, "after": after}, ensure_ascii=False)
    return topic, before, after, raw_payload, None


def call_openai_noise_generator(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    seed: int,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, List[str], List[str], Optional[str], Optional[str]]:
    raw_text = None
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        raw_text = response.choices[0].message.content
        payload = json.loads(raw_text)
    except Exception as exc:
        return "", [], [], raw_text, str(exc)

    topic = str(payload.get("topic", "")).strip()
    before = [text.strip() for text in payload.get("before", []) if isinstance(text, str) and text.strip()]
    after = [text.strip() for text in payload.get("after", []) if isinstance(text, str) and text.strip()]
    return topic, before, after, raw_text, None


def generate_noise_records(
    records: List[Dict[str, Any]],
    generator_config: Dict[str, Any],
    noise_plan: Dict[str, Any],
) -> List[Dict[str, Any]]:
    provider = generator_config.get("provider", "mock")
    client = None
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=generator_config.get("api_key") or os.environ.get("OPENAI_API_KEY"))

    generated_records = []
    for record in records:
        updated_record = dict(record)
        question_sentences, evidence_ids = normalize_example(updated_record)
        core_view = build_core_view(question_sentences, evidence_ids)
        core_words, core_numbers = extract_core_words_and_numbers(" ".join(question_sentences))

        updated_record.setdefault("noise", [])
        updated_record.setdefault("meta", {}).setdefault("gen_raw", {})

        for variant in NOISY_VARIANTS:
            before_n, after_n = counts_for_variant(noise_plan, variant)
            if before_n == 0 and after_n == 0:
                continue

            if provider == "mock":
                topic, before, after, raw_text, error = generate_mock_noise(
                    variant,
                    question_sentences,
                    before_n,
                    after_n,
                )
            elif provider == "openai":
                user_prompt = OPENAI_PROMPTS[variant].format(
                    core_view=core_view,
                    core_words=core_words,
                    core_numbers=core_numbers,
                    before_n=before_n,
                    after_n=after_n,
                )
                topic, before, after, raw_text, error = call_openai_noise_generator(
                    client=client,
                    model=generator_config.get("model", "gpt-4o-mini"),
                    system_prompt="Return STRICT JSON ONLY with keys topic, before, after.",
                    user_prompt=user_prompt,
                    seed=int(generator_config.get("seed", 42)),
                    temperature=float(generator_config.get("temperature", 0.0)),
                    max_tokens=int(generator_config.get("max_tokens", 400)),
                )
            else:
                raise ValueError(f"Unsupported noise generator provider: {provider}")

            before = before[:before_n]
            after = after[:after_n]
            for sentence in before:
                updated_record["noise"].append({"type": canonicalize_variant(variant), "position": "before", "text": sentence})
            for sentence in after:
                updated_record["noise"].append({"type": canonicalize_variant(variant), "position": "after", "text": sentence})

            updated_record["meta"]["gen_raw"][canonicalize_variant(variant)] = {
                "topic": topic,
                "raw": raw_text,
                "error": error,
                "core_words": core_words,
                "core_numbers": core_numbers,
            }

        generated_records.append(updated_record)
    return generated_records


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", dest="output_path", required=True)
    parser.add_argument("--provider", choices=["mock", "openai"], default="mock")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    generated = generate_noise_records(
        records,
        generator_config={"provider": args.provider, "model": args.model},
        noise_plan=_default_noise_plan(),
    )
    with open(args.output_path, "w", encoding="utf-8") as handle:
        for record in generated:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
