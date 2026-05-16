import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from eval.gsm8k_semantic_distractor.gsm8k_semantic_distractor_utils import (
    canonicalize_variant,
    canonicalize_variants,
    extract_final_answer,
    to_numeric,
)


@dataclass
class Chunk:
    text: str
    index: int
    is_evidence: bool = False
    is_noise: bool = False
    is_question: bool = False
    noise_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "index": self.index,
            "is_evidence": self.is_evidence,
            "is_noise": self.is_noise,
            "is_question": self.is_question,
            "noise_type": self.noise_type,
        }


@dataclass
class Problem:
    id: str
    variant: str
    question_text: str
    chunks: List[Chunk] = field(default_factory=list)
    evidence_ids: List[int] = field(default_factory=list)
    noise_ids: List[int] = field(default_factory=list)
    gold_answer: str = ""
    gold_answer_numeric: Optional[float] = None
    calc_chain: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def evidence_chunks(self) -> List[Chunk]:
        return [chunk for chunk in self.chunks if chunk.is_evidence]

    @property
    def noise_chunks(self) -> List[Chunk]:
        return [chunk for chunk in self.chunks if chunk.is_noise]

    @property
    def all_info_chunks(self) -> List[Chunk]:
        return [chunk for chunk in self.chunks if not chunk.is_question]

    def numbered_text(self, indices: Optional[List[int]] = None) -> str:
        pool = self.chunks if indices is None else [self.chunks[index] for index in indices]
        return "\n".join(f"[{chunk.index}] {chunk.text}" for chunk in pool)


def load_problems(
    path: str,
    variants: Optional[List[str]] = None,
    limit: Optional[int] = None,
    offset: int = 0,
    seed: int = 42,
) -> List[Problem]:
    del seed  # no stochastic sampling is used in the loader
    canonical_variants = set(canonicalize_variants(variants) or [])
    problems: List[Problem] = []
    seen_question_ids = set()
    skipped_question_ids = set()

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            example = json.loads(line)
            variant = canonicalize_variant(example.get("variant", "base"))
            if canonical_variants and variant not in canonical_variants:
                continue

            question_id = example["id"]
            if question_id not in seen_question_ids and question_id not in skipped_question_ids:
                if len(skipped_question_ids) < offset:
                    skipped_question_ids.add(question_id)
                    continue
                if limit is not None and len(seen_question_ids) >= limit:
                    continue
                seen_question_ids.add(question_id)
            elif question_id in skipped_question_ids:
                continue

            question_sentences = example.get("question_sentences", [])
            evidence_ids = sorted(set(example.get("evidence_sentence_ids", [])))
            noise_ids = sorted(set(example.get("noise_sentence_ids", [])))
            chunks = []
            for index, sentence in enumerate(question_sentences):
                is_question = index == len(question_sentences) - 1
                chunks.append(
                    Chunk(
                        text=sentence.strip(),
                        index=index,
                        is_evidence=index in evidence_ids,
                        is_noise=index in noise_ids,
                        is_question=is_question,
                        noise_type=variant if index in noise_ids else None,
                    )
                )

            gold_answer = extract_final_answer(example)
            calc_chain = (example.get("meta") or {}).get("calc_chain")
            problems.append(
                Problem(
                    id=question_id,
                    variant=variant,
                    question_text=question_sentences[-1] if question_sentences else "",
                    chunks=chunks,
                    evidence_ids=evidence_ids,
                    noise_ids=noise_ids,
                    gold_answer=gold_answer,
                    gold_answer_numeric=to_numeric(gold_answer),
                    calc_chain=calc_chain,
                    raw=example,
                )
            )
    return problems


def group_by_question(problems: List[Problem]) -> Dict[str, Dict[str, Problem]]:
    grouped: Dict[str, Dict[str, Problem]] = {}
    for problem in problems:
        grouped.setdefault(problem.id, {})[problem.variant] = problem
    return grouped
