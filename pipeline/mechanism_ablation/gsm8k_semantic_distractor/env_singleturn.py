import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, List

from .data_loader import Chunk, Problem


@dataclass
class SingleTurnRetrievalResult:
    chunks: List[Chunk]
    chunk_indices: List[int]


@dataclass
class SingleTurnEnvConfig:
    top_k: int = 5
    shuffle_chunks: bool = True
    seed: int = 42
    guarantee_evidence: bool = False


def _stable_problem_seed(problem_id: str, seed: int) -> int:
    digest = hashlib.md5(problem_id.encode("utf-8")).hexdigest()
    return seed + int(digest[:8], 16)


class SingleTurnEnvironment:
    def __init__(self, problem: Problem, config: SingleTurnEnvConfig):
        self.problem = problem
        self.config = config
        self.rng = random.Random(_stable_problem_seed(problem.id, config.seed))
        self._pool = list(problem.all_info_chunks)
        self._retrieve_count = 0

    def get_question(self) -> str:
        return self.problem.question_text

    def retrieve(self, query: str = "") -> SingleTurnRetrievalResult:
        del query
        self._retrieve_count += 1
        pool = list(self._pool)
        if self.config.guarantee_evidence:
            evidence = [chunk for chunk in pool if chunk.is_evidence]
            non_evidence = [chunk for chunk in pool if not chunk.is_evidence]
            self.rng.shuffle(non_evidence)
            n_evidence = min(len(evidence), max(1, self.config.top_k // 3))
            selected = evidence[:n_evidence] + non_evidence[: self.config.top_k - n_evidence]
        else:
            self.rng.shuffle(pool)
            selected = pool[: self.config.top_k]
        if self.config.shuffle_chunks:
            self.rng.shuffle(selected)
        return SingleTurnRetrievalResult(selected, [chunk.index for chunk in selected])

    def retrieve_all(self) -> SingleTurnRetrievalResult:
        self._retrieve_count += 1
        pool = list(self._pool)
        if self.config.shuffle_chunks:
            self.rng.shuffle(pool)
        return SingleTurnRetrievalResult(pool, [chunk.index for chunk in pool])

    def get_gold(self) -> Dict[str, Any]:
        return {"answer": self.problem.gold_answer, "evidence_ids": self.problem.evidence_ids}

    @property
    def n_retrievals(self) -> int:
        return self._retrieve_count

    def get_stats(self) -> Dict[str, Any]:
        return {
            "mode": "singleturn",
            "variant": self.problem.variant,
            "problem_id": self.problem.id,
            "n_total_chunks": len(self._pool),
            "n_evidence": len(self.problem.evidence_chunks),
            "n_noise": len(self.problem.noise_chunks),
            "n_retrievals": self._retrieve_count,
        }
