"""Dependency-light models used when sklearn/xgboost are unavailable."""

import math
from typing import Any, Dict, List, Sequence


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _mean_vector(rows: Sequence[Sequence[float]]) -> List[float]:
    if not rows:
        return []
    width = len(rows[0])
    totals = [0.0] * width
    for row in rows:
        for i, value in enumerate(row):
            totals[i] += float(value)
    return [value / float(len(rows)) for value in totals]


class LinearGateModel:
    """
    A tiny linear classifier trained from class means.

    This is not intended to replace stronger models. It exists so the public
    release remains runnable without extra ML dependencies.
    """

    def __init__(
        self,
        weights: List[float] = None,
        bias: float = 0.0,
        classes_: List[int] = None,
    ):
        self.weights = list(weights or [])
        self.bias = float(bias)
        self.classes_ = list(classes_ or [0, 1])

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]):
        rows = [list(map(float, row)) for row in X]
        labels = [int(label) for label in y]
        positives = [row for row, label in zip(rows, labels) if label == 1]
        negatives = [row for row, label in zip(rows, labels) if label == 0]

        if not rows:
            self.weights = []
            self.bias = 0.0
            return self

        if positives and negatives:
            pos_mean = _mean_vector(positives)
            neg_mean = _mean_vector(negatives)
            self.weights = [pos - neg for pos, neg in zip(pos_mean, neg_mean)]
            midpoint = [(pos + neg) / 2.0 for pos, neg in zip(pos_mean, neg_mean)]
            self.bias = -_dot(self.weights, midpoint)
        elif positives:
            self.weights = [0.0] * len(rows[0])
            self.bias = 8.0
        else:
            self.weights = [0.0] * len(rows[0])
            self.bias = -8.0
        return self

    def decision_function(self, X: Sequence[Sequence[float]]):
        scores = [_dot(self.weights, list(map(float, row))) + self.bias for row in X]
        if len(scores) == 1:
            return scores[0]
        return scores

    def predict_proba(self, X: Sequence[Sequence[float]]):
        scores = self.decision_function(X)
        if not isinstance(scores, list):
            scores = [scores]
        probs = []
        for score in scores:
            p1 = _sigmoid(float(score))
            probs.append([1.0 - p1, p1])
        return probs

    def predict(self, X: Sequence[Sequence[float]]):
        return [1 if row[1] >= 0.5 else 0 for row in self.predict_proba(X)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "linear_gate_model_v1",
            "weights": list(self.weights),
            "bias": float(self.bias),
            "classes_": list(self.classes_),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LinearGateModel":
        return cls(
            weights=list(payload.get("weights", [])),
            bias=float(payload.get("bias", 0.0)),
            classes_=list(payload.get("classes_", [0, 1])),
        )
