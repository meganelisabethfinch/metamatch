from __future__ import annotations

from abc import abstractmethod

from .reconciliation_service import Entity, Match, ReconciliationService, Rejection


class RankingService(ReconciliationService):
    """Scores every target, then keeps the best (top_k and/or score >= threshold)."""

    def __init__(self, top_k: int | None = None, threshold: float | None = None):
        self.top_k = top_k
        self.threshold = threshold

    @abstractmethod
    def score(self, query: Entity, target: Entity) -> float:
        """Similarity between query and one target (higher = better)."""

    def reconcile(self, query: Entity, targets: list[Entity]) -> list[Match]:
        return self._keep_best([Match(t, score=self.score(query, t)) for t in targets])

    def explain(self, query: Entity, targets, fetched=None) -> list[Rejection]:
        # the candidates ranked below the threshold, or below the top_k
        scored = self._score_all(query, targets or [])
        kept = {m.target.id for m in self._keep_best(scored)}
        return [Rejection(m.target.id, f"score below {self.threshold}" if self.threshold is not None
                          and m.score < self.threshold else f"not in top {self.top_k}", score=m.score)
                for m in scored if m.target.id not in kept]

    def reasons(self) -> list[str]:
        below = [f"score below {self.threshold}"] if self.threshold is not None else []
        return below + ([f"not in top {self.top_k}"] if self.top_k is not None else [])

    def _score_all(self, query: Entity, targets: list[Entity]) -> list[Match]:
        return [Match(t, score=self.score(query, t)) for t in targets]

    def _keep_best(self, matches: list[Match]) -> list[Match]:
        # best first, then score >= threshold and the top_k
        matches = sorted(matches, key=lambda m: m.score, reverse=True)
        if self.threshold is not None:
            matches = [m for m in matches if m.score >= self.threshold]
        return matches[:self.top_k] if self.top_k is not None else matches
