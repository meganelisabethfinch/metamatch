from __future__ import annotations

from typing import Callable

from .reconciliation_service import Entity, Match, ReconciliationService, Rejection, merge_in_order


class FallbackService(ReconciliationService):
    """Runs `fallback` for the queries `primary` left unmatched (optionally only those `retry_if` accepts)."""

    def __init__(self, primary: ReconciliationService, fallback: ReconciliationService,
                 retry_if: Callable[[Entity], bool] | None = None):
        self.primary = primary
        self.fallback = fallback
        self.retry_if = retry_if

    def reconcile(self, query: Entity, targets=None) -> list[Match]:
        return self.reconcile_all([query], {query.id: targets})[query.id]

    def reconcile_all(self, queries: list[Entity], targets=None) -> dict[str, list[Match]]:
        results = self.primary.reconcile_all(queries, targets)
        retry = [q for q in queries if not results[q.id] and (self.retry_if is None or self.retry_if(q))]
        if retry:
            results.update(self.fallback.reconcile_all(retry, targets))
        return results

    def explain(self, query: Entity, targets=None, fetched=None) -> list[Rejection]:
        # the fallback's rejections if the query was retried with it, else the primary's
        retried = self.retry_if is None or self.retry_if(query)
        return (self.fallback if retried else self.primary).explain(query, targets)

    def reasons(self) -> list[str]:
        # the primary's and the fallback's, interleaved stage by stage (e.g. both searches' thresholds, then the checks)
        return merge_in_order(self.primary.reasons(), self.fallback.reasons())

    @property
    def label(self) -> str:
        return self.primary.label

    def __repr__(self) -> str:
        return f"Fallback({self.primary!r}, then {self.fallback!r})"
