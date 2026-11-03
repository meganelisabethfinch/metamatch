from __future__ import annotations

import json
from typing import Callable

from .reconciliation_service import Entity, Match, ReconciliationService, Rejection


class SearchService(ReconciliationService):
    """Matches each query to the best results of a search over the target, e.g. a local full-text index."""

    def __init__(self, search: Callable[[str, int], list[tuple[str, float]]], view, limit: int = 50,
                 threshold: float | None = None, query_text: Callable[[Entity], str] | None = None):
        # search(text, limit) -> [(target id, score)], best first; view: gives the target entities for those ids
        # query_text: optional function query entity -> the text to search for (default: its text)
        self.search = search
        self.query_text = query_text
        self.view = view
        self.limit = limit
        self.threshold = threshold

    def reconcile(self, query: Entity, targets=None) -> list[Match]:
        return self.reconcile_all([query])[query.id]

    def reconcile_all(self, queries: list[Entity], targets=None) -> dict[str, list[Match]]:
        return {q: self.keep(None, ms) for q, ms in self.fetch_all(queries, targets).items()}

    def fetch_all(self, queries: list[Entity], targets=None) -> dict[str, list[Match]]:
        # every search result (up to limit), best first; targets is ignored (the search covers the whole target);
        # the threshold is applied by keep()
        results = {}
        for query in queries:
            hits = self.search(self.query_text(query) if self.query_text else query.text, self.limit)
            entities = {e.id: e for e in self.view.get([id for id, _ in hits])}
            results[query.id] = [Match(entities[id], score=score) for id, score in hits if id in entities]
        return results

    def keep(self, query, matches: list[Match]) -> list[Match]:
        return [m for m in matches if self.threshold is None or m.score >= self.threshold]

    def explain(self, query: Entity, targets=None, fetched=None) -> list[Rejection]:
        # the candidates the search returned but the threshold rejected
        fetched = self.fetch_all([query], targets)[query.id] if fetched is None else fetched
        kept = {m.target.id for m in self.keep(query, fetched)}
        return [Rejection(m.target.id, "score below threshold", score=m.score)
                for m in fetched if m.target.id not in kept]

    def query_key(self, query: Entity, targets=None) -> str:
        text = self.query_text(query) if self.query_text else query.text
        return json.dumps({"service": "search", "view": str(self.view), "text": text, "limit": self.limit},
                          sort_keys=True, ensure_ascii=False)

    def reasons(self) -> list[str]:
        return ["score below threshold"]

    def __repr__(self) -> str:
        return f"Search({self.view}, threshold={self.threshold})"
