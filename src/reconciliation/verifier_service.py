from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace

from graph.relation import Relation

from .property_source import PropertySource
from .reconciliation_service import Entity, Match, ReconciliationService, Rejection


class PropertyVerifier(ABC):
    """One property check on a (query, target) pair."""
    failure = ""   # why a target that fails it is rejected, e.g. "no music-related occupation"

    def __init__(self, name: str | None = None, failure: str | None = None):
        self.name = name or type(self).__name__
        self.failure = failure or self.failure or f"failed {self.name}"

    @abstractmethod
    def check(self, query: Entity, target: Entity) -> bool | None:
        """True = pass, False = fail, None = not applicable (the needed properties are missing)."""


class AnyOf(PropertyVerifier):
    """Passes if any applicable sub-verifier passes; None if none apply."""

    def __init__(self, verifiers: list[PropertyVerifier], name: str | None = None, failure: str | None = None):
        super().__init__(name, failure)
        self.verifiers = verifiers

    def check(self, query: Entity, target: Entity) -> bool | None:
        outcomes = [o for o in (v.check(query, target) for v in self.verifiers) if o is not None]
        return any(outcomes) if outcomes else None


class AllOf(PropertyVerifier):
    """Passes if every applicable sub-verifier passes; None if none apply."""

    def __init__(self, verifiers: list[PropertyVerifier], name: str | None = None, failure: str | None = None):
        super().__init__(name, failure)
        self.verifiers = verifiers

    def check(self, query: Entity, target: Entity) -> bool | None:
        outcomes = [o for o in (v.check(query, target) for v in self.verifiers) if o is not None]
        return all(outcomes) if outcomes else None


class VerifierService(ReconciliationService):
    """Keeps the targets that no verifier rejects (not applicable counts as a pass)."""

    def __init__(
        self,
        verifiers: list[PropertyVerifier],
        query_properties: PropertySource | None = None,
        target_properties: PropertySource | None = None,
        relation: Relation | None = None,
        top_k: int | None = None,
    ):
        # relation: given to every target that passes (e.g. Relation.IS); top_k: keep only the first k that pass
        self.verifiers = verifiers
        self.query_properties = query_properties
        self.target_properties = target_properties
        self.relation = relation
        self.top_k = top_k

    def reconcile(self, query: Entity, targets: list[Entity]) -> list[Match]:
        return self.reconcile_all([query], {query.id: targets})[query.id]

    def reconcile_all(self, queries: list[Entity], targets: dict[str, list[Entity]]) -> dict[str, list[Match]]:
        queries, targets = self._enrich_all(queries, targets)
        results = {}
        for q in queries:
            passed = [Match(t, relation=self.relation) for t in targets[q.id] if self._first_failure(q, t) is None]
            results[q.id] = passed[:self.top_k] if self.top_k is not None else passed
        return results

    def verify(self, query: Entity, targets: list[Entity]) -> list[str | None]:
        """For each target, the name of the first verifier it failed, or None if it passed."""
        return self.verify_all([query], {query.id: targets})[query.id]

    def verify_all(self, queries: list[Entity], targets: dict[str, list[Entity]]) -> dict[str, list[str | None]]:
        queries, targets = self._enrich_all(queries, targets)
        return {q.id: [self._first_failure(q, t) for t in targets[q.id]] for q in queries}

    def _enrich_all(self, queries, targets):
        """Adds properties to the queries and to the set of all targets, fetching each id once."""
        queries = _enrich(queries, self.query_properties)
        unique = list({t.id: t for q in queries for t in targets[q.id]}.values())
        enriched = {t.id: t for t in _enrich(unique, self.target_properties)}
        return queries, {q.id: [enriched[t.id] for t in targets[q.id]] for q in queries}

    def explain(self, query: Entity, targets, fetched=None) -> list[Rejection]:
        # each rejected target's first failed check; progress = how many checks it passed first
        if not targets:
            return []
        (query,), enriched = self._enrich_all([query], {query.id: targets})
        return [Rejection(t.id, f.failure, (self.verifiers.index(f),))
                for t in enriched[query.id] if (f := self._first_failure(query, t))]

    def reasons(self) -> list[str]:
        return [v.failure for v in self.verifiers]   # in the order they're checked

    def __repr__(self) -> str:
        return f"Verifier({', '.join(v.name for v in self.verifiers)}; top_k={self.top_k})"

    def _first_failure(self, query: Entity, target: Entity) -> PropertyVerifier | None:
        for v in self.verifiers:
            if v.check(query, target) is False:
                return v
        return None


def _enrich(entities: list[Entity], source: PropertySource | None) -> list[Entity]:
    if source is None:
        return entities
    props = source.get(entities)
    return [replace(e, properties={**e.properties, **props.get(e.id, {})}) for e in entities]
