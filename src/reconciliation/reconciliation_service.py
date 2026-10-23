from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import zip_longest

from graph.relation import Relation


def merge_in_order(*orders: list[str]) -> list[str]:
    # several ordered lists -> one, keeping each list's order: position by position, e.g. [a70, v1] and [a65, v1]
    # -> [a70, a65, v1]
    merged = []
    for items in zip_longest(*orders):
        for x in items:
            if x is not None and x not in merged:
                merged.append(x)
    return merged


class ServiceUnavailable(Exception):
    """A remote service still failing after its retries (e.g. unreachable); a run stopped by it can resume later."""


@dataclass(frozen=True)
class Entity:
    """An item to be matched: an id, the text used to represent it, and any other properties."""
    id: str
    text: str
    properties: dict = field(default_factory=dict)


@dataclass
class Rejection:
    """Why one candidate a service tried for a query didn't match it."""
    target_id: str
    reason: str                    # e.g. "score below 70", "failed MusicOccupation"
    progress: tuple = ()           # how far it got (e.g. (stage, check)): higher = further
    score: float | None = None     # its score when it was tried, if any


@dataclass
class Match:
    target: Entity
    relation: Relation | None = None   # None if the service doesn't type matches
    score: float | None = None         # None if the service doesn't score matches
    reason: str | None = None          # the service's rationale (e.g. LLM reasoning), if any
    raw: str | None = None             # raw backend output (e.g. LLM response), for debugging


class ReconciliationService(ABC):
    @abstractmethod
    def reconcile(self, query: Entity, targets: list[Entity]) -> list[Match]:
        """Which of ``targets`` match ``query``, and how?"""

    def reconcile_all(self, queries: list[Entity], targets: dict[str, list[Entity]]) -> dict[str, list[Match]]:
        """``targets`` maps each query id to that query's targets."""
        return {q.id: self.reconcile(q, targets[q.id]) for q in queries}

    def query_key(self, query: Entity, targets: list[Entity] | None) -> str | None:
        """Everything this service sends for the query (for CachedService); None if it can't be cached."""
        return None

    def fetch_all(self, queries: list[Entity], targets) -> dict[str, list[Match]]:
        """The slow part of reconcile_all, which CachedService caches; keep() then filters it."""
        return self.reconcile_all(queries, targets)

    def keep(self, query: Entity, matches: list[Match]) -> list[Match]:
        """Which fetched matches to keep (e.g. score >= threshold); all by default."""
        return matches

    def explain(self, query: Entity, targets, fetched=None) -> list[Rejection]:
        """Why each candidate this service tried for the query was rejected (fetched: its fetch_all result, if known)."""
        return [Rejection(t.id, "no match") for t in targets or []]

    def reasons(self) -> list[str]:
        """The reasons explain() can give, in pipeline order (e.g. to order them in plots)."""
        return ["no match"]

    def failure_reason(self, query: Entity, targets, reason_by: str = "best") -> str:
        """Why the query got no matches: a summary of explain(). reason_by: "best" = the highest-scored
        candidate's reason; "last" = the reason of the candidate that got furthest"""
        rejections = self.explain(query, targets)
        if not rejections:
            return "no candidates"
        if reason_by == "last":
            return max(rejections, key=lambda r: r.progress).reason
        return max(rejections, key=lambda r: -float("inf") if r.score is None else r.score).reason

    @property
    def label(self) -> str:
        return type(self).__name__.removesuffix("Service")
