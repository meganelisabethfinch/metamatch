from __future__ import annotations

from dataclasses import dataclass

from reconciliation.reconciliation_service import Match

from .relation import Relation


@dataclass(frozen=True)
class Edge:
    """How a source entity relates to a target entity, by id."""
    source_id: str
    target_id: str
    relation: Relation | None = None   # None if untyped
    score: float | None = None
    reason: str | None = None
    provenance: str | None = None      # what made it, e.g. "gp_works_mb.qwen35-9b" (a job and its model), or an import

    @classmethod
    def from_match(cls, source_id: str, match: Match, provenance: str | None = None) -> Edge:
        return cls(source_id, match.target.id, match.relation, match.score, match.reason, provenance)
