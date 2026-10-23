from __future__ import annotations

from dataclasses import replace

from .reconciliation_service import Entity, Match, ReconciliationService, Rejection


class CascadeService(ReconciliationService):
    """Runs each stage over only the targets the previous stage matched; a match keeps the last score it was given."""

    def __init__(self, stages: list[ReconciliationService]):
        self.stages = stages

    def reconcile(self, query: Entity, targets: list[Entity] | None = None) -> list[Match]:
        return self.reconcile_all([query], {query.id: targets})[query.id]

    def reconcile_all(self, queries: list[Entity], targets: dict[str, list[Entity]] | None = None) -> dict[str, list[Match]]:
        results = {q.id: [] for q in queries}
        active = queries
        for stage in self.stages:
            previous = {q.id: {m.target.id: m.score for m in results[q.id]} for q in active}
            results.update(stage.reconcile_all(active, targets))
            for q in active:   # e.g. a verifier stage gives no score: keep the search stage's
                results[q.id] = [m if m.score is not None else replace(m, score=previous[q.id].get(m.target.id))
                                 for m in results[q.id]]
            # Queries with no matches left drop out; the rest pass their matches on as targets.
            active = [q for q in active if results[q.id]]
            targets = {q.id: [m.target for m in results[q.id]] for q in active}
            if not active:
                break
        return results

    def explain(self, query: Entity, targets=None, fetched=None) -> list[Rejection]:
        # replays the stages for the query (cheap: slow stages are cached): each candidate's reason comes from the stage
        # that rejected it, e.g. "Verifier: failed MusicOccupation"; progress = (stage, progress within the stage)
        rejections, scores = [], {}
        for i, stage in enumerate(self.stages):
            for r in stage.explain(query, targets):
                score = r.score if r.score is not None else scores.get(r.target_id)
                rejections.append(Rejection(r.target_id, f"{stage.label}: {r.reason}", (i, *r.progress), score))
            matches = stage.reconcile_all([query], {query.id: targets} if targets is not None else None)[query.id]
            if not matches:
                break
            scores |= {m.target.id: m.score for m in matches if m.score is not None}
            targets = [m.target for m in matches]
        return rejections

    def reasons(self) -> list[str]:
        # each stage's reasons, stage by stage, prefixed as explain() prefixes them
        return [f"{stage.label}: {r}" for stage in self.stages for r in stage.reasons()]

    def __repr__(self) -> str:
        return f"Cascade({', '.join(map(repr, self.stages))})"
