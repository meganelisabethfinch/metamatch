from __future__ import annotations

from collections import Counter

from graph.relation import Relation

from .reconciliation_service import Entity, Match, ReconciliationService


class VotingService(ReconciliationService):
    """A judge that asks several judges (e.g. one LLM each) about each target and returns their majority answer.
    Each judge keeps its own cache, so answers a model has given before (e.g. in its own run) are reused."""

    def __init__(self, judges: dict[str, ReconciliationService]):
        # judges: voter name -> judge, strongest first: a tie goes to the first voter who gave one of the tied answers.
        # A voter whose answer couldn't be read abstains
        self.judges = judges

    def reconcile(self, query: Entity, targets: list[Entity]) -> list[Match]:
        return self.reconcile_all([query], {query.id: targets})[query.id]

    def reconcile_all(self, queries: list[Entity], targets: dict[str, list[Entity]]) -> dict[str, list[Match]]:
        # each judge answers every query in one call (so each batches its own), then the answers are tallied
        answers = {name: judge.reconcile_all(queries, targets) for name, judge in self.judges.items()}
        results = {}
        for q in queries:
            votes = {name: {m.target.id: m.relation for m in answers[name][q.id]} for name in self.judges}
            results[q.id] = [m for t in targets[q.id]
                             if (m := self._vote(t, {name: v.get(t.id) for name, v in votes.items()}))]
        return results

    def _vote(self, target: Entity, votes: dict[str, Relation | None]) -> Match | None:
        # the majority relation (IS_NOT included), its share of the votes as the score, and every vote as the reason
        cast = {name: r for name, r in votes.items() if r is not None}
        if not cast:
            return None
        counts = Counter(cast.values())
        top = max(counts.values())
        winner = next(r for r in cast.values() if counts[r] == top)   # ties: the first voter's (dicts keep order)
        reason = ", ".join(f"{name}: {r.name if r else 'abstained'}" for name, r in votes.items())
        return Match(target, relation=winner, score=top / len(cast), reason=reason)

    @property
    def label(self) -> str:
        return "Vote"

    def reasons(self) -> list[str]:
        return ["predicted no match"]

    def __repr__(self) -> str:
        return f"Vote({', '.join(f'{n}: {j!r}' for n, j in self.judges.items())})"
