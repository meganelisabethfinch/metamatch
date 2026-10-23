from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

from graph.relation import Relation

from .reconciliation_service import Entity, Match, ReconciliationService, Rejection

logger = logging.getLogger(__name__)


@dataclass
class _Search:
    # one query's search: the targets still to judge (a stack: the next is last), those judged, what it found
    query: Entity
    frontier: list[Entity]
    judged: set[str] = field(default_factory=set)
    positives: list[Match] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)


class GuidedSearchService(ReconciliationService):
    """Asks `judge` about the targets one at a time, in order, until it says IS; returns every positive judgement.
    Many queries are searched together in rounds: each round asks the judge about the next target of every open
    query in one call (e.g. one GPU batch, or one Anthropic batch); each query's order of judgements is as if alone."""

    def __init__(self, judge: ReconciliationService, view=None, hierarchy: str | None = None, tree: bool = False,
                 batch_size: int | None = 1):
        # judge: decides how the query relates to one target (e.g. an LLMService)
        # tree: on a partial match, look at that target's neighbours next (needs view + hierarchy, the property of
        #   view's entities that lists the works they're part of, e.g. "part_of"). The query is A, the target B:
        #   A IS_PART_OF B -> B is bigger, try B's parts; A CONTAINS B -> B is smaller, try what B is part of
        # batch_size: at most this many queries open at once (one judgement each per round); a query that finishes
        #   is replaced by the next waiting one, so rounds stay full. None: every query open from the start
        if tree and (view is None or hierarchy is None):
            raise ValueError("tree search needs view and hierarchy")
        self.judge = judge
        self.view = view
        self.hierarchy = hierarchy
        self.tree = tree
        self.batch_size = batch_size

    def reconcile(self, query: Entity, targets: list[Entity]) -> list[Match]:
        return self.reconcile_all([query], {query.id: targets})[query.id]

    def reconcile_all(self, queries: list[Entity], targets: dict[str, list[Entity]]) -> dict[str, list[Match]]:
        return {s.query.id: s.positives for s in self._search(queries, targets)}

    def explain(self, query: Entity, targets, fetched=None) -> list[Rejection]:
        # the targets the judge was asked about and said IS_NOT (replayed: judgements are cached)
        return self._search([query], {query.id: targets or []})[0].rejections

    def _search(self, queries: list[Entity], targets: dict[str, list[Entity]]) -> list[_Search]:
        searches = [_Search(q, list(reversed(targets[q.id]))) for q in queries]
        waiting, open_ = deque(searches), {}   # open_: search -> the target it's asking about this round
        rounds = 0
        while True:
            while waiting and (self.batch_size is None or len(open_) < self.batch_size):   # refill the window
                s = waiting.popleft()
                if (t := self._next(s)) is not None:
                    open_[id(s)] = (s, t)
            if not open_:
                break
            asked = list(open_.values())
            answers = self.judge.reconcile_all([s.query for s, _ in asked], {s.query.id: [t] for s, t in asked})
            open_ = {}
            for s, t in asked:
                if self._consume(s, t, next(iter(answers[s.query.id]), None)) and (t := self._next(s)) is not None:
                    open_[id(s)] = (s, t)
            rounds += 1
            if len(searches) > 1:
                logger.debug(f"round {rounds}: {len(asked)} judged; {len(open_)} open, {len(waiting)} waiting")
        return searches

    def _next(self, s: _Search) -> Entity | None:
        # the query's next target not yet judged, or None when there are none left
        while s.frontier:
            target = s.frontier.pop()
            if target.id not in s.judged:
                s.judged.add(target.id)
                return target
        return None

    def _consume(self, s: _Search, target: Entity, match: Match | None) -> bool:
        # records the judge's answer about target; whether the query's search goes on
        if match is None or match.relation in (None, Relation.IS_NOT):
            s.rejections.append(Rejection(target.id, "predicted no match"))
            return True
        s.positives.append(match)
        if match.relation is Relation.IS:
            return False
        if self.tree:
            s.frontier += reversed([n for n in self._neighbours(target, match.relation) if n.id not in s.judged])
        return True

    def _neighbours(self, target: Entity, relation: Relation) -> list[Entity]:
        if relation is Relation.IS_PART_OF:   # the query is part of the target: try the target's parts
            ids = self.view.referencing(self.hierarchy, [target.id])[target.id]
        else:                                 # CONTAINS: the target is part of the query: try what the target is part of
            ids = target.properties.get(self.hierarchy) or []
        return self.view.get(ids)

    def reasons(self) -> list[str]:
        return ["predicted no match"]

    @property
    def label(self) -> str:
        # the judge's: the reasons come from its answers, e.g. "LLM: predicted no match"
        return self.judge.label

    def __repr__(self) -> str:
        return f"GuidedSearch({self.judge!r}, {'tree' if self.tree else 'flat'}, batch_size={self.batch_size})"
