from __future__ import annotations

import json
from pathlib import Path

from graph.relation import Relation

from .reconciliation_service import Entity, Match, ReconciliationService, Rejection


class CachedService(ReconciliationService):
    """Runs `service` once per distinct query (its query_key), across queries, jobs and runs; results saved at `path`."""

    def __init__(self, service: ReconciliationService, path, view=None):
        # view: rebuilds cached targets with their properties (e.g. musicbrainz.artists); without one, a cached target
        #   comes back as just its id and text (e.g. an OpenRefine result)
        self.service = service
        self.path = Path(path)
        self.view = view
        self.cache: dict[str, list[dict]] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    self.cache[entry["key"]] = entry["matches"]

    def reconcile(self, query: Entity, targets=None) -> list[Match]:
        return self.reconcile_all([query], {query.id: targets})[query.id]

    def reconcile_all(self, queries: list[Entity], targets=None) -> dict[str, list[Match]]:
        fetched = self.fetch_all(queries, targets)
        return {q.id: self.service.keep(q, fetched[q.id]) for q in queries}

    def explain(self, query: Entity, targets=None, fetched=None) -> list[Rejection]:
        fetched = self.fetch_all([query], {query.id: targets} if targets is not None else None)[query.id]
        return self.service.explain(query, targets, fetched=fetched)

    def reasons(self) -> list[str]:
        return self.service.reasons()

    @property
    def label(self) -> str:
        return self.service.label

    def fetch_all(self, queries: list[Entity], targets=None) -> dict[str, list[Match]]:
        # the inner service's fetch_all (e.g. every search result, before any threshold), from the cache where possible
        targets_of = lambda q: targets[q.id] if targets else None
        keys = {q.id: self.service.query_key(q, targets_of(q)) for q in queries}
        misses = {}   # one query per uncached key
        for q in queries:
            if keys[q.id] is not None and keys[q.id] not in self.cache:
                misses.setdefault(keys[q.id], q)
        uncacheable = [q for q in queries if keys[q.id] is None]
        to_run = list(misses.values()) + uncacheable
        results = self.service.fetch_all(to_run, {q.id: targets_of(q) for q in to_run} if targets else None) if to_run else {}
        if misses:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                for key, q in misses.items():
                    self.cache[key] = [_to_dict(m) for m in results[q.id]]
                    f.write(json.dumps({"key": key, "matches": self.cache[key]}, ensure_ascii=False) + "\n")
        cached = {q.id: self.cache[keys[q.id]] for q in queries if keys[q.id] is not None}
        entities = self._entities({m["id"] for ms in cached.values() for m in ms})
        return {q.id: (results[q.id] if keys[q.id] is None else [_to_match(m, entities) for m in cached[q.id]])
                for q in queries}

    def _entities(self, ids) -> dict[str, Entity]:
        return {e.id: e for e in self.view.get(sorted(ids))} if self.view is not None and ids else {}

    def __repr__(self) -> str:
        return f"Cached({self.service!r})"


def _to_dict(m: Match) -> dict:
    return {"id": m.target.id, "text": m.target.text, "relation": m.relation.name if m.relation else None,
            "score": m.score, "reason": m.reason, "raw": m.raw}


def _to_match(d: dict, entities: dict[str, Entity]) -> Match:
    target = entities.get(d["id"]) or Entity(d["id"], d["text"])
    return Match(target, Relation[d["relation"]] if d["relation"] else None, d["score"], d["reason"], d["raw"])
