from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .edge import Edge
from .relation import Relation, compose_relations


class EdgeSet():
    # The edges between the entities of TWO views, e.g. graded_pieces.composer_credits -> musicbrainz.artists.
    # Either produced by reconciliation (reconcile.py), or a relation a dataset already holds
    # (e.g. graded_pieces.works -> graded_pieces.composer_credits: who wrote each piece).

    def __init__(self, source, target, edges, allowed=None, meta=None):
        # source / target: Views (or their names, e.g. "graded_pieces.works"); only the name is kept
        # allowed: the relations edges may have (e.g. IDENTITY_RELATIONS); None for any. Untyped edges are always allowed
        # meta: how it was made: method, settings, date; or the edge sets it was composed from
        self.source = str(source)
        self.target = str(target)
        self.edges: list[Edge] = list(edges)
        self.allowed: frozenset[Relation] | None = frozenset(allowed) if allowed is not None else None
        self.meta: dict = meta or {}
        if self.allowed is not None:
            for e in self.edges:
                if e.relation is not None and e.relation not in self.allowed:
                    raise ValueError(f"{self.source} -> {self.target} doesn't allow {e.relation.name}: {e}")
        self._by_source: dict[str, list[Edge]] = defaultdict(list)
        self._by_target: dict[str, list[Edge]] = defaultdict(list)
        for e in self.edges:
            self._by_source[e.source_id].append(e)
            self._by_target[e.target_id].append(e)

    @classmethod
    def from_matches(cls, source, target, matches, allowed=None, meta=None, provenance=None) -> EdgeSet:
        # reconcile() output ({source_id: [Match]}) -> EdgeSet, one edge per Match, each made by provenance
        edges = [Edge.from_match(source_id, m, provenance) for source_id, ms in matches.items() for m in ms]
        return cls(source, target, edges, allowed, meta)

    @classmethod
    def from_csv(cls, path) -> EdgeSet:
        # datasets/metamatch/<source>__<target>.<method>.csv (+ a .json alongside for source, target, allowed, meta)
        # edges saved without a provenance get the file's (meta["provenance"]), if it has one
        path = Path(path)
        info = json.loads(path.with_suffix(".json").read_text())
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        default = info["meta"].get("provenance")
        edges = [
            Edge(r["source_id"], r["target_id"],
                 Relation[r["relation"]] if r["relation"] else None,
                 float(r["score"]) if r["score"] else None,
                 r["reason"] or None,
                 r.get("provenance") or default)
            for r in df.to_dict("records")
        ]
        allowed = {Relation[name] for name in info["allowed"]} if info["allowed"] is not None else None
        return cls(info["source"], info["target"], edges, allowed, info["meta"])

    def to_csv(self, path) -> None:
        # one row per edge: source_id, target_id, relation (the enum member's name), score, reason, provenance
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [(e.source_id, e.target_id, e.relation.name if e.relation else "", e.score, e.reason, e.provenance)
             for e in self.edges],
            columns=["source_id", "target_id", "relation", "score", "reason", "provenance"],
        ).to_csv(path, index=False)
        allowed = sorted(r.name for r in self.allowed) if self.allowed is not None else None
        info = {"source": self.source, "target": self.target, "allowed": allowed, "meta": self.meta}
        path.with_suffix(".json").write_text(json.dumps(info, indent=2, ensure_ascii=False))

    def targets(self, source_id) -> list[Edge]:
        # edges from source_id
        return list(self._by_source.get(source_id, []))

    def sources(self, target_id) -> list[Edge]:
        # edges into target_id
        return list(self._by_target.get(target_id, []))

    def filter(self, relations=None, min_score=None) -> EdgeSet:
        # a new EdgeSet with only the edges whose relation is in relations / score >= min_score (e.g. only IS)
        edges = [e for e in self.edges
                 if (relations is None or e.relation in relations)
                 and (min_score is None or (e.score is not None and e.score >= min_score))]
        return EdgeSet(self.source, self.target, edges, self.allowed, self.meta)

    def inverse(self) -> EdgeSet:
        # target -> source; each relation replaced by its inverse (IS_PART_OF <-> CONTAINS, IS stays IS)
        edges = [Edge(e.target_id, e.source_id, e.relation.inverse() if e.relation else None, e.score, e.reason,
                      e.provenance) for e in self.edges]
        allowed = {r.inverse() for r in self.allowed} if self.allowed is not None else None
        return EdgeSet(self.target, self.source, edges, allowed, {"inverse_of": self._describe()})

    def compose(self, other: EdgeSet) -> EdgeSet:
        # self: A -> B, other: B -> C  =>  a new EdgeSet A -> C, with an edge (a, c) wherever a -> b -> c
        # relation of a composed edge: Relation.compose of the two (e.g. IS, IS_PART_OF -> IS_PART_OF); pairs whose
        # relations don't compose (e.g. IS_NOT, IS_NOT) give no edge
        if self.target != other.source:
            raise ValueError(f"can't compose {self.source} -> {self.target} with {other.source} -> {other.target}")
        composed = {}
        for e1 in self.edges:
            for e2 in other.targets(e1.target_id):
                composes, relation = compose_relations(e1.relation, e2.relation)
                if composes:
                    composed.setdefault((e1.source_id, e2.target_id, relation), Edge(e1.source_id, e2.target_id, relation))
        return EdgeSet(self.source, other.target, list(composed.values()), _either(self.allowed, other.allowed),
                       {"composed_of": [self._describe(), other._describe()]})

    def union(self, *others: EdgeSet) -> EdgeSet:
        # several EdgeSets between the same two views (e.g. from different methods) -> one EdgeSet
        for o in others:
            if (o.source, o.target) != (self.source, self.target):
                raise ValueError(f"can't union {self.source} -> {self.target} with {o.source} -> {o.target}")
        edge_sets = [self, *others]
        edges = list(dict.fromkeys(e for s in edge_sets for e in s.edges))
        allowed = None if any(s.allowed is None for s in edge_sets) else frozenset().union(*(s.allowed for s in edge_sets))
        return EdgeSet(self.source, self.target, edges, allowed, {"union_of": [s._describe() for s in edge_sets]})

    def match_rate(self, source) -> float:
        # fraction of the source view's entities with at least one edge (excluding IS_NOT edges)
        # source: the View that was reconciled (e.g. a piano-only view, if that's what was run)
        entity_ids = {e.id for e in source.entities()}
        matched = {e.source_id for e in self.edges if e.relation is None or e.relation.is_match}
        return len(matched & entity_ids) / len(entity_ids) if entity_ids else 0.0

    def __len__(self) -> int:
        return len(self.edges)

    def __repr__(self) -> str:
        return f"EdgeSet({self.source} -> {self.target}, {len(self.edges)} edges)"

    def _describe(self) -> dict:
        return {"source": self.source, "target": self.target, "meta": self.meta}


def _either(a, b):
    # the relations allowed by either set; None (any) if either is None
    return None if a is None or b is None else a | b
