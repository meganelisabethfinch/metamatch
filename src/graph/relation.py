from __future__ import annotations

from enum import Enum


class Relation(Enum):
    """How one entity relates to another; also defines how edges invert and compose."""
    IS = "is"                  # identity: every kind of entity
    IS_NOT = "is not"
    IS_PART_OF = "is part of"  # parts / wholes: e.g. works
    CONTAINS = "contains"

    def inverse(self) -> Relation:
        # given a -[self]-> b, the relation b -> a
        return _INVERSES.get(self, self)

    def compose(self, other: Relation) -> Relation | None:
        # given a -[self]-> b and b -[other]-> c, the relation a -> c; None if it can't be inferred
        if self is Relation.IS:
            return other
        if other is Relation.IS:
            return self
        if self is other and self in (Relation.IS_PART_OF, Relation.CONTAINS):
            return self
        return None

    @property
    def is_match(self) -> bool:
        # whether an edge with this relation counts as a match
        return self is not Relation.IS_NOT


# The relations allowed between two kinds of entity
IDENTITY_RELATIONS = frozenset({Relation.IS, Relation.IS_NOT})                   # e.g. any <-> any
WORK_RELATIONS = IDENTITY_RELATIONS | {Relation.IS_PART_OF, Relation.CONTAINS}   # works <-> works

_INVERSES = {Relation.IS_PART_OF: Relation.CONTAINS}
_INVERSES |= {v: k for k, v in _INVERSES.items()}


def compose_relations(r1: Relation | None, r2: Relation | None) -> tuple[bool, Relation | None]:
    # (whether a -[r1]-> b -[r2]-> c gives an edge a -> c, and its relation); None = an untyped edge:
    # composed with a match it stays untyped, with a non-match it gives no edge
    if r1 is None and r2 is None:
        return True, None
    if r1 is None or r2 is None:
        typed = r2 if r1 is None else r1
        return typed.is_match, None
    relation = r1.compose(r2)
    return relation is not None, relation
