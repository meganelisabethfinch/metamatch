from __future__ import annotations

from constants.wikidata import NOTABLE_OCCUPATION_QIDS
from utils.names import full_name_forms, norm_family_name, norm_name

from .reconciliation_service import Entity
from .verifier_service import AnyOf, PropertyVerifier

# Person checks. Query properties (e.g. graded_pieces.composer_credits): family_name, given_names, given_initials.
# Target properties (e.g. wikidata.entities): given_names, family_names, aliases (incl. the label),
# occupations, instruments, has_music_occupation, has_works. Each check returns None when its properties are missing.


class MusicOccupation(PropertyVerifier):
    """The target has a music-related occupation (or plays an instrument)."""
    failure = "no music-related occupation"

    def check(self, query: Entity, target: Entity) -> bool | None:
        if "has_music_occupation" not in target.properties:
            return None
        return bool(target.properties["has_music_occupation"] or target.properties.get("instruments"))


class NotableOccupation(PropertyVerifier):
    """The target has a broadly notable occupation (writer, painter, ...)."""
    failure = "no notable occupation"

    def check(self, query: Entity, target: Entity) -> bool | None:
        if "occupations" not in target.properties:
            return None
        return any(o in NOTABLE_OCCUPATION_QIDS for o in target.properties["occupations"])


class HasWorks(PropertyVerifier):
    """The target is credited with at least one work (e.g. as a composer in MusicBrainz)."""
    failure = "no credited works"

    def check(self, query: Entity, target: Entity) -> bool | None:
        return target.properties.get("has_works")


class GivenNames(PropertyVerifier):
    """The query's given names match the target's."""
    failure = "mismatched given names"

    def check(self, query: Entity, target: Entity) -> bool | None:
        q = [norm_name(g) for g in query.properties.get("given_names", []) if g.strip()]
        t = [norm_name(g) for g in target.properties.get("given_names", []) if g.strip()]
        return _one_to_one(q, t) if q and t else None


class GivenInitials(PropertyVerifier):
    """The query's given-name initials match the target's (e.g. tells "Reinagle, A." from "Reinagle, J.")."""
    failure = "mismatched initials"

    def check(self, query: Entity, target: Entity) -> bool | None:
        given = query.properties.get("given_names", []) + query.properties.get("given_initials", [])
        q = [norm_name(g)[0].upper() for g in given if norm_name(g)]
        t = [norm_name(g)[0].upper() for g in target.properties.get("given_names", []) if norm_name(g)]
        return _one_to_one(q, t) if q and t else None


class FullName(PropertyVerifier):
    """A way of writing the query's full name is one of the target's names (label or aliases)."""
    failure = "full name not among the target's names"

    def check(self, query: Entity, target: Entity) -> bool | None:
        p = query.properties
        q = {norm_name(f) for f in full_name_forms(p.get("original", query.text), p.get("family_name", ""),
                                                   p.get("given_names", []), p.get("given_initials", []))} - {""}
        t = {norm_name(a) for a in target.properties.get("aliases", [])} - {""}
        return bool(q & t) if q and t else None


class FamilyNameOnly(PropertyVerifier):
    """For a query that is only a family name (e.g. "Burgon"): it matches one of the target's family names or names."""
    failure = "mismatched family name"

    def check(self, query: Entity, target: Entity) -> bool | None:
        p = query.properties
        family = p.get("family_name", "").strip()
        if not family or p.get("given_names") or p.get("given_initials"):
            return None
        t = ({norm_family_name(f) for f in target.properties.get("family_names", [])}
             | {norm_name(a) for a in target.properties.get("aliases", [])}) - {""}
        return bool({norm_name(family), norm_family_name(family)} & t) if t else None


def name_check() -> AnyOf:
    # passes if any of the name checks that apply passes
    return AnyOf([GivenNames(), GivenInitials(), FullName(), FamilyNameOnly()], name="name",
                 failure="mismatched name/initials")


def _one_to_one(q: list[str], t: list[str]) -> bool:
    # every token of the shorter list pairs with a different equal token in the other
    remaining = list(t)
    matched = 0
    for token in q:
        if token in remaining:
            remaining.remove(token)
            matched += 1
    return matched >= min(len(q), len(t))
