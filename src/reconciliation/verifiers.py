from __future__ import annotations

import re

from rapidfuzz import fuzz

from constants.wikidata import NOTABLE_OCCUPATION_QIDS
from utils.names import full_name_forms, norm_family_name, norm_name

from .reconciliation_service import Entity
from .verifier_service import AllOf, AnyOf, PropertyVerifier

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


class FamilyName(PropertyVerifier):
    """The query's family name matches the target's, ignoring particles ("Beethoven" / "van Beethoven") and however
    a compound name is split ("Villa-Lobos" / "Villa", "Lobos"); a single letter (e.g. "Wolfgang M.") is an
    initial, which need only start one of them."""
    failure = "mismatched family name"

    def check(self, query: Entity, target: Entity) -> bool | None:
        q = _words(norm_family_name(query.properties.get("family_name", "")))
        t = [w for f in target.properties.get("family_names", []) for w in _words(norm_family_name(f))]
        if not q or not t:
            return None
        return _match_words(q, t)


class GivenNames(PropertyVerifier):
    """The query's given names match the target's, in any order: in full where the query has them in full, by
    initial where it has only the initial (e.g. "Reinagle, A." passes Alexander and Alice; "Reinagle, Alexander" only
    Alexander). A target may list more given names than the query, or fewer."""
    failure = "mismatched given names"

    def check(self, query: Entity, target: Entity) -> bool | None:
        p = query.properties
        q = [w for g in p.get("given_names", []) + p.get("given_initials", []) for w in _words(g)]
        t = [w for g in target.properties.get("given_names", []) for w in _words(g)]
        if not q or not t:
            return None
        return _match_words(q, t)


class FullName(PropertyVerifier):
    """A way of writing the query's full name is one of the target's names (label or aliases), word for word, e.g.
    a stage name whose family name differs ("Elton John"), or a spelling of the name Wikidata lists only as an alias
    ("Nikolai Rimsky-Korsakov"). A name of two words or more may leave out some of the target name's (e.g.
    "J. Dussek" / "Jan Ladislav Dussek"); a single word must be the whole name. Only ever passes: no such name is not
    held against the target."""

    def check(self, query: Entity, target: Entity) -> bool | None:
        p = query.properties
        q = [_words(f) for f in full_name_forms(p.get("original", query.text), p.get("family_name", ""),
                                                p.get("given_names", []), p.get("given_initials", []))]
        t = [_words(a) for a in target.properties.get("aliases", [])]
        return True if any(_names_match(f, a) for f in q if f for a in t if a) else None


def name_check() -> AnyOf:
    # a full name among the target's names passes outright; otherwise the family and given names must both match
    # where both sides have them (missing names are never held against the target; nothing to compare = a pass)
    return AnyOf([FullName(), AllOf([FamilyName(), GivenNames()])], name="name", failure="mismatched name/initials")


# Words this long or longer may differ a little (a typo, e.g. "Albononi" / "Albinoni") and still match: similarity
# (rapidfuzz ratio) at least FUZZY_THRESHOLD, after folding SPELLINGS. Shorter words must be equal ("Bach" / "Bath")
FUZZY_MIN_LENGTH = 6
FUZZY_THRESHOLD = 93

# Spellings folded together before words are compared: transliterations of one name (e.g. "Nikolay" / "Nicolai",
# "Dmitry" / "Dmitri", "Tschaikowsky" / "Tchaikovsky", "Mussorgsky" / "Musorgsky")
SPELLINGS = [("ph", "f"), ("k", "c"), ("y", "i"), ("w", "v")]


def _words(name: str) -> list[str]:
    # a name's words, compared one by one, ignoring case, accents and punctuation ("BARTOK" / "Bartók", "O'Brien" /
    # "OBrien"); hyphens, full stops, commas and slashes separate words ("Villa-Lobos", "J.S.")
    words = (re.sub(r"[^a-z0-9]", "", w) for w in norm_name(re.sub(r"[-.,/]", " ", name or "")).split())
    return [w for w in words if w]


def _spelling(word: str) -> str:
    for a, b in SPELLINGS:
        word = word.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", word)   # doubled letters as one


def _same_word(a: str, b: str) -> bool:
    # a single letter is an initial: it matches any word it starts (whether it was written as an initial or not)
    if len(a) == 1 or len(b) == 1:
        return a[0] == b[0]
    a, b = _spelling(a), _spelling(b)
    if a == b:
        return True
    return min(len(a), len(b)) >= FUZZY_MIN_LENGTH and fuzz.ratio(a, b) >= FUZZY_THRESHOLD


def _names_match(q: list[str], t: list[str]) -> bool:
    # every word of the query's name is one of the target name's (in any order), and nothing is left over unless the
    # query's name has two words or more
    if len(q) > len(t) or (len(q) < len(t) and len(q) < 2):
        return False
    return _match_words(q, t)


def _match_words(q: list[str], t: list[str]) -> bool:
    # every word of the shorter list pairs with a different matching word of the other (in any order); whole words
    # are paired first, so an initial can't take the word a whole word needs
    remaining = list(t)
    matched = 0
    for word in sorted(q, key=lambda w: len(w) == 1):
        hit = next((r for r in remaining if len(r) > 1 and _same_word(word, r)), None) if len(word) > 1 else None
        hit = hit or next((r for r in remaining if _same_word(word, r)), None)
        if hit:
            remaining.remove(hit)
            matched += 1
    return matched >= min(len(q), len(t))
