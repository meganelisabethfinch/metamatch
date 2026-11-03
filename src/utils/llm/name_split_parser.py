from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz
from unidecode import unidecode

from .parser import MalformedResponseError, Parser


@dataclass
class NameParts:
    entity_type: str                                       # "person", "group" or "traditional"
    substring: str = ""                                    # the part of the input naming this entity
    family_name: str = ""                                  # person
    given_names: list[str] = field(default_factory=list)   # person
    given_initials: list[str] = field(default_factory=list)  # person
    group_name: str = ""                                   # group
    descriptor: str = ""                                   # traditional, e.g. "Irish"


class NameSplitParser(Parser[NameParts]):
    """Parses the JSON array from name_split_prompt.txt: one NameParts per person / group / tradition."""

    def parse(self, response: str) -> list[NameParts]:
        # Qwen sometimes emits a stray '{}' inside / at the end of the array.
        text = re.sub(r"\}\s*,?\s*\{\s*\}", "}", response.strip())
        # ... and sometimes leaves off the array's closing ']'
        if text.startswith("[") and not text.endswith("]"):
            text += "]"
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise MalformedResponseError(f"not JSON: {response!r}") from e
        if not isinstance(data, list):
            raise MalformedResponseError(f"not a JSON array: {response!r}")
        return [self._to_name_parts(elem) for elem in data]

    def _to_name_parts(self, elem) -> NameParts:
        if not isinstance(elem, dict) or elem.get("entity_type") not in ("person", "group", "traditional"):
            raise MalformedResponseError(f"bad element: {elem!r}")
        parts = NameParts(**{k: v for k, v in elem.items() if k in NameParts.__dataclass_fields__})
        if parts.entity_type == "person" and not parts.family_name.strip():
            raise MalformedResponseError(f"person without family_name: {elem!r}")
        if parts.entity_type == "group" and not parts.group_name.strip():
            raise MalformedResponseError(f"group without group_name: {elem!r}")
        return initials_as_initials(parts)


def initials_as_initials(parts: NameParts) -> NameParts:
    # a single letter the LLM put in given_names (e.g. "VIVALDI, A." -> ["A"]) is an initial: moved to given_initials,
    # unless it is there already (the LLM sometimes gives it in both)
    initials = [g for g in parts.given_names if len(g.strip().rstrip(".")) == 1]
    if not initials:
        return parts
    parts.given_names = [g for g in parts.given_names if g not in initials]
    moved = [i for i in (g.strip().rstrip(".") for g in initials) if i not in parts.given_initials]
    parts.given_initials = [*moved, *parts.given_initials]
    return parts


def spelled_as_written(parts: NameParts, text: str) -> NameParts:
    # a name word the LLM spelled differently from the input (e.g. "CHEDEVILLE" -> "Cheville") is replaced by the
    # input's word it is closest to, in standard capitalisation; a word not close to any is left as it is. A name
    # whose words the input writes as one ("MADENSKI" -> "Maden ski") is that word
    written = [w.strip("'\"") for w in re.split(r"[\s.,;:()\[\]/&+]+", text)]   # e.g. "J.S.Bach", "Kander/Ebb"
    written = {_letters(w): w for w in written if _letters(w) and _letters(w) != "and"}
    if not written:
        return parts
    pieces = {_letters(p) for w in written.values() for p in re.split(r"[-']", w)}   # e.g. "Lobos" in "VILLA-LOBOS"

    def as_written(key):
        w = written[key]
        return "-".join(p.capitalize() for p in w.split("-")) if w.isupper() or w.islower() else w

    def respell(name):
        joined = _letters(name)
        if len(name.split()) > 1 and joined in written:
            return as_written(joined)
        words = []
        for word in name.split():
            key = _letters(word)
            if key and key not in written and key not in pieces:
                # a truncated word ("Kohl" for "KÖHLER") is the word it starts, else the closest word
                cut = [w for w in written if len(key) >= 3 and w.startswith(key)]
                best = min(cut, key=len) if cut else max(written, key=lambda w: fuzz.ratio(key, w))
                if cut or fuzz.ratio(key, best) >= SPELLING_THRESHOLD:
                    word = as_written(best)
            words.append(word)
        return " ".join(words)

    parts.family_name = respell(parts.family_name)
    parts.given_names = [respell(g) for g in parts.given_names]
    return parts


# How close (rapidfuzz ratio) a misspelled name word must be to the input's word to be replaced by it
SPELLING_THRESHOLD = 75


def _letters(word: str) -> str:
    return re.sub(r"[^a-z]", "", unidecode(word).lower())
