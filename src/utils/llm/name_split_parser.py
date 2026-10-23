from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

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
        return parts
