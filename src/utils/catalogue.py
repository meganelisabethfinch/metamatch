from __future__ import annotations

import re

from constants.catalogue_patterns import CATALOGUE_PATTERNS


def catalogue_numbers(text: str) -> list[dict]:
    # the catalogue numbers in text (BWV, Op., K., WoO, HWV, D., Hob., ...), as [{system, value}], each value once
    found, seen = [], set()
    for pattern, system, formatter in CATALOGUE_PATTERNS:
        for m in re.finditer(pattern, text or "", re.IGNORECASE):
            value = formatter(m)
            if value not in seen:
                seen.add(value)
                found.append({"system": system, "value": value})
    return found


def with_catalogue_numbers(title: str) -> str:
    # e.g. "Sinfonia Nr. 3 D-Dur BWV 789" -> "Sinfonia Nr. 3 D-Dur BWV 789, BWV 789"
    values = ", ".join(c["value"] for c in catalogue_numbers(title))
    return f"{title}, {values}" if values else title
