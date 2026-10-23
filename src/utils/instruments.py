from __future__ import annotations

import re

from constants.instruments import _SAX_SUBTYPES, INSTRUMENT_MAP


def canonical_instruments(text: str) -> list[tuple[str, str]]:
    # (family, canonical name) for each instrument in text, e.g. "alto/baritone saxophone" ->
    # [("woodwind", "saxophone, alto"), ("woodwind", "saxophone, baritone")]; an unknown instrument -> ("", as written)
    return [_canonical(name) for name in _split(text.strip())] if text.strip() else []


def _split(text: str) -> list[str]:
    # "alto/baritone saxophone", "descant & treble recorder" -> one name per variant; anything else as it is
    if m := re.fullmatch(r"(.+[/&].+?)\s+(\S+)", text):
        variants = [v.strip() for v in re.split(r"\s*(?:/|&)\s*", m.group(1)) if v.strip()]
        return [f"{v} {m.group(2)}" for v in variants]
    return [text]


def _from_code(code: str) -> tuple[str, str]:
    # "brass/horn/french" -> ("brass", "horn, french")
    family, *rest = code.split("/")
    return family, ", ".join(rest)


# canonical name -> family, e.g. "horn, french" -> "brass" (the names the cleaned data already uses)
_CANONICAL = dict(reversed(_from_code(code)) for code in INSTRUMENT_MAP.values())


def _canonical(name: str) -> tuple[str, str]:
    if name.strip().lower() in _CANONICAL:
        return _CANONICAL[name.strip().lower()], name.strip().lower()
    stripped = re.sub(r"^jazz\s+", "", name.strip(), flags=re.IGNORECASE)
    code = INSTRUMENT_MAP.get(stripped.lower()) or INSTRUMENT_MAP.get(name.strip().lower())
    if not code:
        if m := re.match(r"^(?:saxophone|sax)[,\s]+([a-z]+)$", stripped.lower()):
            return "woodwind", f"saxophone, {_SAX_SUBTYPES.get(m.group(1), m.group(1))}"
        return "", name.strip()
    return _from_code(code)
