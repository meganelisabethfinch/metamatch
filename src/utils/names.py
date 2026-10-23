from __future__ import annotations

from unidecode import unidecode

from constants.name_particles import FAMILY_AFFIXES, NAME_PARTICLES, PARTICLES


def norm_name(s: str) -> str:
    # for whole-string comparison: no accents, lower case, single spaces
    return " ".join(unidecode(s or "").lower().split())


def norm_family_name(s: str) -> str:
    # norm_name without leading particles / affixes, e.g. "van der Berg" -> "berg", "O'Brien" -> "brien"
    n = norm_name(s)
    changed = True
    while n and changed:
        changed = False
        affix = next((a for a in FAMILY_AFFIXES if n.startswith(a)), None)
        if affix:
            n, changed = n[len(affix):].strip(), True
            continue
        head, _, rest = n.partition(" ")
        if rest and head in NAME_PARTICLES | PARTICLES:
            n, changed = rest.strip(), True
    return n


def full_name_forms(original: str, family_name: str, given_names: list[str], given_initials: list[str]) -> set[str]:
    # ways the full name could be written, e.g. "Van Heusen, Jimmy" -> also "Jimmy Van Heusen"
    forms = set()

    def add(s, reverse=False):
        if s and s.strip():
            forms.add(s.strip())
            if reverse and "," in s:
                left, _, right = s.partition(",")
                forms.add(f"{right.strip()} {left.strip()}")

    add(original, reverse=True)
    family = (family_name or "").strip()
    givens = [g.strip() for g in given_names if g.strip()]
    dotted = " ".join(f"{i.strip().rstrip('.').upper()}." for i in given_initials if i.strip())
    if family and givens and not dotted:
        add(" ".join(givens + [family]))
        add(f"{family}, {' '.join(givens)}")
    if family and dotted and not givens:
        add(f"{dotted} {family}")
        add(f"{family}, {dotted}")
    if family and givens and dotted:
        add(" ".join(givens + [dotted, family]))
        add(f"{family}, {' '.join(givens)} {dotted}")
    return forms


def split_sort_name(name: str) -> dict:
    # name parts of a "Family, Given" name (e.g. "Bach, Johann Sebastian", "Adam, J."); no comma = family name only
    family, _, given = name.partition(",")
    tokens = given.split()
    return {"original": name, "family_name": family.strip(),
            "given_names": [t for t in tokens if len(t.rstrip(".")) > 1],
            "given_initials": [t.rstrip(".") for t in tokens if len(t.rstrip(".")) == 1]}


def is_family_name_only(parts: dict) -> bool:
    # name parts with a family name but no given names or initials (e.g. "Burgon")
    return bool(parts.get("family_name", "").strip()) and not parts.get("given_names") and not parts.get("given_initials")


def fullest_name(parts: dict, default: str) -> str:
    # the fullest name the parts give: "Given Family", else "I I Family" (initials), else "Family"; else default
    family = parts.get("family_name", "").strip()
    if not family:
        return default
    given = [g.strip() for g in parts.get("given_names", []) if g.strip()]
    initials = [i.strip() for i in parts.get("given_initials", []) if i.strip()]
    return " ".join((given or initials) + [family])
