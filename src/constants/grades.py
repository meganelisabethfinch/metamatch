# Per exam board: grade -> label templates naming that grade ({x} = any text, e.g. the instrument, which is already in
# the instrument column). The grades' order is in datasets/eval/difficulty.json.
GRADE_LABELS = {
    "RCM": {
        **{f"Level {n}": [f"Grade {n}"] for n in range(1, 11)},   # newer syllabi say Grade for Level
        "ARCT": ["ARCT in {x}Performance", "Performer’s ARCT", "Associate Diploma (ARCT) in {x}, {x}"],
        "LRCM": ["Licentiate Diploma (LRCM) in {x}, {x}"],
    },
    "RIAM": {
        "Primary": ["Primary Grade"],   # the recorder syllabus's name for it
        **{str(n): [f"Grade {roman}"] for n, roman in enumerate(["I", "II", "III", "IV", "V", "VI", "VII", "VIII"], 1)},
    },
}
