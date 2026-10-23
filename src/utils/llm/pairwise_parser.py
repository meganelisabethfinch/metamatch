from __future__ import annotations

import re

from graph.relation import Relation

from .parser import MalformedResponseError, Answer, Parser


class PairwiseParser(Parser[Answer]):
    """Parses a one-target response: a bare label (direct), or "Reasoning: ...\\nAnswer: <label>" (``cot=True``)."""

    def __init__(self, labels: dict[str, Relation], cot: bool = False):
        # labels: the prompt's answer labels and the relation each means, e.g. {"The same piece": Relation.IS, ...}
        self.labels = {label.lower(): relation for label, relation in labels.items()}
        self.cot = cot

    def parse(self, response: str) -> list[Answer]:
        if not self.cot:
            return [Answer(self._to_relation(response))]
        parts = re.split(r"(?i)answer\s*:", response)
        if len(parts) == 1:
            raise MalformedResponseError(f"no 'Answer:' in {response!r}")
        answer = parts[-1].strip().split("\n")[0]
        reasons = re.split(r"(?i)reasoning\s*:", "".join(parts[:-1]))
        reason = reasons[-1].strip() if len(reasons) > 1 else None
        return [Answer(self._to_relation(answer), reason=reason)]

    def _to_relation(self, text: str) -> Relation:
        """The relation for the label ``text`` is, ignoring case, whitespace and surrounding punctuation / markdown."""
        key = text.strip().strip(".\"'`*").strip().lower()
        if key not in self.labels:
            raise MalformedResponseError(f"not a label: {text!r}")
        return self.labels[key]
