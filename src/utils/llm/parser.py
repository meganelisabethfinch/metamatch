from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from graph.relation import Relation

T = TypeVar("T")


class MalformedResponseError(ValueError):
    """The LLM's response could not be parsed."""


@dataclass
class Answer:
    relation: Relation
    score: float | None = None     # e.g. a confidence, if the prompt asks for one
    reason: str | None = None      # the LLM's rationale, if the prompt asks for one


class Parser(ABC, Generic[T]):
    @abstractmethod
    def parse(self, response: str) -> list[T]:
        """One item per thing the prompt asks about, in order; raises MalformedResponseError."""
