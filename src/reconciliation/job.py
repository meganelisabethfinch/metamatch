from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from data.dataset import View
from graph.relation import Relation

from .reconciliation_service import ReconciliationService


@dataclass(frozen=True)
class Job:
    """One reconciliation run: source -> target with a service, saved as an EdgeSet in datasets/metamatch/."""
    name: str
    source: Callable[[], View]                           # source view, e.g. graded_pieces.works
    target: Callable[[], View]                           # target view, e.g. musicbrainz.works
    service: Callable[..., ReconciliationService]        # reconciliation service, e.g. Cascade([Embedding, LLM])
    allowed: frozenset[Relation]                         # allowed relation types e.g. WORK_RELATIONS
    candidates_path: Callable[[], list] | None = None    # prefilter: loads the path EntityGraph.follow takes from each
                                                         # source entity to its candidate targets, e.g. MB works by the
                                                         # same composer; None = no prefilter
    takes_model: bool = False                            # the service uses an LLM, chosen by name (paths.yaml
                                                         # models.llm): service(model); each model's output is saved
                                                         # apart, so models can be compared

    def make_service(self, model: str | None = None) -> ReconciliationService:
        return self.service(model) if self.takes_model else self.service()

    def output_name(self, source: View, target: View, model: str | None = None) -> str:
        # e.g. "graded_pieces.works__musicbrainz.works.gp_works_mb.qwen35-9b.csv"
        return f"{source}__{target}.{self.name}{f'.{model}' if self.takes_model and model else ''}.csv"
