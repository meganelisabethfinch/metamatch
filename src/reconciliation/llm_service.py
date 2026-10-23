from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from utils.llm.parser import MalformedResponseError, Answer, Parser

from .reconciliation_service import Entity, Match, ReconciliationService

if TYPE_CHECKING:
    from utils.llm.model import LocalLLM, RemoteLLM

logger = logging.getLogger(__name__)


class LLMService(ReconciliationService):
    def __init__(
        self,
        llm: LocalLLM | RemoteLLM,
        template: str,
        parser: Parser[Answer],
        targets_per_prompt: int = 1,
        batch: bool | None = None,
    ):
        # llm: e.g. utils.llm.model.llm("qwen35-9b"); anything with a name, its settings and generate(prompts, ...)
        # batch: run all the prompts of one call together (local: GPU batches; remote: one Anthropic batch, half
        #   price), rather than one at a time; None: the model's default (llm.batch)
        self.llm = llm
        self.template = template
        self.parser = parser
        self.targets_per_prompt = targets_per_prompt
        self.batch = llm.batch if batch is None else batch
        self.malformed_count = 0

    def reconcile(self, query: Entity, targets: list[Entity]) -> list[Match]:
        return self.reconcile_all([query], {query.id: targets})[query.id]

    def reconcile_all(self, queries: list[Entity], targets: dict[str, list[Entity]]) -> dict[str, list[Match]]:
        # every query's prompts, run in one generate call
        n = self.targets_per_prompt
        jobs = [(q, targets[q.id][i:i + n]) for q in queries for i in range(0, len(targets[q.id]), n)]
        responses = self.generate([self.render(q, chunk) for q, chunk in jobs]) if jobs else []
        matches = {q.id: [] for q in queries}
        for (query, chunk), response in zip(jobs, responses):
            try:
                answers = self.parser.parse(response)
                if len(answers) != len(chunk):
                    raise MalformedResponseError(f"{len(answers)} answers for {len(chunk)} targets")
            except MalformedResponseError as e:
                self.malformed_count += 1
                logger.warning(f"Malformed LLM response for query {query.id!r}: {e}")
                continue
            matches[query.id] += [
                Match(target, relation=a.relation, score=a.score, reason=a.reason, raw=response)
                for target, a in zip(chunk, answers)
            ]
        return matches

    def query_key(self, query: Entity, targets: list[Entity]) -> str:
        n = self.targets_per_prompt
        prompts = [self.render(query, targets[i:i + n]) for i in range(0, len(targets), n)]
        return json.dumps({"service": "llm", "model": self.llm.name, "generate_kwargs": self.llm.settings,
                           "prompts": prompts, "targets": [t.id for t in targets]}, sort_keys=True, ensure_ascii=False)

    def __repr__(self) -> str:
        return f"LLM({self.llm.name}, targets_per_prompt={self.targets_per_prompt}, batch={self.batch})"

    def render(self, query: Entity, targets: list[Entity]) -> str:
        if self.targets_per_prompt == 1:
            targets_text = targets[0].text
        else:
            targets_text = "\n".join(f"{i}. {t.text}" for i, t in enumerate(targets, 1))
        # side A: the query's "query_text" property if it has one (e.g. an IMSLP work, whose text is how it reads as a
        # target), else its text. str.replace rather than str.format: templates may contain literal braces.
        query_text = query.properties.get("query_text") or query.text
        return self.template.replace("{query}", query_text).replace("{targets}", targets_text)

    def generate(self, prompts: list[str]) -> list[str]:
        """Run the prompts through the LLM; returns one response string per prompt."""
        # the template up to the query is the same in every prompt (a remote model caches it)
        return self.llm.generate(prompts, prefix=self.template.split("{query}")[0], batch=self.batch)
