from __future__ import annotations

import json
import logging
import time

import requests

from utils.http import describe

from .reconciliation_service import Entity, Match, ReconciliationService, Rejection, ServiceUnavailable

logger = logging.getLogger(__name__)

WIKIDATA_ENDPOINT = "https://wikidata.reconci.link/en/api"


class OpenRefineService(ReconciliationService):
    """
    Reconciles against a remote dataset (e.g. Wikidata) via a W3C / OpenRefine
    reconciliation API. ``targets`` is ignored: the endpoint searches its own dataset.
    """

    def __init__(
        self,
        endpoint: str = WIKIDATA_ENDPOINT,
        type_id: str | None = None,
        properties: list[dict] | None = None,
        limit: int = 5,
        threshold: float | None = None,
        batch_size: int = 5,
        delay: float = 1.5,
        timeout: float = 60,
        retries: int = 12,
        max_wait: float = 300,
        empty_rechecks: int = 2,
        recheck_wait: float = 5,
        query_text=None,
    ):
        # query_text: optional function query entity -> the text to search for (default: its text)
        # retries / max_wait: the public endpoint fails in bursts (Wikidata rate-limits it), so failed requests are
        # retried with backoff 2, 4, 8, ... s, capped at max_wait: about 20 minutes in all before giving up
        # empty_rechecks / recheck_wait: when rate-limited it also answers some queries with no candidates instead of
        # failing (about 1 in 10, at random), so a query with no candidates is asked again, up to empty_rechecks times
        # (recheck_wait s apart), before its empty answer is believed
        self.endpoint = endpoint
        self.query_text = query_text
        self.type_id = type_id
        self.properties = properties
        self.limit = limit
        self.threshold = threshold
        self.batch_size = batch_size
        self.delay = delay
        self.timeout = timeout
        self.retries = retries
        self.max_wait = max_wait
        self.empty_rechecks = empty_rechecks
        self.recheck_wait = recheck_wait
        self.session = requests.Session()
        self.session.headers["User-Agent"] = (
            "MetaMatch/0.1 (https://github.com/meganelisabethfinch/metamatch) "
            f"python-requests/{requests.__version__}"
        )

    def reconcile(self, query: Entity, targets=None) -> list[Match]:
        return self.reconcile_all([query], targets)[query.id]

    def reconcile_all(self, queries: list[Entity], targets=None) -> dict[str, list[Match]]:
        return {q: self.keep(None, ms) for q, ms in self.fetch_all(queries, targets).items()}

    def fetch_all(self, queries: list[Entity], targets=None) -> dict[str, list[Match]]:
        # every candidate the API returns (up to limit), best first; the threshold is applied by keep()
        results = self._fetch(queries)
        for _ in range(self.empty_rechecks):
            empty = [q for q in queries if not results[q.id]]
            if not empty:
                break
            time.sleep(self.recheck_wait)
            results.update(self._fetch(empty))
        return results

    def _fetch(self, queries: list[Entity]) -> dict[str, list[Match]]:
        results = {}
        for start in range(0, len(queries), self.batch_size):
            if start > 0:
                time.sleep(self.delay)
            batch = queries[start:start + self.batch_size]
            response = self._post({f"q{i}": self._query(q) for i, q in enumerate(batch)})
            for i, q in enumerate(batch):
                candidates = response.get(f"q{i}", {}).get("result", [])
                results[q.id] = self._to_matches(candidates)
        return results

    def keep(self, query, matches: list[Match]) -> list[Match]:
        return [m for m in matches if self.threshold is None or m.score >= self.threshold]

    def explain(self, query: Entity, targets=None, fetched=None) -> list[Rejection]:
        # the candidates the search returned but the threshold rejected
        fetched = self.fetch_all([query], targets)[query.id] if fetched is None else fetched
        kept = {m.target.id for m in self.keep(query, fetched)}
        return [Rejection(m.target.id, "score below threshold", score=m.score)
                for m in fetched if m.target.id not in kept]

    def reasons(self) -> list[str]:
        return ["score below threshold"]

    def query_key(self, query: Entity, targets=None) -> str:
        return json.dumps({"service": "openrefine", "endpoint": self.endpoint, **self._query(query)},
                          sort_keys=True, ensure_ascii=False)

    def __repr__(self) -> str:
        return f"OpenRefine(threshold={self.threshold}, type={self.type_id}, properties={self.properties})"

    def _query(self, query: Entity) -> dict:
        text = self.query_text(query) if self.query_text else query.text
        q = {"query": text.strip(), "limit": self.limit}
        if self.type_id:
            q["type"] = self.type_id
        if self.properties:
            q["properties"] = self.properties
        return q

    def _post(self, queries: dict) -> dict:
        """POST a batch of queries, retrying with exponential backoff."""
        for attempt in range(self.retries):
            try:
                resp = self.session.post(
                    self.endpoint, data={"queries": json.dumps(queries)}, timeout=self.timeout
                )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                if attempt == self.retries - 1:
                    raise ServiceUnavailable(f"OpenRefine at {self.endpoint}: {describe(e)} "
                                             f"({self.retries} attempts)") from e
                wait = min(2 ** (attempt + 1), self.max_wait)
                logger.warning(f"OpenRefine request failed (attempt {attempt + 1}/{self.retries}): {describe(e)}; "
                               f"retrying in {wait}s")
                time.sleep(wait)

    def _to_matches(self, candidates: list[dict]) -> list[Match]:
        matches = [
            Match(Entity(c["id"], c.get("name", "")), score=c.get("score", 0), raw=json.dumps(c, ensure_ascii=False))
            for c in candidates
        ]
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches
