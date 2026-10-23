from __future__ import annotations

import logging
import time
from collections import defaultdict

import pandas as pd
import requests

from constants.wikidata import (
    MUSIC_OCCUPATION_QIDS, P_FAMILY_NAME, P_GIVEN_NAME, P_INSTRUMENT, P_OCCUPATION, P_PSEUDONYM,
    P_SERIES_ORDINAL, P_SUBCLASS_OF,
)
from reconciliation.reconciliation_service import Entity, ServiceUnavailable
from utils.http import describe

from .dataset import Dataset, View

logger = logging.getLogger(__name__)

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "MetaMatch/0.1 (https://github.com/meganelisabethfinch/metamatch)"

_PERSON_QUERY = """
SELECT ?item ?occ ?instr ?is_musician WHERE {
  VALUES ?item { %(items)s }
  OPTIONAL { ?item wdt:%(occ)s ?occ . }
  OPTIONAL { ?item wdt:%(instr)s ?instr . }
  BIND(EXISTS { ?item wdt:%(occ)s/wdt:%(sub)s* ?m . VALUES ?m { %(music)s } } AS ?is_musician)
}"""
_GIVEN_NAMES_QUERY = """
SELECT ?item ?name ?ordinal WHERE {
  VALUES ?item { %(items)s }
  ?item p:%(given)s ?st . ?st ps:%(given)s ?g . ?g rdfs:label ?name . FILTER(LANG(?name) = "en")
  OPTIONAL { ?st pq:%(ordinal)s ?ordinal . }
}"""
_FAMILY_NAMES_QUERY = """
SELECT ?item ?name WHERE {
  VALUES ?item { %(items)s }
  ?item wdt:%(family)s ?f . ?f rdfs:label ?name . FILTER(LANG(?name) = "en")
}"""
_NAMES_QUERY = """
SELECT ?item ?name ?kind WHERE {
  VALUES ?item { %(items)s }
  { ?item rdfs:label ?name . FILTER(LANG(?name) = "en") BIND("label" AS ?kind) }
  UNION { ?item skos:altLabel ?name . FILTER(LANG(?name) = "en") BIND("alias" AS ?kind) }
  UNION { ?item wdt:%(pseudonym)s ?name . BIND("alias" AS ?kind) }
}"""


class WikiData(Dataset):
    # The Wikidata entities fetched so far (a cached dataset): one record per QID, with the properties the
    # verifiers check. fetch() queries Wikidata (SPARQL) for any QIDs not held yet, and saves them.
    name = "wikidata"
    json_columns = ("aliases", "given_names", "family_names", "occupations", "instruments", "has_music_occupation")
    columns = ["id", "label", *json_columns]

    def __init__(self, df=None, id_column=None, path=None, batch_size=100, delay=1.0, retries=4):
        super().__init__(df if df is not None else pd.DataFrame(columns=self.columns), id_column, path)
        self.batch_size, self.delay, self.retries = batch_size, delay, retries
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    @classmethod
    def load(cls, path) -> WikiData:
        # the cache at path, or an empty one that will be saved there
        try:
            return cls.from_csv(path)
        except FileNotFoundError:
            return cls(path=str(path))

    def _fetch_remote(self, ids):
        qids = [i for i in ids if i.startswith("Q")]
        records = []
        for start in range(0, len(qids), self.batch_size):
            if start:
                time.sleep(self.delay)
            records += self._fetch_batch(qids[start:start + self.batch_size])
        return records

    def _fetch_batch(self, qids):
        values = dict(items=" ".join(f"wd:{q}" for q in qids), occ=P_OCCUPATION, instr=P_INSTRUMENT, sub=P_SUBCLASS_OF,
                      music=" ".join(f"wd:{q}" for q in sorted(MUSIC_OCCUPATION_QIDS)), given=P_GIVEN_NAME,
                      ordinal=P_SERIES_ORDINAL, family=P_FAMILY_NAME, pseudonym=P_PSEUDONYM)
        records = {q: {"id": q, "label": "", "aliases": [], "given_names": [], "family_names": [], "occupations": [],
                       "instruments": [], "has_music_occupation": False} for q in qids}

        for row in self._query(_PERSON_QUERY % values):
            r = records[_qid(row["item"])]
            if "occ" in row and _qid(row["occ"]) not in r["occupations"]:
                r["occupations"].append(_qid(row["occ"]))
            if "instr" in row and _qid(row["instr"]) not in r["instruments"]:
                r["instruments"].append(_qid(row["instr"]))
            r["has_music_occupation"] |= row["is_musician"]["value"] == "true"

        given = defaultdict(dict)   # qid -> {name: ordinal}, so names come out in written order
        for row in self._query(_GIVEN_NAMES_QUERY % values):
            given[_qid(row["item"])][row["name"]["value"]] = int(row.get("ordinal", {}).get("value", 99))
        for q, names in given.items():
            records[q]["given_names"] = sorted(names, key=names.get)

        for row in self._query(_FAMILY_NAMES_QUERY % values):
            r = records[_qid(row["item"])]
            if row["name"]["value"] not in r["family_names"]:
                r["family_names"].append(row["name"]["value"])

        for row in self._query(_NAMES_QUERY % values):
            r = records[_qid(row["item"])]
            name = row["name"]["value"]
            if row["kind"]["value"] == "label":
                r["label"] = name
            if name not in r["aliases"]:
                r["aliases"].append(name)   # the label counts as an alias too
        return list(records.values())

    def _query(self, query):
        for attempt in range(self.retries):
            try:
                resp = self.session.get(SPARQL_ENDPOINT, params={"query": query, "format": "json"}, timeout=90)
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 30 * (attempt + 1)))
                    logger.warning(f"SPARQL rate-limited; waiting {wait}s")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()["results"]["bindings"]
            except requests.RequestException as e:
                if attempt == self.retries - 1:
                    raise ServiceUnavailable(f"Wikidata SPARQL: {describe(e)} ({self.retries} attempts)") from e
                logger.warning(f"SPARQL request failed (attempt {attempt + 1}/{self.retries}): {describe(e)}")
                time.sleep(10 * (attempt + 1))
        raise ServiceUnavailable(f"Wikidata SPARQL: still rate-limited ({self.retries} attempts)")


def _qid(binding):
    return binding["value"].rsplit("/", 1)[-1]


def _entity(id, records):
    r = records[0]
    return Entity(id, r["label"], {k: v for k, v in r.items() if k not in ("id", "label")})


WikiData.views = {
    # entities: one per QID, with its names and occupations as properties
    "entities": lambda ds: View(ds, key=lambda r: r["id"], to_entity=_entity),
}
