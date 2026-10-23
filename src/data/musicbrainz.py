from __future__ import annotations

import json
import re
import sqlite3

import pandas as pd
from rapidfuzz import fuzz

from constants.name_particles import NAME_PARTICLES, PARTICLES
from reconciliation.reconciliation_service import Entity
from utils.catalogue import with_catalogue_numbers
from utils.names import norm_name, split_sort_name

from .dataset import Dataset, View

# MusicBrainz is read from local SQLite indexes (built from the MusicBrainz JSON dumps): artists in
# config paths.musicbrainz.artist_index, works + work -> composer in paths.musicbrainz.works_index.
# Like a cached dataset, records are read from the index as they're needed, rather than all loaded up front.

_FTS_STOPWORDS = NAME_PARTICLES | PARTICLES | {"and", "the"}
_FTS_SPECIAL = re.compile(r"[^\w\sÀ-ž-]")


class MusicBrainzArtists(Dataset):
    # One record per MusicBrainz artist (person, group, ...); id = MBID
    name = "musicbrainz"
    json_columns = ("aliases", "members")
    columns = ["id", "name", "sort_name", "type", "disambiguation", "aliases", "members", "wikidata_qid", "has_works"]

    def __init__(self, artist_index, works_index=None, df=None, id_column=None, path=None):
        # works_index: used for has_works (whether the artist, or a group's members, composed any work)
        super().__init__(df if df is not None else pd.DataFrame(columns=self.columns), id_column, path)
        self.artist_index, self.works_index = str(artist_index), works_index and str(works_index)

    def search(self, text: str, limit: int = 50) -> list[tuple[str, float]]:
        # (mbid, score) for the artists whose names best match text, best first; score: 0-100 name similarity
        query, fallback = _fts_queries(text)
        if query is None:
            return []
        with _connect(self.artist_index) as con:
            rows = _fts_rows(con, query, limit * 2) or (_fts_rows(con, fallback, limit * 2) if fallback != query else [])
        self.fetch([r["mbid"] for r in rows])
        scored = []
        for r in rows:
            names = [r["name"], r["sort_name"], *json.loads(r["aliases_json"] or "[]")]
            score = max(fuzz.token_sort_ratio(norm_name(text), norm_name(n)) for n in names if n)
            scored.append((r["mbid"], score, "," in r["sort_name"]))
        scored.sort(key=lambda s: (-s[1], -s[2]))   # ties: prefer a proper "Family, Given" sort name
        return [(mbid, score) for mbid, score, _ in scored[:limit]]

    def referencing(self, property: str, ids) -> dict[str, list[str]]:
        # for View.referencing: the artists whose wikidata_qid is each of ids
        if property != "wikidata_qid":
            raise KeyError(f"musicbrainz.artists has no reference {property!r}")
        with _connect(self.artist_index) as con:
            rows = _select(con, "SELECT mbid, wikidata_qid FROM artists WHERE wikidata_qid IN ({})", ids)
        artists = {i: [] for i in ids}
        for r in rows:
            artists[r["wikidata_qid"]].append(r["mbid"])
        return artists

    def _fetch_remote(self, ids):
        with _connect(self.artist_index) as con:
            rows = _select(con, "SELECT * FROM artists WHERE mbid IN ({})", ids)
        records = [{"id": r["mbid"], "name": r["name"], "sort_name": r["sort_name"], "type": r["type"] or "",
                    "disambiguation": r["disambiguation"] or "", "aliases": json.loads(r["aliases_json"] or "[]"),
                    "members": json.loads(r["members_json"] or "[]"), "wikidata_qid": r["wikidata_qid"] or ""}
                   for r in rows]
        if self.works_index:
            composers = set()
            with _connect(self.works_index) as con:
                artist_ids = {m for r in records for m in [r["id"], *r["members"]]}
                composers = {r["composer_mbid"] for r in _select(
                    con, "SELECT DISTINCT composer_mbid FROM work_composer WHERE composer_mbid IN ({})", artist_ids)}
            for r in records:
                r["has_works"] = bool({r["id"], *r["members"]} & composers)
        return records


class MusicBrainzWorks(Dataset):
    # One record per MusicBrainz work; id = work MBID
    name = "musicbrainz"
    json_columns = ("aliases", "composers", "part_of", "part_of_titles")
    columns = ["id", "title", "type", "language", "aliases", "composers", "part_of", "part_of_titles"]

    def __init__(self, works_index, df=None, id_column=None, path=None):
        super().__init__(df if df is not None else pd.DataFrame(columns=self.columns), id_column, path)
        self.works_index = str(works_index)

    def referencing(self, property: str, ids) -> dict[str, list[str]]:
        # for View.referencing, from the index: "composers" -> each artist's works (work_composer);
        # "part_of" -> each work's parts (work_parts)
        sql = {"composers": "SELECT work_id, composer_mbid AS ref FROM work_composer WHERE composer_mbid IN ({})",
               "part_of": "SELECT work_id, parent_work_id AS ref FROM work_parts WHERE parent_work_id IN ({})"}
        if property not in sql:
            raise KeyError(f"musicbrainz.works has no reference {property!r}")
        with _connect(self.works_index) as con:
            rows = _select(con, sql[property], ids)
        works = {i: [] for i in ids}
        for r in rows:
            works[r["ref"]].append(r["work_id"])
        return works

    def _fetch_remote(self, ids):
        with _connect(self.works_index) as con:
            works = _select(con, "SELECT * FROM works WHERE work_id IN ({})", ids)
            composers, parents = {}, {}
            for r in _select(con, "SELECT work_id, composer_mbid FROM work_composer WHERE work_id IN ({})", ids):
                composers.setdefault(r["work_id"], []).append(r["composer_mbid"])
            for r in _select(con, "SELECT work_id, parent_work_id FROM work_parts WHERE work_id IN ({})", ids):
                parents.setdefault(r["work_id"], []).append(r["parent_work_id"])
            titles = {r["work_id"]: r["title"] for r in _select(
                con, "SELECT work_id, title FROM works WHERE work_id IN ({})", {p for ps in parents.values() for p in ps})}
        return [{"id": r["work_id"], "title": r["title"], "type": r["type"] or "", "language": r["language"] or "",
                 "aliases": json.loads(r["aliases"] or "[]"), "composers": composers.get(r["work_id"], []),
                 "part_of": parents.get(r["work_id"], []),
                 "part_of_titles": [titles[p] for p in parents.get(r["work_id"], []) if titles.get(p)]}
                for r in works]


def _artist_entity(id, records):
    r = records[0]
    properties = {"type": r["type"], "aliases": [r["name"], r["sort_name"], *r["aliases"]],
                  "has_works": r["has_works"], "wikidata_qid": r["wikidata_qid"]}
    if r["type"] == "Person":   # name parts from the sort name, e.g. "Reinagle, Alexander"
        parts = split_sort_name(r["sort_name"])
        properties |= {"given_names": parts["given_names"], "family_names": [parts["family_name"]]}
    return Entity(id, r["name"], properties)


def _work_descriptor(r):
    # what the pairwise prompts show for a work (side B): title with catalogue numbers and type, its first alias,
    # and the works it is part of
    lines = [with_catalogue_numbers(r["title"]) + (f" [type: {r['type']}]" if r["type"] and r["type"] != "Work" else "")]
    if alias := next((a for a in r["aliases"] if a and a != r["title"]), ""):
        lines.append(f"also known as: {alias}")
    if r["part_of_titles"]:
        lines.append(f"part of: {'; '.join(with_catalogue_numbers(t) for t in r['part_of_titles'])}")
    return "\n".join(lines)


def _work_entity(id, records):
    r = records[0]
    embed_text = " | ".join(dict.fromkeys(t for t in [r["title"], *r["aliases"]] if t))   # title + aliases
    return Entity(id, _work_descriptor(r), {"type": r["type"], "aliases": r["aliases"], "composers": r["composers"],
                                            "part_of": r["part_of"], "embed_text": embed_text})


MusicBrainzArtists.views = {"artists": lambda ds: View(ds, key=lambda r: r["id"], to_entity=_artist_entity,
                                                    same_as={"wikidata_qid": "wikidata.entities"})}
MusicBrainzWorks.views = {"works": lambda ds: View(ds, key=lambda r: r["id"], to_entity=_work_entity,
                                                  references={"composers": "musicbrainz.artists",
                                                              "part_of": "musicbrainz.works"})}


def _connect(path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _select(con, sql, ids, chunk=900):
    # rows for an "IN ({})" query over many ids, in chunks (SQLite limits the number of parameters)
    ids = list(dict.fromkeys(ids))
    rows = []
    for i in range(0, len(ids), chunk):
        part = ids[i:i + chunk]
        rows += con.execute(sql.format(",".join("?" * len(part))), part).fetchall()
    return rows


def _fts_queries(text):
    # (query, fallback) FTS5 expressions for a name: its words (+ initials as prefixes); fallback = its longest word
    words, initials = [], []
    for t in _FTS_SPECIAL.sub(" ", text).split():
        if t.lower() in _FTS_STOPWORDS:
            continue
        if len(t) == 1:
            initials.append(f"{t.lower()}*")
        else:
            words.append(f'"{t}"')
    if not words:
        return None, None
    return " ".join(words + initials), max(words, key=len)


def _fts_rows(con, query, limit):
    try:
        return con.execute("SELECT a.* FROM artist_fts f JOIN artists a ON a.mbid = f.mbid "
                           "WHERE artist_fts MATCH ? ORDER BY f.rank LIMIT ?", (query, limit)).fetchall()
    except sqlite3.OperationalError:
        return []
