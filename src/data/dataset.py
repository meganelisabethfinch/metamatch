from __future__ import annotations

import json
import os
from typing import Callable

import pandas as pd

from reconciliation.reconciliation_service import Entity


class Dataset():
    name: str = ""                     # e.g. "graded_pieces"; views are named "<name>.<view>"
    id_column: str = "id"
    json_columns: tuple[str, ...] = ()
    views: dict[str, Callable[[Dataset], View]] = {}

    def __init__(self, df: pd.DataFrame, id_column: str | None = None, path: str | None = None):
        # df: the records; id_column: the column holding each record's unique id
        # path: the .csv this dataset is saved to (needed by cached datasets, which save after each fetch)
        self.id_column = id_column or self.id_column
        if self.id_column not in df.columns:
            raise ValueError(f"no id column {self.id_column!r} in {list(df.columns)}")
        df = df.astype({self.id_column: str})
        duplicates = df[self.id_column][df[self.id_column].duplicated()].unique()
        if len(duplicates):
            raise ValueError(f"duplicate ids: {list(duplicates[:5])}")
        self.df = df.set_index(self.id_column, drop=False)
        self.path = path

    @classmethod
    def from_csv(cls, path: str, **kwargs) -> Dataset:
        # read records from a .csv file; every column is read as a string, empty cells as ""
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        for column in cls.json_columns:
            if column in df.columns:
                df[column] = df[column].map(lambda v: json.loads(v) if v else None)
        return cls(df, path=path, **kwargs)

    def to_csv(self, path: str | None = None) -> None:
        # write records to a .csv file (defaults to the one this dataset was loaded from)
        path = str(path or self.path)
        df = self.df.copy()
        for column in self.json_columns:
            if column in df.columns:
                df[column] = df[column].map(lambda v: json.dumps(v, ensure_ascii=False) if v is not None else "")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
        self.path = path

    def fetch(self, ids: list[str]) -> dict[str, dict]:
        # the records for ids, getting any we don't hold yet from the remote source (in one batch) and saving them
        # complete datasets have no remote source, so this just returns the records they hold
        missing = [i for i in dict.fromkeys(ids) if i not in self.df.index]
        if missing:
            new = self._fetch_remote(missing)
            if new:
                self.df = pd.concat([self.df, pd.DataFrame(new).set_index(self.id_column, drop=False)])
                if self.path:
                    self.to_csv()
        return {i: self.get(i) for i in ids if i in self.df.index}

    def _fetch_remote(self, ids: list[str]) -> list[dict]:
        # cached datasets override this to query their source; one record (with an id_column value) per id found
        return []

    def ids(self) -> list[str]:
        # all record ids
        return list(self.df.index)

    def get(self, id: str) -> dict:
        # direct access to one record (a dict of column -> value) by id
        return self.df.loc[id].to_dict()

    def filter(self, condition: Callable[[pd.DataFrame], pd.Series]) -> Dataset:
        # a new Dataset with only the records matching condition, e.g. gp.filter(lambda df: df.instrument == "piano")
        return type(self)(self.df[condition(self.df)].reset_index(drop=True), self.id_column)

    def view(self, name: str, **options):
        # one of this dataset's views by name, e.g. GradedPieces.view("works", primary_only=True)
        # subclasses register their views, e.g. {"composers": ..., "works": ...}; options go to the view's function
        if name not in self.views:
            raise KeyError(f"{type(self).__name__} has no view {name!r}; views: {list(self.views)}")
        view = self.views[name](self, **options)
        view.name = f"{self.name}.{name}"
        return view

    def __len__(self) -> int:
        return len(self.df)


class View():
    # e.g. GP.composers, GP.works

    def __init__(
        self,
        dataset: Dataset,
        key: Callable[[dict], str | list[str] | None],
        to_entity: Callable[[str, list[dict]], Entity],
        references: dict[str, str] | None = None,
        same_as: dict[str, str] | None = None,
    ):
        # dataset: the Dataset this view is over
        # key: the id(s) of the entity / entities a record belongs to, None or [] for none
        #   e.g. composers: normalised composer name; works: the record id (one entity per record)
        # to_entity: builds one Entity from its id and its records, i.e. decides its text and properties
        #   e.g. composers: text = composer name; works: text = labelled descriptor (title / movement / work / book)
        # references: properties whose values are ids in another view, e.g. works {"composers": "musicbrainz.artists"}
        self.dataset = dataset
        # same_as: properties whose value is the id of the SAME entity in another view, e.g. MusicBrainz artists
        #   {"wikidata_qid": "wikidata.entities"}; build_graph turns them into IS edges
        self.references = references or {}
        self.same_as = same_as or {}
        self._reverse: dict[str, dict[str, list[str]]] = {}   # property -> referenced id -> ids of entities here
        self.key = key
        self.to_entity = to_entity
        self._index: dict[str, list[str]] | None = None   # entity id -> record ids
        self._indexed_len = -1
        self.name = ""   # set by Dataset.view, e.g. "graded_pieces.composer_credits"

    def __str__(self) -> str:
        return self.name

    def entities(self) -> list[Entity]:
        # all entities in this view
        # for a cached dataset, only those fetched so far (e.g. the WD composers our queries have been matched to)
        return [self._entity(i) for i in self._get_index()]

    def get(self, ids: list[str]) -> list[Entity]:
        # the entities for ids (in order, skipping unknown ids); for a cached dataset, any it lacks are fetched first
        # (fetching assumes entity ids are record ids, e.g. WD.composers: one entity per QID)
        # e.g. VerifierService gets its target properties this way: OpenRefine returns QIDs -> WD.composers.get(qids)
        missing = [i for i in ids if i not in self._get_index()]
        if missing:
            self.dataset.fetch(missing)
        index = self._get_index()
        return [self._entity(i) for i in ids if i in index]

    def referencing(self, property: str, ids) -> dict[str, list[str]]:
        # for each id in ids, the ids of this view's entities whose `property` references it (e.g. a composer's works);
        # from the dataset if it can look this up itself (e.g. an SQL index), else from all this view's entities
        ids = list(ids)
        lookup = getattr(self.dataset, "referencing", None)
        if lookup is not None:   # asked once per id, then remembered
            known = self._reverse.setdefault(property, {})
            if missing := [i for i in ids if i not in known]:
                known.update(lookup(property, missing))
            return {i: known[i] for i in ids}
        if property not in self._reverse:
            reverse = {}
            for e in self.entities():
                for ref in e.properties.get(property) or []:
                    reverse.setdefault(ref, []).append(e.id)
            self._reverse[property] = reverse
        return {i: self._reverse[property].get(i, []) for i in ids}

    def records(self, entity_id: str) -> list[str]:
        # the ids of the dataset records this entity came from (e.g. every piece by this composer)
        return list(self._get_index().get(entity_id, []))

    def _entity(self, entity_id: str) -> Entity:
        return self.to_entity(entity_id, [self.dataset.get(r) for r in self._get_index()[entity_id]])

    def _get_index(self) -> dict[str, list[str]]:
        # (re)built when the dataset has grown, e.g. after a fetch
        if self._index is None or self._indexed_len != len(self.dataset):
            index: dict[str, list[str]] = {}
            for record in self.dataset.df.to_dict("records"):
                keys = self.key(record)
                for k in [keys] if isinstance(keys, str) else (keys or []):
                    index.setdefault(k, []).append(record[self.dataset.id_column])
            self._index, self._indexed_len = index, len(self.dataset)
        return self._index
