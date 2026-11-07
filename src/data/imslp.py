from __future__ import annotations

import ast
from urllib.parse import unquote

import pandas as pd

from reconciliation.reconciliation_service import Entity
from utils.names import split_sort_name

from .dataset import Dataset, View

# Per-work descriptive fields joined from datasets/imslp/works_metadata.csv (as in the original pipeline).
METADATA_COLUMNS = [
    "info_Year/Date of Composition",
    "info_Instrumentation",
    "info_Key",
    "info_Piece Style",
    "info_Movements/Sections",
    "info_Average Duration",
]



def page_title(url: str) -> str:
    # an IMSLP work URL -> its page title, i.e. the work's id, e.g. ".../wiki/Scherzo_capriccioso,_Op.66_(Dvo%C5%99%C3%A1k,
    # _Anton%C3%ADn)" -> "Scherzo capriccioso, Op.66 (Dvořák, Antonín)"
    return unquote(url.strip().split("/wiki/", 1)[-1]).replace("_", " ").strip()


def for_instrumentation(*instrumentations: str):
    # for IMSLPWorks.filter (needs the metadata joined): works for one of these instrumentations, as IMSLP's categories
    # name them ("For <instrumentation>"), or arranged for one ("For <instrumentation> (arr)"). e.g.
    #   solo piano: for_instrumentation("piano")
    #   violin (unaccompanied, or with piano, as in exams): for_instrumentation("violin", "violin, piano")
    wanted = {f"For {i}{arr}" for i in instrumentations for arr in ("", " (arr)")}
    return lambda df: df["categories"].map(lambda c: bool(wanted & set(c)))


def _composer_key(name: str) -> str | None:
    # IMSLP composer names are category names ("Beethoven, Ludwig van"), shared by works and the composer list
    return name.strip() or None


class IMSLPWorks(Dataset):
    name = "imslp"
    # One record per IMSLP work page; id = the page title, e.g. "Piano Sonata No.14, Op.27 No.2 (Beethoven, Ludwig van)"
    # (pageid isn't usable as an id: ~2,000 works have none and some share one)

    @classmethod
    def from_dumps(cls, works_csv: str, metadata_csv: str | None = None) -> IMSLPWorks:
        # clean datasets/imslp/works.csv (API export): drop junk rows, unpack intvals, optionally join METADATA_COLUMNS
        # and the categories (a list) by permlink
        raw = pd.read_csv(works_csv, dtype=str, keep_default_na=False)
        raw = raw[raw["id"] != ""]   # rows with no page title are CSV-export junk
        intvals = raw["intvals"].map(lambda s: ast.literal_eval(s) if s else {})
        df = pd.DataFrame({
            "id": raw["id"],
            "composer": intvals.map(lambda d: d.get("composer", "")),
            "worktitle": intvals.map(lambda d: d.get("worktitle", "")),
            "icatno": intvals.map(lambda d: d.get("icatno", "")),
            "pageid": intvals.map(lambda d: d.get("pageid", "")),
            "permlink": raw["permlink"],
        })
        if metadata_csv:
            meta = pd.read_csv(metadata_csv, dtype=str, keep_default_na=False,
                               usecols=["permlink", "status", "categories", *METADATA_COLUMNS])
            meta = meta[meta["status"] == "ok"].drop(columns="status").drop_duplicates("permlink")
            df = df.merge(meta, on="permlink", how="left").fillna("")
            df["categories"] = df["categories"].map(lambda c: c.split("|") if c else [])
        return cls(df.reset_index(drop=True))


class IMSLPComposers(Dataset):
    name = "imslp"
    # One record per IMSLP composer category; id = the category, e.g. "Category:Beethoven, Ludwig van"

    @classmethod
    def from_dumps(cls, composers_csv: str, works_csv: str | None = None, credited_only: bool = True) -> IMSLPComposers:
        # clean datasets/imslp/composers.csv (API export): keep composer categories, add the bare name
        # credited_only: keep only composers credited on at least one work in works_csv
        raw = pd.read_csv(composers_csv, dtype=str, keep_default_na=False)
        raw = raw[raw["id"].str.startswith("Category:")]   # other rows are CSV-export junk
        df = pd.DataFrame({
            "id": raw["id"],
            "name": raw["id"].str.removeprefix("Category:").str.strip(),
            "permlink": raw["permlink"],
        })
        if credited_only:
            if works_csv is None:
                raise ValueError("credited_only needs works_csv")
            df = df[df["name"].isin(set(IMSLPWorks.from_dumps(works_csv).df["composer"].str.strip()))]
        return cls(df.reset_index(drop=True))


def _work_entity(id: str, records: list[dict]) -> Entity:
    # the text, for prompts and embedding alike, is the title alone (which usually holds its Op./BWV/... number):
    # IMSLP's own catalogue no. (e.g. "IJB 946") reads to the LLM as a conflicting number and adds noise to the
    # embedding, and the composer is the same for every candidate
    r = records[0]
    properties = {k: v for k, v in r.items() if k not in ("id", "worktitle", "categories") and v}
    composer = _composer_key(r["composer"])
    return Entity(id, r["worktitle"].strip(), properties | {"composers": [composer] if composer else [],
                                                            "query_text": _query_text(r)})


def _query_text(r: dict, max_sections: int = 500) -> str:
    # how the work reads as a query (side A of the pairwise prompt, prompts/pairwise/query_imslp.txt): labelled fields,
    # like a graded piece's; the sections list is cut at max_sections characters (some works list 40+ pieces)
    sections = r.get("info_Movements/Sections", "").strip()
    if len(sections) > max_sections:
        sections = sections[:max_sections].rsplit(" ", 1)[0] + " …"
    fields = [("piece_title", r["worktitle"]), ("catalogue", r["icatno"]), ("key", r.get("info_Key", "")),
              ("sections", sections), ("year", r.get("info_Year/Date of Composition", ""))]
    return "\n".join(f"{name}: {value.strip()}" for name, value in fields if value.strip())


def _composer_entity(id: str, records: list[dict]) -> Entity:
    # IMSLP composer names are "Family, Given", e.g. "Bach, Johann Sebastian"
    return Entity(id, id, split_sort_name(id))


IMSLPWorks.views = {
    # works: one entity per work page
    "works": lambda ds: View(ds, key=lambda r: r["id"], to_entity=_work_entity,
                             references={"composers": "imslp.composers"}),
    # composers: one entity per composer name credited on a work
    "composers": lambda ds: View(ds, key=lambda r: _composer_key(r["composer"]), to_entity=_composer_entity),
}

IMSLPComposers.views = {
    # composers: one entity per composer category, with the same ids as IMSLPWorks' composers view
    "composers": lambda ds: View(ds, key=lambda r: _composer_key(r["name"]), to_entity=_composer_entity),
}
