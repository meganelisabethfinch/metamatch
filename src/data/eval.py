from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import cache

import pandas as pd

from graph.relation import Relation
from utils.paths import PATHS

from .imslp import IMSLPWorks, page_title

logger = logging.getLogger(__name__)

# The labels in the eval CSVs, as the relation (query -> labelled work) each means
EVAL_LABELS = {"exact": Relation.IS, "part_of": Relation.IS_PART_OF, "contains": Relation.CONTAINS,
               "no_match": Relation.IS_NOT}

_IMSLP_URL = re.compile(r"https?://imslp\.org/wiki/[^\s\]]+")


@dataclass(frozen=True)
class Truth:
    """One hand-labelled query: the source rows it covers, and how it relates to which target works."""
    key: str                           # the eval CSV's id for the query, e.g. a graded piece's piece_id
    source_ids: tuple[str, ...]        # the source view's ids for it (a piece listed several times has several rows)
    relation: Relation                 # its label: the relation to its best match, IS_NOT if it has none
    targets: dict[str, Relation]       # target work id -> relation (empty for an unlabelled no_match)


def graded_pieces_truth(target: str, path=None) -> list[Truth]:
    # datasets/eval/graded_pieces_eval.csv (paths.eval.graded_pieces): ~450 graded pieces, each labelled with its best
    # match in MusicBrainz (mb_match_type / mb_ground_truth: MBIDs) and in IMSLP (imslp_*: work URLs)
    # target: "musicbrainz" or "imslp"; rows with no label for that target are skipped
    columns = {"musicbrainz": ("mb_match_type", "mb_ground_truth", _mbids),
               "imslp": ("imslp_match_type", "imslp_ground_truth", _imslp_ids)}
    type_column, id_column, parse = columns[target]
    df = pd.read_csv(path or PATHS["eval"]["graded_pieces"], dtype=str, keep_default_na=False)
    works = pd.read_csv(PATHS["graded_pieces"]["works"], dtype=str, keep_default_na=False, usecols=["id", "piece_id"])
    rows = works.groupby("piece_id")["id"].apply(tuple).to_dict()
    truths = []
    for r in df.to_dict("records"):
        labels = [t.strip() for t in r[type_column].split("|") if t.strip()]
        if not labels:
            continue
        if unknown := [t for t in labels if t not in EVAL_LABELS]:
            logger.warning(f"{r['piece_id']}: unknown {type_column} {unknown}; skipped")
            continue
        relations = [EVAL_LABELS[t] for t in labels]
        ids = parse(r[id_column])
        # several labels for several works ("exact|contains"): paired in order; otherwise the first label is the row's
        pairs = zip(ids, relations) if len(relations) == len(ids) > 1 else ((i, relations[0]) for i in ids)
        truths.append(Truth(r["piece_id"], rows[r["piece_id"]], relations[0], dict(pairs)))
    return truths


# The ground truth for a source view: source view name -> function(target dataset name) -> list[Truth]
GROUND_TRUTH = {"graded_pieces.works": graded_pieces_truth}


def _mbids(text: str) -> list[str]:
    # a bare MBID, a list "[a, b]", or "a|b"
    return [t for t in re.split(r"[\s,|\[\]]+", text) if t]


def _imslp_ids(text: str) -> list[str]:
    # IMSLP work URLs (a bare URL, or a list "[url, url]") -> work ids; a note like "this is no. 2 from <url> - NOT
    # <url>" keeps only the URLs before "NOT"
    text = re.split(r"\bNOT\b", text)[0]
    return [_known_imslp_id(page_title(u.rstrip(","))) for u in _IMSLP_URL.findall(text)]


def _known_imslp_id(id: str) -> str:
    # the id of the work in our IMSLP dump: pages renamed since (usually to add a catalogue number, e.g. "2 Arabesques,
    # CD 74 (Debussy, Claude)" was "2 Arabesques (Debussy, Claude)") are tried without the title's last ", ..." part
    ids = _imslp_work_ids()
    if id not in ids:
        title, _, composer = id.rpartition(" (")
        shorter = f"{title.rpartition(', ')[0]} ({composer}"
        if ", " in title and shorter in ids:
            return shorter
        logger.warning(f"IMSLP work {id!r} isn't in the IMSLP dump")
    return id


@cache
def _imslp_work_ids() -> frozenset[str]:
    return frozenset(IMSLPWorks.from_dumps(PATHS["imslp"]["works"]).df["id"])
