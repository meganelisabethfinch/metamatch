# Preprocesses the clean datasets: instruments and grades (graded pieces), tables for Pedro Ramoneda's datasets, and
# LLM name splitting (composer_parts etc.). A source already preprocessed is skipped unless --rerun.
# e.g. python src/preprocess.py graded_pieces cipi
#      python src/preprocess.py --all --rerun

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from constants.grades import GRADE_LABELS
from data.dataset import Dataset
from data.graded_pieces import GradedPieces
from utils.llm.name_split_parser import NameParts
from utils.instruments import canonical_instruments
from utils.llm.model import is_remote, llm_names, load_llm
from utils.llm.name_splitter import NameSplitter
from utils.paths import PATHS

logger = logging.getLogger(__name__)

SOURCES = ["graded_pieces", "mikrokosmos", "pianostreet", "cipi", "freescores", "psyllabus"]

# The local LLM that splits names when none is named: a key of paths.yaml models.llm (fits the GPU in bf16)
DEFAULT_SPLIT_LLM = "qwen35-4b"


def preprocess(sources, model=DEFAULT_SPLIT_LLM, resume=True, plot=False) -> None:
    # resume: skip sources already preprocessed (done()); otherwise redo them. model: the name-splitting LLM, loaded
    # only if a name isn't in the name-parts cache
    skipped = [s for s in sources if resume and done(s)]
    todo = [s for s in sources if s not in skipped]
    logger.info(f"preprocessing {todo or 'nothing'}" + (f" (already done: {skipped})" if skipped else ""))
    template = (Path(__file__).parent / "prompts" / "name_split_prompt.txt").read_text()
    path = PATHS["models"]["llm"][model]
    name_splitter = NameSplitter(lambda *a, **k: load_llm(path)(*a, **k), template, PATHS["name_parts_cache"])
    for name in todo:
        if name == "graded_pieces":
            preprocess_graded_pieces(PATHS["graded_pieces"]["works"], name_splitter)
        elif name == "mikrokosmos":
            preprocess_mikrokosmos(PATHS["mikrokosmos"]["raw"], PATHS["mikrokosmos"]["works"])
        else:
            PREPROCESS[name](PATHS[name]["raw"], PATHS[name]["works"], name_splitter)
    if plot:
        from stats.datasets import plot_dataset
        for name in sources:
            plot_dataset(name)


def done(source) -> bool:
    # graded pieces are preprocessed in place: done once they have the instruments column it adds; the others once
    # their clean table exists
    if source == "graded_pieces":
        return "instruments" in pd.read_csv(PATHS["graded_pieces"]["works"], nrows=0).columns
    return Path(PATHS[source]["works"]).exists()


def preprocess_graded_pieces(csv, name_splitter=None):
    # instruments always; names only with a name_splitter (otherwise composer_parts / arranger_parts are kept as they are)
    graded_pieces = GradedPieces.from_csv(csv)
    graded_pieces.df = split_instruments(graded_pieces.df)
    graded_pieces.df = collapse_grades(graded_pieces.df, GRADE_LABELS)
    if name_splitter is not None:
        graded_pieces.df = split_names(graded_pieces.df, ["composer", "arranger"], name_splitter)
    graded_pieces.to_csv()


def split_instruments(df):
    # adds instruments / instrument_families: one per instrument the entry is for, e.g. "alto/baritone saxophone" ->
    # ["saxophone, alto", "saxophone, baritone"], ["woodwind", "woodwind"]; instrument keeps the value as written
    parsed = df["instrument"].map(canonical_instruments)
    df = df.copy()
    df["instruments"] = parsed.map(lambda ps: [name for _, name in ps])
    df["instrument_families"] = parsed.map(lambda ps: [family for family, _ in ps])
    return df


def collapse_grades(df, grade_labels):
    # labels matching one of a grade's templates are written as that grade, e.g. RCM "ARCT in Flute Performance" -> "ARCT"
    df = df.copy()
    for board, grades in grade_labels.items():
        rows = df["exam_board"] == board
        for grade, templates in grades.items():
            pattern = "|".join(re.escape(t).replace(re.escape("{x}"), ".*?") for t in templates)
            df.loc[rows & df["grade"].str.fullmatch(pattern, na=False), "grade"] = grade
    return df


# Pedro Ramoneda's difficulty datasets: already fairly clean, so each just needs loading into a table (keeping its own
# columns - they don't share a format) with a stable id, plus composer name splitting.
# Raw and clean paths are in config/paths.yaml.

def preprocess_mikrokosmos(metadata_csv, out_csv):
    # composer is always Bartók (written "Béla Bártok"), so no LLM needed
    df = pd.read_csv(metadata_csv, dtype=str, keep_default_na=False)
    df.insert(0, "id", df["piece number"])
    bartok = NameParts("person", substring="Béla Bártok", family_name="Bartók", given_names=["Béla"])
    df["composer_parts"] = [[asdict(bartok)] for _ in range(len(df))]
    _save(df, out_csv)


def preprocess_pianostreet(index_json, out_csv, name_splitter):
    df = _load_json_index(index_json)
    df["work"] = df["work"].str.strip()
    _save(split_names(df, ["composer"], name_splitter), out_csv)


def preprocess_cipi(index_json, out_csv, name_splitter):
    # composer is upper-case (e.g. "LUDWIG VAN BEETHOVEN")
    df = _load_json_index(index_json)
    _save(split_names(df, ["composer"], name_splitter), out_csv)


def preprocess_freescores(index_json, out_csv, name_splitter):
    # composer and title come combined in the first line of piece_text ("Composer : Title")
    df = _load_json_index(index_json)
    first_line = df["piece_text"].str.split("\n").str[0]
    has_composer = first_line.str.contains(" : ", regex=False)
    df["composer"] = first_line.where(has_composer, "").str.split(" : ", n=1).str[0].str.strip()
    df["work_name"] = first_line.where(~has_composer, first_line.str.split(" : ", n=1).str[1]).str.strip()
    _save(split_names(df, ["composer"], name_splitter), out_csv)


def preprocess_psyllabus(index_json, out_csv, name_splitter):
    df = _load_json_index(index_json)
    _save(split_names(df, ["composer"], name_splitter), out_csv)


def split_names(df, columns, name_splitter):
    # adds <column>_parts; names the splitter couldn't parse get None (an empty cell), not [] ("no names")
    parts = name_splitter.split(sorted({n for c in columns for n in df[c]}))
    df = df.copy()
    for c in columns:
        df[f"{c}_parts"] = df[c].map(lambda n: [asdict(p) for p in parts[n]] if n in parts else None)
    return df


def _load_json_index(path):
    # {key: {field: value}} -> one row per key, id = the key
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame.from_dict(data, orient="index").fillna("")
    df.insert(0, "id", df.index)
    return df.reset_index(drop=True)


def _save(df, out_csv):
    dataset = Dataset(df)
    dataset.json_columns = tuple(c for c in df.columns if df[c].map(lambda v: isinstance(v, (list, dict))).any())
    dataset.to_csv(out_csv)


PREPROCESS = {"pianostreet": preprocess_pianostreet, "cipi": preprocess_cipi, "freescores": preprocess_freescores,
              "psyllabus": preprocess_psyllabus}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Preprocess the clean datasets (instruments, grades, LLM name splitting).")
    ap.add_argument("sources", nargs="*", help=f"sources, run in the order given: {', '.join(SOURCES)}")
    ap.add_argument("--all", action="store_true", help="every source, in the order above")
    ap.add_argument("--rerun", action="store_true", help="redo sources already preprocessed instead of skipping them")
    local = [m for m in llm_names() if not is_remote(m)]
    ap.add_argument("--model", choices=local, default=DEFAULT_SPLIT_LLM, metavar="MODEL",
                    help=f"local LLM for name splitting: {', '.join(local)}. default: {DEFAULT_SPLIT_LLM}")
    ap.add_argument("--plot", action="store_true", help="then draw the datasets' plots (as plot.py datasets)")
    args = ap.parse_args()
    if not args.sources and not args.all:
        ap.error(f"name sources to preprocess, or --all. Sources: {', '.join(SOURCES)}")
    if unknown := [s for s in args.sources if s not in SOURCES]:
        ap.error(f"unknown sources {unknown}. Sources: {', '.join(SOURCES)}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    preprocess(SOURCES if args.all else args.sources, args.model, resume=not args.rerun, plot=args.plot)
