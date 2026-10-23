# plots of the cleaned datasets: counts by instrument, exam board, grade / difficulty, composer, and their cross sections

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from data.graded_pieces import GradedPieces
from utils.paths import PATHS

from . import plots

logger = logging.getLogger(__name__)

FAMILY_COLOURS = {
    "piano": "#4e79a7",
    "keyboard": "#4e79a7",
    "strings": "#f28e2b",
    "woodwind": "#59a14f",
    "brass": "#e15759",
    "guitar": "#76b7b2",
    "voice": "#edc948",
    "percussion": "#b07aa1",
}

# datasets with one difficulty label per work: (path, difficulty column, key in difficulty.json, top composers?)
DIFFICULTY_DATASETS = {
    "cipi": (PATHS["cipi"]["works"], "henle", "cipi", True),
    "pianostreet": (PATHS["pianostreet"]["works"], "difficulty", "pianostreet", True),
    "freescores": (PATHS["freescores"]["works"], "difficulty", "freescores", True),
    "psyllabus": (PATHS["psyllabus"]["works"], "ps_rating", "psyllabus", True),
    "mikrokosmos": (PATHS["mikrokosmos"]["works"], "henle_difficulty", "mikrokosmos", False),   # all Bartók
    "imslp_difficulty": (PATHS["imslp"]["difficulty"], "difficulty_level", "imslp_difficulty", True),
}
DATASETS = ["graded_pieces", *DIFFICULTY_DATASETS]
DEFAULT_SCALES = ("count",)   # which of plots.SCALES to draw each chart at


def difficulty_orders() -> dict:
    return json.loads(Path(PATHS["eval"]["difficulty"]).read_text())


def plot_dataset(name: str, out_dir=None, fmt: str = "pdf", scales=DEFAULT_SCALES) -> None:
    out_dir = Path(out_dir or PATHS["plots"]) / "datasets" / name
    logger.info(f"dataset plots for {name} -> {out_dir}")
    if name == "graded_pieces":
        df = GradedPieces.from_csv(PATHS["graded_pieces"]["works"]).df
        plot_graded_pieces(df, out_dir, fmt=fmt, scales=scales)
    else:
        path, column, order_key, composers = DIFFICULTY_DATASETS[name]
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        plot_difficulty(df, name, column, difficulty_orders().get(order_key), out_dir, composers, fmt=fmt,
                        scales=scales)


def plot_graded_pieces(df: pd.DataFrame, out_dir: Path, fmt: str = "pdf", scales=DEFAULT_SCALES) -> None:
    orders = difficulty_orders()
    boards = sorted(df["exam_board"].unique())
    # all boards' grades on one axis: the pooled order, then each board's own order for the labels it doesn't have
    grade_order = list(dict.fromkeys([*orders["graded_pieces"], *(g for b in boards for g in orders.get(b, []))]))
    per_instrument = explode_instruments(df)
    families = dict(zip(per_instrument["instruments"], per_instrument["instrument_families"]))

    def by(column, frame=df):
        return frame[column][frame[column] != ""].value_counts().to_dict()

    plots.save_scales(lambda s: plots.bar_chart(
        by("instruments", per_instrument), "Number of entries per instrument", "Instrument", groups=families,
        group_colours=FAMILY_COLOURS, group_label="Family", scale=s),
        out_dir / "entries_per_instrument", scales=scales, fmt=fmt)
    plots.save_scales(lambda s: plots.bar_chart(
        by("exam_board"), "Number of entries per exam board", "Exam board", scale=s),
        out_dir / "entries_per_exam_board", scales=scales, fmt=fmt)
    plots.save_scales(lambda s: plots.bar_chart(
        by("grade"), "Number of entries per grade", "Grade", order=grade_order, scale=s),
        out_dir / "entries_per_grade", scales=scales, fmt=fmt)

    plots.save(plots.heatmap(
        cross(df, "grade", "exam_board"), "Number of entries: Grade × Exam board", row_order=grade_order),
        out_dir / "entries_by_grade_x_exam_board", fmt=fmt)
    plots.save(plots.heatmap(
        cross(per_instrument, "instruments", "exam_board"), "Number of entries: Instrument × Exam board",
        row_groups=families), out_dir / "entries_by_instrument_x_exam_board", fmt=fmt)
    plots.save(plots.heatmap(
        cross(per_instrument, "grade", "instruments"), "Number of entries: Grade × Instrument",
        row_order=grade_order, col_groups=families), out_dir / "entries_by_grade_x_instrument", fmt=fmt)
    for board in boards:
        rows = per_instrument[per_instrument["exam_board"] == board]
        plots.save(plots.heatmap(
            cross(rows, "grade", "instruments"), f"Number of entries: Grade × Instrument ({board})",
            row_order=orders.get(board), col_groups=families),
            out_dir / "entries_by_grade_x_instrument" / board, fmt=fmt)


def plot_difficulty(df: pd.DataFrame, name: str, column: str, order, out_dir: Path, composers: bool = True,
                    top_n: int = 20, fmt: str = "pdf", scales=DEFAULT_SCALES) -> None:
    counts = df[column][df[column].str.strip() != ""].str.strip().value_counts().to_dict()
    plots.save_scales(lambda s: plots.bar_chart(
        counts, f"Number of entries per difficulty: {name}", column, order=order, scale=s),
        out_dir / "entries_per_difficulty", scales=scales, fmt=fmt)
    if composers and "composer" in df.columns:
        top = df["composer"][df["composer"].str.strip() != ""].str.strip().value_counts().head(top_n)
        plots.save_scales(lambda s: plots.bar_chart(
            top.to_dict(), f"Number of entries per composer: {name} (top {top_n})", "Composer", order=list(top.index),
            scale=s), out_dir / "entries_per_composer", scales=scales, fmt=fmt)


def explode_instruments(df: pd.DataFrame) -> pd.DataFrame:
    # one row per (entry, instrument): an entry for several instruments counts once for each
    df = df.assign(pairs=[list(zip(i or [], f or [])) for i, f in zip(df["instruments"], df["instrument_families"])])
    df = df.explode("pairs").dropna(subset=["pairs"])
    return df.assign(instruments=df["pairs"].str[0], instrument_families=df["pairs"].str[1]).drop(columns="pairs")


def cross(df: pd.DataFrame, rows: str, columns: str) -> pd.DataFrame:
    df = df[(df[rows] != "") & (df[columns] != "")]
    return df.groupby([rows, columns]).size().unstack(fill_value=0)
