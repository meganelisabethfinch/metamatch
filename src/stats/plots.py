# generic chart helpers, independent of any dataset

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")   # no display needed
import matplotlib.patches as mpatches   # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402
import matplotlib.ticker as mticker   # noqa: E402
import pandas as pd   # noqa: E402

logger = logging.getLogger(__name__)

SCALES = ("count", "log", "pct")   # saved as <name>, <name>_log, <name>_pct
_SUFFIX = {"count": "", "log": "_log", "pct": "_pct"}
FORMATS = ("pdf", "png", "jpg")


def ordered(categories, order=None, groups=None) -> list:
    # categories in `order` first (as listed), then the rest sorted - by group, if groups maps category -> group
    categories = list(dict.fromkeys(categories))
    known = [c for c in (order or []) if c in set(categories)]
    rest = [c for c in categories if c not in set(known)]
    rest.sort(key=lambda c: (str(groups.get(c, "")), str(c)) if groups else str(c))
    return known + rest


def palette(names) -> dict:
    # a colour per name, from matplotlib's default cycle
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {n: colours[i % len(colours)] for i, n in enumerate(dict.fromkeys(names))}


def bar_chart(counts: dict, title: str, xlabel: str, ylabel: str = "Number of entries", order=None,
              groups: dict | None = None, group_colours: dict | None = None, group_label: str = "",
              scale: str = "count"):
    # one bar per category, labelled with its count; groups (category -> group, e.g. instrument -> family) colours the
    # bars, orders categories by group and separates groups with dashed lines
    cats = ordered(counts, order, groups)
    values = [counts[c] for c in cats]
    if scale == "pct":
        total = sum(values) or 1
        values = [v / total for v in values]
    colours = None
    if groups:
        group_colours = group_colours or palette(groups[c] for c in cats)
        colours = [group_colours.get(groups.get(c), "#aaaaaa") for c in cats]

    fig, ax = plt.subplots(figsize=(max(8, len(cats) * 0.45 + 2), 6))
    bars = ax.bar(range(len(cats)), values, color=colours)
    for bar, n in zip(bars, (counts[c] for c in cats)):
        if n:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(n), ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=55 if len(cats) > 8 else 0, ha="right" if len(cats) > 8 else "center", fontsize=8)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel)
    _set_scale(ax, scale, ylabel)
    if groups:
        for i in range(1, len(cats)):
            if groups.get(cats[i]) != groups.get(cats[i - 1]):
                ax.axvline(i - 0.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.6)
        present = list(dict.fromkeys(groups.get(c) for c in cats))
        ax.legend(handles=[mpatches.Patch(color=group_colours.get(g, "#aaaaaa"), label=g or "other") for g in present],
                  title=group_label or None, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, title_fontsize=8)
    fig.tight_layout()
    return fig


def heatmap(counts: pd.DataFrame, title: str, label: str = "Number of entries", row_order=None, col_order=None,
            row_groups: dict | None = None, col_groups: dict | None = None):
    # counts: rows x columns of counts (e.g. grade x instrument), each non-zero cell labelled; groups order the rows /
    # columns and are separated by white lines
    rows = ordered(counts.index, row_order, row_groups)
    cols = ordered(counts.columns, col_order, col_groups)
    counts = counts.reindex(index=rows, columns=cols, fill_value=0)
    n_rows, n_cols = len(rows), len(cols)
    fig, ax = plt.subplots(figsize=(max(6, n_cols * (1.5 if n_cols <= 12 else 0.45) + 2),
                                    max(8, n_rows * (0.4 if n_rows <= 15 else 0.3))))
    im = ax.imshow(counts.values, aspect="auto", cmap="YlOrRd")
    fig.colorbar(im, ax=ax, label=label, shrink=0.6)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(cols, rotation=45 if n_cols <= 12 else 55, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_title(title, fontsize=14)
    font = 7 if max(n_rows, n_cols) <= 12 else 5
    for i in range(n_rows):
        for j in range(n_cols):
            if counts.iat[i, j]:
                ax.text(j, i, str(counts.iat[i, j]), ha="center", va="center", fontsize=font)
    for groups, labels, line in ((row_groups, rows, ax.axhline), (col_groups, cols, ax.axvline)):
        if groups:
            for k in range(1, len(labels)):
                if groups.get(labels[k]) != groups.get(labels[k - 1]):
                    line(k - 0.5, color="white", linewidth=1.5)
    fig.tight_layout()
    return fig


def stacked_bars(counts: dict[str, dict], title: str, ylabel: str = "Number of entries", segment_order=None,
                 segment_labels: dict | None = None, segment_colours: dict | None = None, scale: str = "count"):
    # one stacked bar per key of counts (e.g. a source dataset), split into segments (e.g. outcomes); with
    # scale="pct", each bar is its proportions, labelled with its total (n=...). segment_order: top to bottom, in the
    # bars and the legend alike
    bars = [b for b in counts if sum(counts[b].values())]
    segments = ordered({s for b in bars for s, n in counts[b].items() if n}, segment_order)
    segment_labels = segment_labels or {}
    segment_colours = segment_colours or palette(segments)
    legend_width = max((len(str(segment_labels.get(s, s))) for s in segments), default=0) * 0.075   # inches
    fig, ax = plt.subplots(figsize=(max(4.5, len(bars) * 1.4 + 1.5) + legend_width, 6))
    for x, b in enumerate(bars):
        total = sum(counts[b].values())
        bottom = 0.0
        for s in reversed(segments):   # drawn bottom up
            n = counts[b].get(s, 0)
            if n:
                height = n / total if scale == "pct" else n
                ax.bar(x, height, 0.6, bottom=bottom, color=segment_colours.get(s, "#aaaaaa"))
                bottom += height
        if scale == "pct":
            ax.text(x, bottom + 0.01, f"n={total}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([textwrap.fill(b.replace("_", " "), 14, break_long_words=False) for b in bars], fontsize=9)
    ax.set_title(title, fontsize=13)
    _set_scale(ax, scale, ylabel, headroom=0.15)
    ax.legend(handles=[mpatches.Patch(color=segment_colours.get(s, "#aaaaaa"), label=segment_labels.get(s, s))
                       for s in segments], loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    fig.tight_layout()
    return fig


def grouped_bars(values: dict[str, dict[str, float]], title: str, ylabel: str = "Score", series_order=None,
                 series_colours: dict | None = None, ylim=(0, 1.08)):
    # one group of bars per key of values (e.g. a class, or a model), one bar per series within it (e.g. a model, or
    # a metric), each labelled with its value
    groups = list(values)
    series = ordered({s for g in groups for s in values[g]}, series_order)
    series_colours = series_colours or palette(series)
    width = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(max(6, len(groups) * (0.35 * len(series) + 0.8) + 2), 4.6))
    for j, s in enumerate(series):
        offset = (j - (len(series) - 1) / 2) * width
        for i, g in enumerate(groups):
            if (v := values[g].get(s)) is not None:
                ax.bar(i + offset, v, width, color=series_colours.get(s, "#aaaaaa"), label=s if i == 0 else None)
                ax.text(i + offset, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([str(g).replace("_", " ") for g in groups], fontsize=9)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.tight_layout()
    return fig


def save(fig, path, fmt: str = "pdf") -> None:
    # path without extension, e.g. out/plots/datasets/graded_pieces/instrument; fmt: one of FORMATS
    path = Path(path).with_name(f"{Path(path).name}.{fmt}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved {path}")


def save_scales(make: Callable[[str], object], path, scales=("count",), fmt: str = "pdf") -> None:
    # make(scale) draws the chart at each of scales (from SCALES); saved as <path>, <path>_log, <path>_pct
    for scale in scales:
        save(make(scale), f"{path}{_SUFFIX[scale]}", fmt)


def _set_scale(ax, scale: str, ylabel: str, headroom: float = 0.08) -> None:
    if scale == "log":
        ax.set_yscale("log")
        ax.set_ylabel(f"{ylabel} (log scale)")
    elif scale == "pct":
        ax.set_ylim(0, 1 + headroom)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.set_ylabel("Proportion")
    else:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.set_ylabel(ylabel)
