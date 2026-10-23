# plots of evaluate.py's scores (out/eval/): the pipeline eval's outcomes and where labelled works were lost, per job;
# the pair eval's confusion matrix, per-class recall and overall metrics, per job and model

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from evaluation import pipeline as P
from evaluation.pairs import CLASSES, MALFORMED
from graph.edge_set import EdgeSet
from reconciliation.jobs import JOBS
from reconciliation.reconciliation_service import merge_in_order
from utils.paths import PATHS

from . import plots

logger = logging.getLogger(__name__)

# which of plots.SCALES to draw each chart at
PIPELINE_SCALES = ("count", "pct")

OUTCOME_COLOURS = {P.CORRECT: "#27ae60", P.CORRECTLY_UNMATCHED: "#a9dfbf", P.WRONG_RELATION: "#f1c40f",
                   P.OTHER_WORK: "#e67e22", P.FALSE_MATCH: "#e74c3c"}
LOST_COLOURS = {P.FOUND: "#27ae60", P.NOT_JUDGED: "#f1c40f", P.NOT_A_CANDIDATE: "#7f8c8d", P.NOT_IN_TARGET: "#d5d8dc"}
STAGE_COLOURS = ["#8e44ad", "#3498db", "#c39bd3", "#5d6d7e", "#d35400"]   # the jobs' own stage reasons
METRICS = {"accuracy": "accuracy", "precision": "macro_precision", "recall": "macro_recall", "macro-F1": "macro_f1"}
DETECTION = ["accuracy", "precision", "recall", "f1"]


def plot_evaluation(names=None, out_dir=None, fmt: str = "pdf", scales=PIPELINE_SCALES) -> None:
    # names: jobs (default: every job with scores in out/eval/). Pipeline scores are per run: a job, with its model
    # if it takes one (e.g. "gp_works_mb.qwen35-9b"), one bar each
    scores = Path(PATHS["evaluation"])
    out_dir = Path(out_dir or PATHS["plots"]) / "eval"
    runs = sorted(p.stem for p in (scores / "pipeline").glob("*.csv"))
    paired = {p.name for p in (scores / "llm").glob("*") if p.is_dir()}
    names = names or sorted({r.split(".")[0] for r in runs} | paired)
    scored = {r: pd.read_csv(scores / "pipeline" / f"{r}.csv", dtype=str, keep_default_na=False)
              for r in runs if r.split(".")[0] in names}
    if scored:
        plot_pipeline(scored, out_dir / "pipeline", fmt, scales)
    for n in names:
        results = [json.loads(p.read_text()) for p in sorted((scores / "llm" / n).glob("*.json"))]
        if results:
            plot_pairs(n, results, out_dir / "llm" / n, fmt)


def plot_pipeline(scored: dict[str, pd.DataFrame], out_dir: Path, fmt: str = "pdf", scales=PIPELINE_SCALES) -> None:
    # per job (one bar each): each labelled query's outcome, and for the positively labelled ones, where the labelled
    # work was lost - both top to bottom in pipeline order, ending with the successes
    stages = merge_in_order(*(_reasons(n) for n in scored))
    not_matched = [f"{P.NOT_MATCHED}: {r}" for r in stages]
    outcomes = {_label(n): df["outcome"].value_counts().to_dict() for n, df in scored.items()}
    order = [*not_matched, P.FALSE_MATCH, P.OTHER_WORK, P.WRONG_RELATION, P.CORRECTLY_UNMATCHED, P.CORRECT]
    colours = OUTCOME_COLOURS | {r: STAGE_COLOURS[i % len(STAGE_COLOURS)] for i, r in enumerate(not_matched)}
    plots.save_scales(lambda s: plots.stacked_bars(
        outcomes, "Pipeline eval: outcome of each labelled query", "Labelled queries", segment_order=order,
        segment_colours=colours, scale=s), out_dir / "outcomes", scales=scales, fmt=fmt)

    lost = {_label(n): df["lost"][df["lost"] != ""].value_counts().to_dict() for n, df in scored.items()}
    order = [P.NOT_IN_TARGET, P.NOT_A_CANDIDATE, *stages, P.NOT_JUDGED, P.FOUND]
    colours = LOST_COLOURS | {r: STAGE_COLOURS[i % len(STAGE_COLOURS)] for i, r in enumerate(stages)}
    plots.save_scales(lambda s: plots.stacked_bars(
        lost, "Pipeline eval: where each labelled work was lost", "Positively labelled queries", segment_order=order,
        segment_colours=colours, scale=s), out_dir / "lost", scales=scales, fmt=fmt)


def plot_pairs(name: str, results: list[dict], out_dir: Path, fmt: str = "pdf") -> None:
    # one job's pair eval, for each model scored: confusion matrices, per-class recall and the overall metrics
    classes = [c.name for c in CLASSES]
    for r in results:
        confusion = pd.DataFrame(r["confusion"]).T.loc[classes]
        confusion = confusion.loc[:, (confusion.sum() > 0) | (confusion.columns != MALFORMED)]
        plots.save(plots.heatmap(confusion, f"{name} [{r['model']}]: label (rows) vs judged (columns)",
                                 label="Pairs", row_order=classes, col_order=[*classes, MALFORMED]),
                   out_dir / f"confusion_{r['model']}", fmt=fmt)
    recall = {c: {r["model"]: r["per_class"][c]["recall"] for r in results} for c in classes
              if any(r["per_class"][c]["n"] for r in results)}
    plots.save(plots.grouped_bars(recall, f"{name}: recall per label", "Recall (share of the label judged right)"),
               out_dir / "recall_per_label", fmt=fmt)
    overall = {r["model"]: {m: r[key] for m, key in METRICS.items()} for r in results}
    plots.save(plots.grouped_bars(overall, f"{name}: overall", series_order=list(METRICS)), out_dir / "overall",
               fmt=fmt)
    detection = {r["model"]: {m: r["match_detection"][m] for m in DETECTION} for r in results if "match_detection" in r}
    if detection:
        plots.save(plots.grouped_bars(detection, f"{name}: match detection (any match vs no match)",
                                      series_order=DETECTION), out_dir / "match_detection", fmt=fmt)


def _label(run: str) -> str:
    # a run's bar label: "gp_works_mb.qwen35-9b" -> "gp_works_mb [qwen35-9b]"
    name, _, model = run.partition(".")
    return f"{name} [{model}]" if model else name


def _reasons(run: str) -> list[str]:
    # the run's reasons in pipeline order, as its eval output saved them (meta["reasons"]); run: e.g.
    # "gp_works_mb.qwen35-9b"
    name, _, model = run.partition(".")
    job = JOBS[name]
    source, target = job.source(), job.target()
    path = Path(PATHS["evaluation"]) / "pipeline" / "runs" / job.output_name(source, target, model or None)
    return EdgeSet.from_csv(path).meta.get("reasons", []) if path.exists() else []
