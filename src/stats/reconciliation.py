# plots of reconciliation jobs' outputs (datasets/metamatch/): how many queries each job matched and why the rest
# didn't, and whether two jobs that should agree (via a same_as link between their targets) do

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from data.dataset import View
from graph.edge_set import EdgeSet
from reconciliation.job import Job
from reconciliation.jobs import JOBS
from reconciliation.reconciliation_service import merge_in_order
from utils.paths import PATHS

from . import plots

logger = logging.getLogger(__name__)

MATCHED = "matched"
NOT_RECORDED = "not matched: no reason recorded"   # outputs saved before reasons were recorded
OUTCOME_COLOURS = {MATCHED: "#27ae60", NOT_RECORDED: "#d5d8dc"}
REASON_COLOURS = ["#e67e22", "#f1c40f", "#8e44ad", "#e74c3c", "#3498db", "#7f8c8d", "#d35400", "#bdc3c7", "#c39bd3",
                  "#5d6d7e"]   # no greens: green is matched

AGREE, DISAGREE = "agree", "disagree"
AGREEMENT_COLOURS = {AGREE: "#2ecc71", DISAGREE: "#e74c3c"}

# which of plots.SCALES to draw each chart at
RATE_SCALES = ("count", "pct")
AGREEMENT_SCALES = ("count", "pct")


@dataclass
class JobOutput:
    """A job's saved EdgeSet, with the views it links (and the LLM it used, for a job that takes one)."""
    job: Job
    source: View
    target: View
    edges: EdgeSet
    model: str | None = None

    @property
    def dataset(self) -> str:
        # what its bar is labelled: the source dataset, e.g. "graded_pieces", and the model if any (one bar each)
        return f"{self.source.dataset.name} [{self.model}]" if self.model else self.source.dataset.name

    def matches(self) -> dict[str, list[str]]:
        # source id -> the target ids it matched (IS_NOT edges excluded)
        matched = {}
        for e in self.edges.edges:
            if e.relation is None or e.relation.is_match:
                matched.setdefault(e.source_id, []).append(e.target_id)
        return matched

    def outcomes(self) -> dict[str, str]:
        # source id -> MATCHED or why it wasn't (meta["outcomes"]), for every queried source id
        meta, matched = self.edges.meta, self.matches()
        queried = meta.get("queried") or [e.id for e in self.source.entities()]
        reasons = meta.get("outcomes", {})
        return {i: MATCHED if i in matched else reasons.get(i, NOT_RECORDED) for i in queried}


def load(job: Job) -> list[JobOutput]:
    # the job's saved outputs: one, or one per model it has been run with (job.takes_model); none if it hasn't run
    source, target = job.source(), job.target()
    folder = Path(PATHS["metamatch"])
    if not job.takes_model:
        path = folder / job.output_name(source, target)
        return [JobOutput(job, source, target, EdgeSet.from_csv(path))] if path.exists() else []
    stem = job.output_name(source, target).removesuffix(".csv")
    return [JobOutput(job, source, target, EdgeSet.from_csv(p), p.name.removeprefix(f"{stem}.").removesuffix(".csv"))
            for p in sorted(folder.glob(f"{stem}.*.csv"))]


def plot_reconciliation(names=None, out_dir=None, fmt: str = "pdf", rate_scales=RATE_SCALES,
                        agreement_scales=AGREEMENT_SCALES) -> None:
    # names: job names (default: every job that has an output)
    out_dir = Path(out_dir or PATHS["plots"]) / "reconciliation"
    outputs = [o for n in (names or JOBS) for o in load(JOBS[n])]
    by_target = {}
    for o in outputs:
        by_target.setdefault(str(o.target), []).append(o)
    for target, group in by_target.items():
        plot_rate(group, out_dir / "rate" / f"to_{target}", fmt, rate_scales)
    by_targets = {}
    for a, b in agreeing_pairs(outputs):
        by_targets.setdefault((str(a.target), str(b.target)), []).append((a, b))
    for (a, b), pairs in by_targets.items():
        plot_agreement(pairs, out_dir / "agreement" / f"{a}_x_{b}", fmt, agreement_scales)


def plot_rate(outputs: list[JobOutput], path, fmt: str = "pdf", scales=RATE_SCALES) -> None:
    # one stacked bar per source dataset (all to the same target): each reason for not matching, then matched
    counts = {o.dataset: Counter(by_text(o.source, o.outcomes(), _text_outcome).values()) for o in outputs}
    total = Counter()
    for c in counts.values():
        total.update(c)
    # pipeline order (meta["reasons"], saved by reconcile.py), then any other reasons, most common first
    pipeline = merge_in_order(*(o.edges.meta.get("reasons", []) for o in outputs))
    reasons = [r for r in pipeline if total[r]] + [r for r, _ in total.most_common()
                                                   if r not in pipeline and r not in (MATCHED, NOT_RECORDED)]
    order = [NOT_RECORDED, *reasons, MATCHED]   # top to bottom: where the pipeline drops them, then matched
    colours = OUTCOME_COLOURS | {r: REASON_COLOURS[i % len(REASON_COLOURS)] for i, r in enumerate(reasons)}
    plots.save_scales(lambda s: plots.stacked_bars(
        counts, f"Reconciliation to {outputs[0].target}", "Number of distinct queries",
        segment_order=order, segment_colours=colours, scale=s), path, scales=scales, fmt=fmt)


def agreeing_pairs(outputs: list[JobOutput]) -> list[tuple[JobOutput, JobOutput]]:
    # (a, b): jobs from the same source whose targets are linked, a's target holding b's ids in a same_as property
    # (e.g. musicbrainz.artists.wikidata_qid -> wikidata.entities), so a's and b's matches can be compared
    return [(a, b) for a in outputs for b in outputs
            if str(a.source) == str(b.source) and str(b.target) in a.target.same_as.values()]


def link_property(a: JobOutput, b: JobOutput) -> str:
    # the property of a's targets holding b's target ids, e.g. "wikidata_qid"
    return next(p for p, view in a.target.same_as.items() if view == str(b.target))


def agreement(a: JobOutput, b: JobOutput) -> dict[str, str]:
    # source id -> how a's match (through its same_as link) compares with b's match, for ids both matched
    property = link_property(a, b)
    a_matches, b_matches = a.matches(), b.matches()
    both = [i for i in a_matches if i in b_matches]
    linked = {t.id: t.properties.get(property) or "" for t in a.target.get(sorted({t for i in both for t in a_matches[i]}))}
    result = {}
    for i in both:
        via_a = {linked.get(t, "") for t in a_matches[i]} - {""}
        if not via_a:
            result[i] = f"{a.target} match has no {property}"
        else:
            result[i] = AGREE if via_a & set(b_matches[i]) else DISAGREE
    return result


def plot_agreement(pairs: list[tuple[JobOutput, JobOutput]], path, fmt: str = "pdf",
                   scales=AGREEMENT_SCALES) -> None:
    # one stacked bar per source dataset (pairs: the same two targets): of the names both jobs matched, do the
    # matches agree?
    a, b = pairs[0]
    no_link = f"{a.target} match has no {link_property(a, b)}"
    counts = {x.dataset: Counter(by_text(x.source, agreement(x, y), _first).values()) for x, y in pairs}
    plots.save_scales(lambda s: plots.stacked_bars(
        counts, f"Agreement: {a.target} (via {link_property(a, b)}) vs {b.target}", "Distinct names matched by both",
        segment_order=[no_link, DISAGREE, AGREE], segment_colours=AGREEMENT_COLOURS | {no_link: "#f39c12"}, scale=s),
        path, scales=scales, fmt=fmt)


def by_text(source: View, values: dict[str, str], combine) -> dict[str, str]:
    # source id -> value  =>  distinct query text -> one value (e.g. a name credited on many rows counts once)
    texts = {e.id: e.text for e in source.get(list(values))}
    grouped = {}
    for i, value in values.items():
        grouped.setdefault(texts.get(i, i), []).append(value)
    return {text: combine(vs) for text, vs in grouped.items()}


def _text_outcome(outcomes: list[str]) -> str:
    # a text is matched if any of its queries was; otherwise its most common reason
    return MATCHED if MATCHED in outcomes else Counter(outcomes).most_common(1)[0][0]


def _first(values: list[str]) -> str:
    return values[0]
