# Scores a job's eval run (reconcile.py <job> --eval) against the hand labels: did the whole pipeline (prefilter,
# ranking, LLM) find each labelled query's work, with the right relation? And where it didn't, where was it lost?

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

from data.eval import GROUND_TRUTH, Truth
from graph.edge_set import EdgeSet
from graph.relation import Relation
from reconcile import build_graph, candidates_from_path, run_name
from reconciliation.job import Job
from reconciliation.jobs import DEFAULT_LLM, model_names
from utils.paths import PATHS

logger = logging.getLogger(__name__)

# a query's outcome: what the job matched it to, against its label (positive labels, then no_match labels)
CORRECT = "correct"
WRONG_RELATION = "right work, wrong relation"
OTHER_WORK = "matched a different work"
NOT_MATCHED = "not matched"   # + ": <the job's reason>", e.g. "not matched: prefilter: no candidates"
CORRECTLY_UNMATCHED = "correctly unmatched"
FALSE_MATCH = "matched, but has no match"

# where a labelled work dropped out of the job's pipeline (or the stage's own reason, e.g. "Embedding: not in top 30")
NOT_IN_TARGET = "labelled work not in the target dataset"
NOT_A_CANDIDATE = "labelled work not a prefilter candidate"
NOT_JUDGED = "labelled work never judged (search stopped at another work)"
FOUND = "found"


def score(job: Job, model: str | None = None, out_dir=None) -> pd.DataFrame:
    # one row per labelled query the eval run queried: its label, what the job matched, the outcome, and (for a
    # positive label) where its labelled work was lost. Saved, with a summary, to out/eval/pipeline/<run>.{csv,json}
    # (run: the job, and its model if it takes one, e.g. "gp_works_mb.qwen35-9b"). model may also name an imported run
    # (e.g. "imported-qwen35-9b"): its outcomes are scored, but where works were lost isn't (no cache to replay)
    model = (model or DEFAULT_LLM) if job.takes_model else None
    source, target = job.source(), job.target()
    path = Path(PATHS["evaluation"]) / "pipeline" / "runs" / job.output_name(source, target, model)
    if not path.exists():
        raise FileNotFoundError(f"no eval run for {run_name(job, model)}: run reconcile.py {job.name} --eval"
                                f"{f' --model {model}' if model else ''}")
    output = EdgeSet.from_csv(path)
    queried = set(output.meta.get("queried", []))
    truths = [t for t in GROUND_TRUTH[str(source)](target.dataset.name) if queried & set(t.source_ids)]
    matches = {}
    for e in output.edges:
        if e.relation is None or e.relation.is_match:
            matches.setdefault(e.source_id, {})[e.target_id] = e.relation
    replayable = model is None or model in model_names()
    positives = [t for t in truths if t.relation is not Relation.IS_NOT]
    lost = _where_lost(job, model, source, target, positives, queried) if replayable else {}
    rows = []
    for t in truths:
        matched = {w: r for i in t.source_ids for w, r in matches.get(i, {}).items()}
        reason = next((output.meta.get("outcomes", {}).get(i) for i in t.source_ids if i in queried), None)
        rows.append({"key": t.key, "source_ids": "|".join(t.source_ids), "label": t.relation.name,
                     "labelled": json.dumps({w: r.name for w, r in t.targets.items()}, ensure_ascii=False),
                     "matched": json.dumps({w: r.name if r else None for w, r in matched.items()}, ensure_ascii=False),
                     "outcome": outcome(t, matched, reason), "lost": lost.get(t.key, "")})
    df = pd.DataFrame(rows)
    out = Path(out_dir or PATHS["evaluation"]) / "pipeline"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{run_name(job, model)}.csv", index=False)
    (out / f"{run_name(job, model)}.json").write_text(json.dumps(summary(df), indent=2, ensure_ascii=False))
    logger.info(f"{run_name(job, model)}: {summary(df)['accuracy']}")
    return df


def outcome(truth: Truth, matched: dict[str, Relation | None], reason: str | None) -> str:
    if truth.relation is Relation.IS_NOT:
        return FALSE_MATCH if matched else CORRECTLY_UNMATCHED
    if not matched:
        return f"{NOT_MATCHED}: {reason or 'no reason recorded'}"
    if any(truth.targets.get(w) == r for w, r in matched.items()):
        return CORRECT
    return WRONG_RELATION if set(matched) & set(truth.targets) else OTHER_WORK


def summary(df: pd.DataFrame) -> dict:
    # strict accuracy: correct relation (or correctly unmatched); lenient: also the right work with another relation
    n = len(df)
    right = df["outcome"].isin([CORRECT, CORRECTLY_UNMATCHED]).sum()
    lenient = right + (df["outcome"] == WRONG_RELATION).sum()
    return {"n": n, "accuracy": {"strict": round(right / n, 4) if n else None,
                                 "lenient": round(lenient / n, 4) if n else None},
            "outcomes": dict(Counter(df["outcome"])), "lost": dict(Counter(df["lost"][df["lost"] != ""]))}


def _where_lost(job: Job, model, source, target, truths: list[Truth], queried: set[str]) -> dict[str, str]:
    # for each positively labelled query: where its labelled work(s) dropped out. Replays the job's service on the
    # query (cheap: its LLM judgements are cached, so the model is never loaded) and looks up the labelled works in
    # its explain() - the stage that rejected each - or finds they never reached it
    exists = {e.id for e in target.get(sorted({w for t in truths for w in t.targets}))}
    candidates = None
    if job.candidates_path:
        steps = job.candidates_path()
        candidates = candidates_from_path(build_graph(steps), source, steps)
    service = job.make_service(model)
    lost = {}
    for t in truths:
        labelled = [w for w in t.targets if w in exists]
        if not labelled:
            lost[t.key] = NOT_IN_TARGET
            continue
        query_id = next(i for i in t.source_ids if i in queried)
        (query,) = source.get([query_id])
        pool = candidates(query_id) if candidates else None
        if pool is not None and not set(labelled) & set(pool):
            lost[t.key] = NOT_A_CANDIDATE
            continue
        targets = target.get(pool) if pool is not None else None
        reasons = {r.target_id: r.reason for r in service.explain(query, targets)}
        found = [w for w in labelled if w not in reasons]
        if found and any(m.target.id in found for m in service.reconcile(query, targets)):
            lost[t.key] = FOUND
        else:
            lost[t.key] = reasons.get(labelled[0], NOT_JUDGED)
    return lost
