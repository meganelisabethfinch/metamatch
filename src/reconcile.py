# Runs reconciliation jobs (defined in reconciliation/jobs.py): for each job, links the entities of a source view to
# a target view with the job's ReconciliationService, and saves the result as an EdgeSet in datasets/metamatch/.
# e.g. python src/reconcile.py gp_composers_wd gp_composers_mb
#
# The same script serves every stage, e.g.
#   composers: graded_pieces.composer_credits -> wikidata.entities    (no prefilter)
#   works:     graded_pieces.works -> musicbrainz.works               (prefiltered: only works by the same composer)

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

from tqdm import tqdm

from data.eval import GROUND_TRUTH
from graph.edge import Edge
from graph.edge_set import EdgeSet
from graph.entity_graph import EntityGraph
from graph.relation import Relation
from reconciliation.jobs import DEFAULT_LLM, JOBS, model_names, uses_remote
from reconciliation.reconciliation_service import ServiceUnavailable
from utils.paths import PATHS

logger = logging.getLogger(__name__)


def reconcile(source, target, service, allowed, queries, candidates=None, provenance=None) -> EdgeSet:
    # source / target: Views; service: a ReconciliationService; allowed: the relations the edges may have
    # queries: the source entities to reconcile (all of them, or those not done yet when resuming)
    # candidates: optional function source id -> candidate target ids (the prefilter). Without one, the service gets no
    #   targets, so it must search the target itself (e.g. OpenRefine, MusicBrainz search)
    # provenance: what made the edges, recorded on each (e.g. "gp_works_mb.qwen35-9b")
    if candidates is None:
        targets = None
    else:
        ids = {q.id: candidates(q.id) for q in queries}
        entities = {e.id: e for e in target.get(sorted({i for c in ids.values() for i in c}))}
        queries = [q for q in queries if ids[q.id]]   # no candidates: no service call, no edges
        targets = {q.id: [entities[i] for i in ids[q.id] if i in entities] for q in queries}
    matches = service.reconcile_all(queries, targets) if queries else {}
    return EdgeSet.from_matches(source, target, matches, allowed, provenance=provenance)


def build_graph(path) -> EntityGraph:
    # the EntityGraph a prefilter follows: the IS edges reconciliation has saved in datasets/metamatch/ (e.g. composer
    # credits -IS-> MusicBrainz artists / Wikidata entities), plus the same-entity links the path's views hold (e.g.
    # MusicBrainz artists' Wikidata QIDs), only for the ids in the graph. Nothing here is specific to composers ->
    # works: any path works (e.g. composers prefiltered by a known work)
    graph = EntityGraph()
    for csv in sorted(PATHS["metamatch"].glob("*.csv")):
        graph.add(EdgeSet.from_csv(csv).filter(relations={Relation.IS}))
    for view in _views(path):
        for property, other in view.same_as.items():
            linked = view.referencing(property, graph.ids(other))   # other id -> ids in view
            graph.add(EdgeSet(view, other, [Edge(id, o, Relation.IS) for o, ids in linked.items() for id in ids]))
    return graph


def candidates_from_path(graph, source, path):
    # the prefilter for a job with a candidates_path: source id -> the target ids graph.follow reaches along path
    # e.g. path = [("composers", gp_credits), mb_artists, ("composers", mb_works)]: MB works by the same composer
    views = [source, *_views(path)]
    return lambda id: graph.follow(source, id, path, views)


def _views(path):
    return [step[1] if isinstance(step, tuple) else step for step in path if not isinstance(step, str)]


def run(job, resume=True, limit=None, chunk_size=None, reason_by="best", eval_only=False, model=None) -> EdgeSet:
    # resume: carry on from the saved output, skipping every source id already queried (matched or not);
    #   otherwise rerun from scratch, replacing it. limit: only the first `limit` queries (e.g. to try a job out)
    # reason_by: for unmatched queries, why ("best": the best candidate's failure; "last": where the last one fell)
    # eval_only: only the source entities with a hand label (data/eval.py), saved apart in out/eval/pipeline/runs/
    #   (it isn't part of the graph); the same service and caches, so pieces the full run has judged are free
    # model: for a job that takes an LLM (job.takes_model), its name in paths.yaml models.llm (default DEFAULT_LLM)
    # chunk_size: queries reconciled (and saved) together; default 100, or every query for a remote model, whose
    #   batches each take minutes: few large ones beat many small (answers are cached as they come, so none are lost)
    model = (model or DEFAULT_LLM) if job.takes_model else None
    source, target, service = job.source(), job.target(), job.make_service(model)
    folder = PATHS["evaluation"] / "pipeline" / "runs" if eval_only else PATHS["metamatch"]
    path = folder / job.output_name(source, target, model)
    provenance = run_name(job, model)
    labelled = eval_ids(source, target) if eval_only else None
    candidates = None
    if job.candidates_path:
        steps = job.candidates_path()
        candidates = candidates_from_path(build_graph(steps), source, steps)

    edges, queried, outcomes = [], [], {}
    if resume and path.exists():
        saved = EdgeSet.from_csv(path)
        edges, queried, outcomes = saved.edges, saved.meta.get("queried", []), saved.meta.get("outcomes", {})
    done = set(queried)
    queries = [e for e in source.entities() if e.id not in done and (labelled is None or e.id in labelled)][:limit]
    logger.info(f"{job.name}{f' [{model}]' if model else ''}: {source} -> {target}: {len(queries)} to query "
                f"({len(done)} already done)")

    chunk_size = chunk_size or (max(len(queries), 1) if model and uses_remote(model) else 100)
    edge_set = EdgeSet(source, target, edges, job.allowed)
    matched = {e.source_id for e in edges if e.relation is None or e.relation.is_match}
    with tqdm(total=len(done) + len(queries), initial=len(done), desc=job.name, unit="query") as bar:
        for start in range(0, len(queries), chunk_size):
            chunk = queries[start:start + chunk_size]
            new = reconcile(source, target, service, job.allowed, chunk, candidates, provenance)
            queried += [q.id for q in chunk]
            outcomes |= _outcomes(service, target, [q for q in chunk if not _matched(new, q.id)], candidates, reason_by)
            meta = {"job": job.name, "model": model, "provenance": provenance, "service": repr(service),
                    "reason_by": reason_by,
                    "date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "queried": queried,
                    "outcomes": outcomes, "reasons": reasons(service, candidates), "eval": eval_only}
            edge_set = EdgeSet(source, target, edge_set.edges + new.edges, job.allowed, meta)
            edge_set.to_csv(path)
            matched |= {e.source_id for e in new.edges if e.relation is None or e.relation.is_match}
            bar.set_postfix(matched=len(matched), refresh=False)
            bar.update(len(chunk))

    logger.info(f"{job.name}: {len(matched)}/{len(queried)} queried {source} matched")
    return edge_set


def run_jobs(names, models=(DEFAULT_LLM,), **options) -> dict[str, str]:
    # runs the jobs in order, a job that takes a model once per model; options: run()'s. Jobs with a prefilter need
    # the jobs they depend on to have run first (e.g. composers before works). A job whose remote service stays
    # unreachable stops (its progress up to the last chunk is saved) and the next job runs; rerunning resumes it.
    # Returns the stopped runs: label -> why
    stopped = {}
    for name in names:
        for model in (models if JOBS[name].takes_model else [None]):
            try:
                run(JOBS[name], model=model, **options)
            except ServiceUnavailable as e:
                label = f"{name} [{model}]" if model else name
                logger.error(f"{label}: stopped: {e}. Progress so far is saved; rerun without --rerun to resume.")
                stopped[label] = str(e)
    return stopped


def run_name(job, model=None) -> str:
    # what a run's edges record as their provenance, and its eval scores are named: the job, and its model if it
    # takes one, e.g. "gp_works_mb.qwen35-9b"
    return f"{job.name}.{model}" if model else job.name


def eval_ids(source, target) -> set[str]:
    # the source ids with a hand label for the target dataset (GROUND_TRUTH in data/eval.py)
    if str(source) not in GROUND_TRUTH:
        raise ValueError(f"no ground truth for {source}; there is for {', '.join(GROUND_TRUTH)}")
    return {i for t in GROUND_TRUTH[str(source)](target.dataset.name) for i in t.source_ids}


def _matched(edge_set, source_id) -> bool:
    return any(e.relation is None or e.relation.is_match for e in edge_set.targets(source_id))


def reasons(service, candidates=None) -> list[str]:
    # every reason _outcomes can give, in pipeline order (meta["reasons"]: plots order the outcomes by it)
    return ["prefilter: no candidates"] * (candidates is not None) + ["no candidates", *service.reasons()]


def _outcomes(service, target, unmatched, candidates, reason_by) -> dict[str, str]:
    # why each unmatched query got nothing, e.g. "Verifier: failed MusicOccupation" or "prefilter: no candidates"
    outcomes = {}
    for q in unmatched:
        targets = None
        if candidates is not None:
            targets = target.get(candidates(q.id))
            if not targets:
                outcomes[q.id] = "prefilter: no candidates"
                continue
        outcomes[q.id] = service.failure_reason(q, targets, reason_by)
    return outcomes


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run reconciliation jobs; each saves an EdgeSet in datasets/metamatch/.")
    ap.add_argument("jobs", nargs="*", help=f"job names, run in the order given: {', '.join(JOBS)}")
    ap.add_argument("--all", action="store_true", help="run every job, in the order they're defined")
    ap.add_argument("--rerun", action="store_true", help="start over instead of resuming from the saved output")
    ap.add_argument("--limit", type=int, help="only the first N queries of each job (to try it out)")
    ap.add_argument("--reason-by", choices=["best", "last"], default="best",
                    help="why an unmatched query failed: its best candidate's failure, or where the last candidate fell")
    ap.add_argument("--plot", action="store_true", help="then draw the jobs' plots (as plot.py reconciliation)")
    ap.add_argument("--eval", action="store_true",
                    help="only the hand-labelled queries (datasets/eval/), saved in out/eval/pipeline/runs/")
    ap.add_argument("--model", nargs="+", default=[DEFAULT_LLM], choices=model_names(), metavar="MODEL",
                    help=f"for jobs that use an LLM: one or more of {', '.join(model_names())} (LLMs in "
                         f"config/paths.yaml, vote sets in reconciliation/jobs.py); "
                         f"each model's output is saved apart. default: {DEFAULT_LLM}")
    ap.add_argument("--allow-remote", action="store_true", help="allow remote models (Anthropic API: costs money)")
    ap.add_argument("--chunk-size", type=int, help="queries reconciled and saved together (default 100; every query "
                                                   "for a remote model)")
    args = ap.parse_args()
    if (remote := [m for m in args.model if uses_remote(m)]) and not args.allow_remote:
        ap.error(f"{remote} run remotely and cost money per call: add --allow-remote to use them")
    if not args.jobs and not args.all:
        ap.error(f"name jobs to run, or --all. Jobs: {', '.join(JOBS)}")
    if unknown := [j for j in args.jobs if j not in JOBS]:
        ap.error(f"unknown jobs {unknown}. Jobs: {', '.join(JOBS)}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    stopped = run_jobs(list(JOBS) if args.all else args.jobs, args.model, resume=not args.rerun, limit=args.limit,
                       reason_by=args.reason_by, eval_only=args.eval, chunk_size=args.chunk_size)

    if args.plot:   # every job with an output, so a job just run is compared with ones run earlier (agreement)
        from stats.reconciliation import plot_reconciliation
        plot_reconciliation()
    if stopped:
        logger.error(f"{len(stopped)} job(s) stopped: " + "; ".join(f"{n} ({e})" for n, e in stopped.items()))
        sys.exit(1)
