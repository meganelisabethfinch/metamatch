# Scores reconciliation against the hand labels (datasets/eval/), into out/eval/:
#   pipeline: a job's eval run (reconcile.py <job> --eval): how often the whole job finds each labelled work, and
#             where the rest were lost
#   llm:      the job's LLM judge alone, on labelled (query, work) pairs balanced across the relations
# e.g. python src/evaluate.py pipeline gp_works_mb gp_works_imslp
#      python src/evaluate.py llm gp_works_mb --cap 400
#      python src/evaluate.py pipeline gp_works_mb_tree --model imported-qwen35-9b   (imported from the old repo)

from __future__ import annotations

import argparse
import logging

from evaluation import pairs, pipeline
from reconciliation.jobs import DEFAULT_LLM, JOBS, model_names, uses_remote

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score reconciliation against the hand labels, into out/eval/.")
    ap.add_argument("mode", choices=["pipeline", "llm"])
    ap.add_argument("jobs", nargs="+", help=f"jobs with hand labels for their source: {', '.join(JOBS)}")
    ap.add_argument("--model", nargs="+", default=[DEFAULT_LLM], metavar="MODEL",
                    help=f"the LLM(s) whose runs to score / that judge the pairs: {', '.join(model_names())}; pipeline "
                         f"also takes imported runs (e.g. imported-qwen35-9b). default: {DEFAULT_LLM}")
    ap.add_argument("--allow-remote", action="store_true", help="llm: allow remote models (costs money)")
    balancing = ap.add_argument_group("llm: rebuild the sample with other balancing (default: reuse the saved one)")
    balancing.add_argument("--seed", type=int)
    balancing.add_argument("--cap", type=int, help="at most this many pairs (default 1600)")
    balancing.add_argument("--out-of-tree", type=int, help="hierarchy targets: other works by the composer per exact "
                                                           "match, as negatives (default 3)")
    balancing.add_argument("--negatives-per-work", type=int,
                           help="flat targets: negatives per labelled work (default 1)")
    args = ap.parse_args()
    if unknown := [j for j in args.jobs if j not in JOBS]:
        ap.error(f"unknown jobs {unknown}. Jobs: {', '.join(JOBS)}")
    if args.mode == "llm" and (unknown := [m for m in args.model if m not in model_names()]):
        ap.error(f"unknown models {unknown}: {', '.join(model_names())}")
    if args.mode == "llm" and (remote := [m for m in args.model if uses_remote(m)]) and not args.allow_remote:
        ap.error(f"{remote} run remotely and cost money per call: add --allow-remote to use them")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    balance = {k: v for k, v in [("seed", args.seed), ("cap", args.cap), ("out_of_tree", args.out_of_tree),
                                 ("negatives_per_work", args.negatives_per_work)] if v is not None}
    for name in args.jobs:
        for model in args.model:
            if args.mode == "pipeline":
                pipeline.score(JOBS[name], model)
            else:
                pairs.score(JOBS[name], model, **balance)
