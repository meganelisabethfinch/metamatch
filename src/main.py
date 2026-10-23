# Runs the whole pipeline: preprocesses the datasets, reconciles composers, then works (whose prefilters follow the
# composer links). Every step resumes where it left off unless --rerun; each can also be run alone (preprocess.py,
# reconcile.py).
# e.g. python src/main.py
#      python src/main.py --model vote-qwen --plot
#
# Still to come (TODO.md): cross-corpus queries - for each graded_pieces.works entry, its imslp.works entry (if any),
# following graded_pieces.works -> musicbrainz.works -> imslp.works, output as difficulty-labelled IMSLP works

from __future__ import annotations

import argparse
import logging
import sys

from preprocess import DEFAULT_SPLIT_LLM, SOURCES, preprocess
from reconcile import run_jobs
from reconciliation.jobs import DEFAULT_LLM, model_names, uses_remote
from utils.llm.model import is_remote, llm_names

logger = logging.getLogger(__name__)

# no dependencies between these
COMPOSER_JOBS = ["gp_composers_wd", "imslp_composers_wd", "gp_composers_mb", "imslp_composers_mb"]
# each needs the composer jobs first
WORKS_JOBS = ["gp_works_mb_tree", "gp_works_imslp", "imslp_works_mb"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the whole pipeline: preprocess, reconcile composers, then works.")
    ap.add_argument("--rerun", action="store_true", help="start every step over instead of resuming")
    ap.add_argument("--limit", type=int, help="only the first N queries of each reconciliation job (to try it out)")
    ap.add_argument("--model", nargs="+", default=[DEFAULT_LLM], choices=model_names(), metavar="MODEL",
                    help=f"for the works jobs: one or more of {', '.join(model_names())}. default: {DEFAULT_LLM}")
    local = [m for m in llm_names() if not is_remote(m)]
    ap.add_argument("--split-model", choices=local, default=DEFAULT_SPLIT_LLM, metavar="MODEL",
                    help=f"local LLM for name splitting: {', '.join(local)}. default: {DEFAULT_SPLIT_LLM}")
    ap.add_argument("--allow-remote", action="store_true", help="allow remote models (Anthropic API: costs money)")
    ap.add_argument("--plot", action="store_true", help="then draw the datasets' and reconciliation plots")
    args = ap.parse_args()
    if (remote := [m for m in args.model if uses_remote(m)]) and not args.allow_remote:
        ap.error(f"{remote} run remotely and cost money per call: add --allow-remote to use them")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    preprocess(SOURCES, args.split_model, resume=not args.rerun, plot=args.plot)
    stopped = run_jobs(COMPOSER_JOBS, resume=not args.rerun, limit=args.limit)
    stopped |= run_jobs(WORKS_JOBS, args.model, resume=not args.rerun, limit=args.limit)

    if args.plot:
        from stats.reconciliation import plot_reconciliation
        plot_reconciliation()
    if stopped:
        logger.error(f"{len(stopped)} job(s) stopped: " + "; ".join(f"{n} ({e})" for n, e in stopped.items()))
        sys.exit(1)
