# Draws the plots (into out/plots/) from what's saved: the cleaned datasets, and reconciliation jobs' outputs.
# e.g. python src/plot.py datasets graded_pieces cipi
#      python src/plot.py reconciliation gp_composers_mb imslp_composers_mb
#      python src/plot.py eval gp_works_mb gp_works_imslp            (from evaluate.py's scores)
# (preprocess.py --plot / reconcile.py --plot call the same functions after they run)

from __future__ import annotations

import argparse
import logging

from reconciliation.jobs import JOBS
from stats import datasets, evaluation, reconciliation
from stats.datasets import DATASETS, plot_dataset
from stats.plots import FORMATS, SCALES
from stats.reconciliation import plot_reconciliation

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Draw plots of the datasets or of reconciliation outputs into out/plots/.")
    ap.add_argument("kind", choices=["datasets", "reconciliation", "eval"])
    ap.add_argument("names", nargs="*", help=f"datasets ({', '.join(DATASETS)}) or jobs ({', '.join(JOBS)}); "
                                             "default: all datasets / every job with an output (or scores, for eval)")
    ap.add_argument("--out", help="output directory (default: plots in config/paths.yaml)")
    ap.add_argument("--format", choices=FORMATS, default="pdf", help="file format of the plots (default: pdf)")
    ap.add_argument("--scales", nargs="+", choices=SCALES,
                    help="draw each bar chart at these scales: count (<name>), log (<name>_log), pct (<name>_pct). "
                         f"default per plot type: datasets {list(datasets.DEFAULT_SCALES)}, reconciliation rate "
                         f"{list(reconciliation.RATE_SCALES)} and agreement {list(reconciliation.AGREEMENT_SCALES)}, "
                         f"eval pipeline {list(evaluation.PIPELINE_SCALES)}")
    args = ap.parse_args()
    known = DATASETS if args.kind == "datasets" else list(JOBS)
    if unknown := [n for n in args.names if n not in known]:
        ap.error(f"unknown {args.kind} {unknown}; choose from {', '.join(known)}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.kind == "datasets":
        for name in args.names or DATASETS:
            plot_dataset(name, args.out, args.format, args.scales or datasets.DEFAULT_SCALES)
    elif args.kind == "reconciliation":
        plot_reconciliation(args.names or None, args.out, args.format, args.scales or reconciliation.RATE_SCALES,
                            args.scales or reconciliation.AGREEMENT_SCALES)
    else:
        evaluation.plot_evaluation(args.names or None, args.out, args.format,
                                   args.scales or evaluation.PIPELINE_SCALES)
