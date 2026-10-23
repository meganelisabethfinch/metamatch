# Scores a job's LLM judge on its own: labelled (query, work) pairs, built from the hand labels, each judged with the
# job's pairwise prompt, model and cache; the predicted relation is compared with the label. Measures the judge, not
# the search around it (pipeline.py measures that)

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from data.eval import GROUND_TRUTH, Truth
from graph.relation import Relation
from reconcile import build_graph, candidates_from_path
from reconciliation.job import Job
from reconciliation.jobs import DEFAULT_LLM
from utils.paths import PATHS

logger = logging.getLogger(__name__)

CLASSES = [Relation.IS, Relation.IS_PART_OF, Relation.CONTAINS, Relation.IS_NOT]
MALFORMED = "MALFORMED"   # the judge's answer couldn't be parsed


@dataclass(frozen=True)
class Pair:
    key: str           # the labelled query's eval id (e.g. a graded piece's piece_id)
    query_id: str      # its id in the job's source view
    target_id: str
    relation: str      # the label: a Relation name


def build_pairs(job: Job, seed: int = 10, cap: int = 1600, out_of_tree: int = 3,
                negatives_per_work: int = 1) -> list[Pair]:
    # the eval pairs for a job, balanced (as linked-difficulty-data's balance_eval_dataset.py): the hand labels only
    # name each query's best match, so more pairs are derived from them -
    #   target with a part-of hierarchy (e.g. MusicBrainz), from an exact match: the work's parents (IS_PART_OF), its
    #     parts (CONTAINS), and as IS_NOT its sibling parts plus `out_of_tree` other works by the composer
    #   target without (e.g. IMSLP): `negatives_per_work` other works by the composer per labelled work (IS_NOT)
    #   an unlabelled no_match: one work by the composer (IS_NOT)
    # then at most `cap` pairs, balanced across the classes (cap_and_balance). "Works by the composer" are the job's
    # prefilter candidates; seed makes the sample reproducible
    rng = random.Random(seed)
    source, target = job.source(), job.target()
    steps = job.candidates_path()
    candidates = candidates_from_path(build_graph(steps), source, steps)
    hierarchy = next((p for p, v in target.references.items() if v == str(target)), None)   # e.g. "part_of"
    truths = GROUND_TRUTH[str(source)](target.dataset.name)
    have = {e.id for e in source.entities()}
    pairs = []
    for t in truths:
        query_id = next((i for i in t.source_ids if i in have), None)
        if query_id is None:
            continue
        pool = candidates(query_id)
        pairs += _pairs(t, query_id, pool, target, hierarchy, rng, out_of_tree, negatives_per_work)
    before = Counter(p.relation for p in pairs)
    pairs = cap_and_balance(pairs, cap, rng)
    logger.info(f"{job.name}: {len(pairs)} pairs {dict(Counter(p.relation for p in pairs))} (from {dict(before)})")
    return pairs


def _pairs(t: Truth, query_id, pool, target, hierarchy, rng, out_of_tree, negatives_per_work) -> list[Pair]:
    pair = lambda work, relation: Pair(t.key, query_id, work, relation.name)
    if not t.targets:   # an unlabelled no_match: any work by the composer
        return [pair(rng.choice(pool), Relation.IS_NOT)] if pool else []
    pairs = []
    known = {e.id: e for e in target.get(list(t.targets))}
    for work, relation in t.targets.items():
        if work not in known:
            continue
        pairs.append(pair(work, relation))
        if hierarchy is None:
            others = [w for w in pool if w != work]
            rng.shuffle(others)
            pairs += [pair(w, Relation.IS_NOT) for w in others[:negatives_per_work]]
        elif relation is Relation.IS:
            parents = known[work].properties.get(hierarchy) or []
            parts = target.referencing(hierarchy, [work])[work]
            pairs += [pair(p, Relation.IS_PART_OF) for p in parents] + [pair(c, Relation.CONTAINS) for c in parts]
            related = _related(target, hierarchy, work)
            siblings = {s for p in parents for s in target.referencing(hierarchy, [p])[p]} - related
            others = [w for w in pool if w not in related and w not in siblings]
            rng.shuffle(others)
            pairs += [pair(w, Relation.IS_NOT) for w in sorted(siblings) + others[:out_of_tree]]
    return pairs


def _related(target, hierarchy, work) -> set[str]:
    # the work, everything it is (transitively) part of, and all its (transitive) parts: none of these is IS_NOT
    related = {work}
    for step in (lambda w: _parents(target, hierarchy, w), lambda w: target.referencing(hierarchy, [w])[w]):
        stack = [work]
        while stack:
            for nxt in step(stack.pop()):
                if nxt not in related:
                    related.add(nxt)
                    stack.append(nxt)
    return related


def _parents(target, hierarchy, work) -> list[str]:
    found = target.get([work])
    return (found[0].properties.get(hierarchy) or []) if found else []


def cap_and_balance(pairs: list[Pair], cap: int, rng: random.Random) -> list[Pair]:
    # at most cap pairs, each class an equal share: classes smaller than their share are kept whole, and what they
    # leave is shared among the rest
    by_class = {}
    for p in pairs:
        by_class.setdefault(p.relation, []).append(p)
    todo = [c.name for c in CLASSES if by_class.get(c.name)]
    quotas, remaining = {}, cap
    while todo:
        share = remaining // len(todo)
        if small := [c for c in todo if len(by_class[c]) <= share]:
            for c in small:
                quotas[c] = len(by_class[c])
                remaining -= len(by_class[c])
                todo.remove(c)
            continue
        base, extra = divmod(remaining, len(todo))
        quotas |= {c: base + (i < extra) for i, c in enumerate(todo)}
        break
    kept = []
    for c, group in by_class.items():
        group = list(group)
        rng.shuffle(group)
        kept += group[:quotas.get(c, 0)]
    rng.shuffle(kept)
    return kept


def judge_of(service):
    # the pairwise judge inside a works service (e.g. the GuidedSearch stage's LLM)
    for stage in getattr(service, "stages", [service]):
        if hasattr(stage, "judge"):
            return stage.judge
    raise ValueError(f"no judge in {service!r}")


def score(job: Job, model: str | None = None, pairs: list[Pair] | None = None, out_dir=None, **balance) -> dict:
    # judges every pair with the job's judge, using model (a paths.yaml models.llm name; default DEFAULT_LLM); saves
    # the pairs (datasets/eval/pairs/<job>.csv, reused if there, so every model sees the same sample), each prediction
    # and the metrics (out/eval/llm/<job>/<model>.{csv,json})
    model = model or DEFAULT_LLM
    sample = Path(PATHS["eval"]["graded_pieces"]).parent / "pairs" / f"{job.name}.csv"
    if pairs is None and sample.exists() and not balance:
        pairs = [Pair(**r) for r in pd.read_csv(sample, dtype=str, keep_default_na=False).to_dict("records")]
    if pairs is None:
        pairs = build_pairs(job, **balance)
        sample.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([asdict(p) for p in pairs]).to_csv(sample, index=False)
    source, target = job.source(), job.target()
    judge = judge_of(job.make_service(model))
    queries = {e.id: e for e in source.get(sorted({p.query_id for p in pairs}))}
    works = {e.id: e for e in target.get(sorted({p.target_id for p in pairs}))}
    predicted = [MALFORMED] * len(pairs)
    for batch in tqdm(_rounds(pairs), desc=f"{job.name}: judging pairs", unit="round"):
        # one call per round (e.g. one batch), each query asked about one work, as the search asks: same cache keys
        answers = judge.reconcile_all([queries[pairs[i].query_id] for i in batch],
                                      {pairs[i].query_id: [works[pairs[i].target_id]] for i in batch})
        for i in batch:
            match = next(iter(answers[pairs[i].query_id]), None)
            predicted[i] = match.relation.name if match and match.relation else MALFORMED
    df = pd.DataFrame([asdict(p) for p in pairs]).assign(predicted=predicted)
    df["correct"] = df["relation"] == df["predicted"]
    result = {"job": job.name, "model": model} | metrics(df["relation"], df["predicted"])
    out = Path(out_dir or PATHS["evaluation"]) / "llm" / job.name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{model}.csv", index=False)
    (out / f"{model}.json").write_text(json.dumps(result, indent=2))
    logger.info(f"{job.name} [{model}]: {len(df)} pairs, accuracy {result['accuracy']}, macro-F1 {result['macro_f1']}")
    return result


def _rounds(pairs: list[Pair]) -> list[list[int]]:
    # the pairs' indices in rounds: round k holds each query's k-th pair, so no query is asked twice in one call
    by_query = {}
    for i, p in enumerate(pairs):
        by_query.setdefault(p.query_id, []).append(i)
    longest = max(map(len, by_query.values()), default=0)
    return [[ids[k] for ids in by_query.values() if k < len(ids)] for k in range(longest)]


def metrics(truth: pd.Series, predicted: pd.Series) -> dict:
    # accuracy, per-class precision / recall / F1 and their macro averages, the confusion matrix (truth -> predicted),
    # and match detection: whether a match (any relation but IS_NOT) is told from no match
    classes = [c.name for c in CLASSES]
    per_class = {}
    for c in classes:
        tp = ((truth == c) & (predicted == c)).sum()
        precision = tp / (predicted == c).sum() if (predicted == c).any() else 0.0
        recall = tp / (truth == c).sum() if (truth == c).any() else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = {"n": int((truth == c).sum()), "precision": round(float(precision), 4),
                        "recall": round(float(recall), 4), "f1": round(float(f1), 4)}
    present = [c for c in classes if per_class[c]["n"]]
    macro = lambda k: round(sum(per_class[c][k] for c in present) / len(present), 4) if present else None
    confusion = {c: {p: int(((truth == c) & (predicted == p)).sum()) for p in [*classes, MALFORMED]} for c in classes}
    return {"n": int(len(truth)), "accuracy": round(float((truth == predicted).mean()), 4),
            "macro_precision": macro("precision"), "macro_recall": macro("recall"), "macro_f1": macro("f1"),
            "malformed": int((predicted == MALFORMED).sum()), "per_class": per_class, "confusion": confusion,
            "match_detection": match_detection(truth, predicted)}


def match_detection(truth: pd.Series, predicted: pd.Series) -> dict:
    # match (IS, IS_PART_OF, CONTAINS) vs no match (IS_NOT; an unparseable answer counts as none): binary accuracy,
    # and precision / recall / F1 of finding the matches
    is_match = lambda s: s.isin([c.name for c in CLASSES if c is not Relation.IS_NOT])
    t, p = is_match(truth), is_match(predicted)
    tp = (t & p).sum()
    precision = tp / p.sum() if p.any() else 0.0
    recall = tp / t.sum() if t.any() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": round(float((t == p).mean()), 4), "precision": round(float(precision), 4),
            "recall": round(float(recall), 4), "f1": round(float(f1), 4)}

