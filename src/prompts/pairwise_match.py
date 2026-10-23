from pathlib import Path

from graph.relation import WORK_RELATIONS, Relation

# The answer labels used by the pairwise prompts, and the relation (A -> B) each one means
LABELS = {
    "The same piece": Relation.IS,
    "A is a partial of B": Relation.IS_PART_OF,
    "B is a partial of A": Relation.CONTAINS,
    "Distinct piece": Relation.IS_NOT,
}
assert set(LABELS.values()) <= WORK_RELATIONS

PAIRWISE = Path(__file__).parent / "pairwise"


def pairwise_prompt(query: str, target: str, style: str = "direct", examples: str | None = None) -> str:
    # the pairwise template for queries from one dataset and targets from another, e.g. ("graded_pieces", "musicbrainz"):
    # pairwise/<style>.txt (task, labels, rules) with the parts that depend on the datasets filled in:
    #   query_<query>.txt        how side A's fields are laid out
    #   target_<target>.txt      what side B's text may hold
    #   examples_<examples>.txt  worked examples (default: written for side A's dataset, examples_<query>.txt)
    def part(name):
        return (PAIRWISE / f"{name}.txt").read_text(encoding="utf-8")
    return (part(style).replace("{query_fields}", part(f"query_{query}"))
            .replace("{target_fields}", part(f"target_{target}"))
            .replace("{examples}", part(f"examples_{examples or query}")))
