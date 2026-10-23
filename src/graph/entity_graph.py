from __future__ import annotations

from collections import defaultdict

from .edge import Edge
from .edge_set import EdgeSet
from .relation import Relation, compose_relations

Node = tuple[str, str]   # (view, id), so the same id in two datasets never collides


class EntityGraph():
    # The relations between entities across MANY views, built from the EdgeSets inferred by reconciliation
    # (e.g. graded_pieces.composer_credits -IS-> wikidata.entities). Answers how two entities relate by ANY path,
    # not just one spelled-out composition; follow() also steps along the references views declare
    # (e.g. a work's composers), which are facts the datasets hold rather than edges.
    #
    # Entities joined by IS edges (by any chain of them) form one cluster = one real-world entity. Every other
    # edge is followed from cluster to cluster, so e.g. A -IS-> B -IS_PART_OF-> C gives A IS_PART_OF C.

    def __init__(self, edge_sets=()):
        self._parent: dict[Node, Node] = {}                         # union-find over IS edges
        self._members: dict[Node, set[Node]] = {}                   # cluster root -> its nodes
        self._adjacent: dict[Node, list[tuple[Node, Relation | None]]] = defaultdict(list)   # node -> (node, relation)
        for edge_set in edge_sets:
            self.add(edge_set)

    def add(self, edge_set: EdgeSet) -> None:
        # IS edges merge the two ends' clusters; every other edge is stored both ways (inverse relation backwards)
        for e in edge_set.edges:
            a, b = (edge_set.source, e.source_id), (edge_set.target, e.target_id)
            if e.relation is Relation.IS:
                self._union(a, b)
            else:
                self._find(a), self._find(b)
                self._adjacent[a].append((b, e.relation))
                self._adjacent[b].append((a, e.relation.inverse() if e.relation else None))

    def cluster_of(self, view, id) -> Node:
        # the cluster (view, id) belongs to (named by one of its nodes); an entity with no IS edges is its own cluster
        return self._root((str(view), id))

    def ids(self, view) -> list[str]:
        # every id the graph holds for view
        return sorted(id for v, id in self._parent if v == str(view))

    def members(self, cluster: Node) -> dict[str, list[str]]:
        # {view: [ids]} in the cluster, e.g. {"graded_pieces.composer_credits": ["12/0"],
        #   "wikidata.entities": ["Q1339"]}
        by_view = defaultdict(list)
        for view, id in sorted(self._members.get(self._root(cluster), {cluster})):
            by_view[view].append(id)
        return dict(by_view)

    def follow(self, view, id, path, views=()) -> list[str]:
        # the ids in the last view reached from (view, id) by going through the views in path, in order. Each step moves
        # to every entity it can reach in the next view: along a reference between the two views (a property holding
        # the other's ids, e.g. a work's composers - either direction), else within a cluster (IS) or along edges
        # (never IS_NOT). A step is a view, or (property, view) to say which reference to take when there's more
        # than one between the two views (e.g. a work's composers vs its arrangers).
        # views: the Views on the path that have references. e.g. MB works by the same composer as graded work w:
        #   follow(gp_works, w, [("composers", gp_credits), mb_artists, mb_works], views=[gp_works, mb_works])
        by_name = {str(v): v for v in views}
        current_view, current = str(view), {id}
        for step in path:
            property, next_view = (step[0], str(step[1])) if isinstance(step, tuple) else (None, str(step))
            current = self._step(current_view, current, next_view, property, by_name)
            current_view = next_view
        return sorted(current)

    def _step(self, view, ids, next_view, property, views) -> set[str]:
        source, target = views.get(view), views.get(next_view)
        references = ([("forward", p) for p, v in (source.references.items() if source else ()) if v == next_view]
                      + [("backward", p) for p, v in (target.references.items() if target else ()) if v == view])
        if property is not None:
            references = [r for r in references if r[1] == property]
            if not references:
                raise ValueError(f"no reference {property!r} between {view} and {next_view}")
        if len(references) > 1:
            raise ValueError(f"more than one reference between {view} and {next_view} "
                             f"({[p for _, p in references]}): name one, e.g. ({references[0][1]!r}, {next_view!r})")
        if references:
            direction, property = references[0]
            if direction == "forward":   # e.g. a work's composers
                return {ref for e in source.get(sorted(ids)) for ref in e.properties.get(property) or []}
            return {w for ws in target.referencing(property, sorted(ids)).values() for w in ws}   # e.g. a composer's works
        reached = set()
        for id in ids:
            node = (view, id)
            for same in self._members.get(self._root(node), {node}):
                if same[0] == next_view:
                    reached.add(same[1])
                for neighbour, relation in self._adjacent.get(same, []):
                    if neighbour[0] == next_view and (relation is None or relation.is_match):
                        reached.add(neighbour[1])
        return reached

    def relations(self, view_a, id_a, view_b, id_b, max_depth=4) -> set[Relation | None]:
        # every relation a -> b found along any path of up to max_depth edges between clusters, composed with
        # Relation.compose (IS if a and b are in the same cluster). Paths can disagree, so this is a set;
        # None means related only through untyped edges
        target = self._root((str(view_b), id_b))
        return self._reach((str(view_a), id_a), max_depth).get(target, set())

    def edge_set(self, source_view, target_view, max_depth=4) -> EdgeSet:
        # an EdgeSet holding every relation found between the two views' entities, one edge per (pair, relation)
        source_view, target_view = str(source_view), str(target_view)
        edges = []
        for node in sorted(n for n in self._parent if n[0] == source_view):
            for cluster, relations in self._reach(node, max_depth).items():
                for _, target_id in sorted(n for n in self._members.get(cluster, {cluster}) if n[0] == target_view):
                    edges += [Edge(node[1], target_id, r) for r in relations]
        return EdgeSet(source_view, target_view, edges, meta={"inferred_by": "EntityGraph", "max_depth": max_depth})

    def _reach(self, start: Node, max_depth: int) -> dict[Node, set[Relation | None]]:
        # cluster -> every relation start has to it, over paths of up to max_depth edges between clusters
        found = defaultdict(set)
        start_cluster = self._root(start)
        found[start_cluster].add(Relation.IS)
        frontier = {(start_cluster, Relation.IS)}
        seen = set(frontier)
        for _ in range(max_depth):
            next_frontier = set()
            for cluster, relation in frontier:
                for node in self._members.get(cluster, {cluster}):
                    for neighbour, edge_relation in self._adjacent.get(node, []):
                        composes, composed = compose_relations(relation, edge_relation)
                        state = (self._find(neighbour), composed)
                        if composes and state not in seen:
                            seen.add(state)
                            next_frontier.add(state)
                            found[state[0]].add(composed)
            frontier = next_frontier
        return found

    def _root(self, node: Node) -> Node:
        return self._find(node) if node in self._parent else node

    def _find(self, node: Node) -> Node:
        if node not in self._parent:
            self._parent[node] = node
            self._members[node] = {node}
        root = node
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[node] != root:   # path compression
            self._parent[node], node = root, self._parent[node]
        return root

    def _union(self, a: Node, b: Node) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            return
        if len(self._members[ra]) < len(self._members[rb]):
            ra, rb = rb, ra
        self._parent[rb] = ra
        self._members[ra] |= self._members.pop(rb)
