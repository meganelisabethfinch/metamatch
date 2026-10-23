from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np

from .ranking_service import RankingService
from .reconciliation_service import Entity, Match


class EmbeddingService(RankingService):
    """Ranks targets by cosine similarity of sentence embeddings; target embeddings are cached on disk by text."""

    def __init__(self, model_path, cache_dir, top_k: int | None = None, threshold: float | None = None,
                 batch_size: int = 64):
        # each entity is embedded by its "embed_text" property if it has one (e.g. a plain title, without the
        # prompt labels), else its text
        super().__init__(top_k, threshold)
        from sentence_transformers import SentenceTransformer
        self.model_name = Path(str(model_path).rstrip("/")).name
        self.model = SentenceTransformer(str(model_path))
        self.batch_size = batch_size
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.cache = sqlite3.connect(Path(cache_dir) / f"{self.model_name}.sqlite")
        self.cache.execute("CREATE TABLE IF NOT EXISTS embeddings (hash TEXT PRIMARY KEY, vector BLOB)")

    def score(self, query: Entity, target: Entity) -> float:
        q, t = self._vectors([_text(query), _text(target)])
        return float(q @ t)

    def reconcile_all(self, queries: list[Entity], targets: dict[str, list[Entity]]) -> dict[str, list[Match]]:
        # all queries' and targets' texts embedded together (each distinct text once), then ranked per query
        texts = sorted({_text(e) for q in queries for e in [q, *targets[q.id]]})
        vectors = dict(zip(texts, self._vectors(texts)))
        results = {}
        for q in queries:
            matches = [Match(t, score=float(vectors[_text(q)] @ vectors[_text(t)])) for t in targets[q.id]]
            results[q.id] = self._keep_best(matches)
        return results

    def reconcile(self, query: Entity, targets: list[Entity]) -> list[Match]:
        return self.reconcile_all([query], {query.id: targets})[query.id]

    def _score_all(self, query: Entity, targets: list[Entity]) -> list[Match]:
        vectors = dict(zip(texts := sorted({_text(e) for e in [query, *targets]}), self._vectors(texts)))
        return [Match(t, score=float(vectors[_text(query)] @ vectors[_text(t)])) for t in targets]

    def _vectors(self, texts: list[str]) -> list[np.ndarray]:
        # unit-length embeddings, from the cache where possible
        hashes = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
        cached = {}
        for i in range(0, len(hashes), 900):
            part = hashes[i:i + 900]
            cached |= dict(self.cache.execute(
                f"SELECT hash, vector FROM embeddings WHERE hash IN ({','.join('?' * len(part))})", part))
        missing = [(h, t) for h, t in zip(hashes, texts) if h not in cached]
        if missing:
            vecs = self.model.encode([t for _, t in missing], batch_size=self.batch_size, convert_to_numpy=True,
                                     normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
            rows = [(h, v.tobytes()) for (h, _), v in zip(missing, vecs)]
            self.cache.executemany("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", rows)
            self.cache.commit()
            cached |= dict(rows)
        return [np.frombuffer(cached[h], dtype=np.float32) for h in hashes]

    def __repr__(self) -> str:
        return f"Embedding({self.model_name}, top_k={self.top_k})"


def _text(entity: Entity) -> str:
    return entity.properties.get("embed_text") or entity.text
