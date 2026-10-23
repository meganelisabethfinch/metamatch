from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod

from .reconciliation_service import Entity


class PropertySource(ABC):
    """Gets extra properties for entities, fetching each id at most once (persisted if ``cache_path`` is set)."""

    def __init__(self, cache_path: str | None = None):
        self.cache_path = cache_path
        self.cache: dict[str, dict] = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                self.cache = json.load(f)

    def get(self, entities: list[Entity]) -> dict[str, dict]:
        """Entity id -> properties."""
        unique = {e.id: e for e in entities}
        missing = [e for i, e in unique.items() if i not in self.cache]
        if missing:
            fetched = self._fetch(missing)
            for e in missing:
                self.cache[e.id] = fetched.get(e.id, {})
            self._save()
        return {i: self.cache[i] for i in unique}

    @abstractmethod
    def _fetch(self, entities: list[Entity]) -> dict[str, dict]:
        """The actual lookup (API, LLM, local index, ...); only called for uncached entities."""

    def _save(self) -> None:
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False)
        os.replace(tmp, self.cache_path)


class ViewProperties(PropertySource):
    """Properties from a View's entities, e.g. wikidata.entities (a cached dataset fetches what it lacks)."""

    def __init__(self, view):
        super().__init__()
        self.view = view

    def _fetch(self, entities: list[Entity]) -> dict[str, dict]:
        return {e.id: e.properties for e in self.view.get([e.id for e in entities])}
