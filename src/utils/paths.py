from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_paths(config: Path = ROOT / "config" / "paths.yaml") -> dict:
    """config/paths.yaml as nested dicts of Paths, relative paths resolved against the repo root; URIs (e.g.
    anthropic://claude-haiku-4-5) stay strings."""
    def resolve(node):
        if isinstance(node, dict):
            return {k: resolve(v) for k, v in node.items()}
        if "://" in str(node):
            return str(node)
        return ROOT / node   # an absolute path stays as it is
    return resolve(yaml.safe_load(config.read_text()))


PATHS = load_paths()
