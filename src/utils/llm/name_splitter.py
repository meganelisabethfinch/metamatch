from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict

from tqdm import tqdm

from .name_split_parser import NameParts, NameSplitParser, initials_as_initials, spelled_as_written
from .parser import MalformedResponseError

logger = logging.getLogger(__name__)

PLACEHOLDER_NAMES = {"", "anon.", "anon", "anonymous", "unknown", "various", "n/a"}


class NameSplitter():
    # splits name strings into NameParts with an LLM, caching each (normalised) string's answer across runs

    def __init__(self, llm, template, cache_path=None, batch_size=4, generate_kwargs=None):
        self.llm = llm
        self.template = template
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.generate_kwargs = generate_kwargs or {
            "max_new_tokens": 256, "do_sample": False, "tokenizer_encode_kwargs": {"enable_thinking": False},
        }
        self.parser = NameSplitParser()
        self.cache: dict[str, list[dict]] = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                self.cache = json.load(f)

    def split(self, names):
        # placeholders ("Anon.", ...) -> [] without asking the LLM; malformed responses aren't cached (retried next run)
        as_written = {}   # the LLM gets the name as written (it copies it into substring), cached under the key
        for n in names:
            as_written.setdefault(_norm_name(n), n.strip())
        todo = sorted(as_written.keys() - self.cache.keys() - PLACEHOLDER_NAMES)
        for start in tqdm(range(0, len(todo), self.batch_size), desc="Splitting names", disable=not todo):
            batch = todo[start:start + self.batch_size]
            chats = [[{"role": "user", "content": self.template.replace("{composer_string}", as_written[n])}]
                     for n in batch]
            outputs = self.llm(chats, return_full_text=False, batch_size=self.batch_size, **self.generate_kwargs)
            for name, out in zip(batch, outputs):
                try:
                    self.cache[name] = [asdict(p) for p in self.parser.parse(out[0]["generated_text"])]
                except MalformedResponseError as e:
                    logger.warning(f"Couldn't split {name!r}: {e}")
            self._save()

        result = {}
        for n in names:
            key = _norm_name(n)
            if key in PLACEHOLDER_NAMES:
                result[n] = []
            elif key in self.cache:
                # the checks apply to answers cached before them, too
                result[n] = [spelled_as_written(initials_as_initials(NameParts(**p)), n) for p in self.cache[key]]
        return result

    def _save(self):
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        tmp = f"{self.cache_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False)
        os.replace(tmp, self.cache_path)


def _norm_name(name):
    # the LLM's split doesn't depend on case or spacing, so e.g. "BACH, J. S." and "Bach, J. S." share a cache entry
    return " ".join(name.strip().lower().split())
