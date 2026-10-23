from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from pathlib import Path

from reconciliation.reconciliation_service import ServiceUnavailable
from utils.paths import PATHS

logger = logging.getLogger(__name__)

# paths.yaml models.llm entries for models behind the Anthropic API start with this, e.g. anthropic://claude-haiku-4-5
REMOTE = "anthropic://"


# Local models loaded at once: more can fill the GPU (which also drives the display) and freeze the desktop
MAX_LOADED = 1

_loaded: OrderedDict = OrderedDict()   # (path, 4-bit) -> pipeline, least recently used first


def load_llm(path: str, load_in_4bit: bool = False):
    # a Hugging Face text-generation pipeline for a local model, loaded once and kept; with MAX_LOADED already
    # loaded, the least recently used is unloaded first. 4-bit fits e.g. a 9B model on 16 GB
    key = (str(path), load_in_4bit)
    if key in _loaded:
        _loaded.move_to_end(key)
        return _loaded[key]
    while len(_loaded) >= MAX_LOADED:
        unload(next(iter(_loaded)))
    import torch
    from transformers import BitsAndBytesConfig, pipeline
    kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
    if load_in_4bit:
        kwargs["model_kwargs"] = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True,
                                                                            bnb_4bit_compute_dtype=torch.bfloat16)}
    llm = pipeline("text-generation", model=str(path), **kwargs)
    llm.tokenizer.padding_side = "left"   # decoder-only models need left padding to batch
    _loaded[key] = llm
    return llm


def unload(key) -> None:
    # frees a loaded model's GPU memory
    import gc

    import torch
    logger.info(f"unloading {key[0]} to make room for another model")
    del _loaded[key]
    gc.collect()
    torch.cuda.empty_cache()


def llm_names() -> list[str]:
    # the LLMs that can be named (paths.yaml models.llm), e.g. ["qwen25-7b", ..., "claude-haiku-4-5"]
    return list(PATHS["models"]["llm"])


def is_remote(name: str) -> bool:
    # whether the named LLM runs remotely (and costs money per call)
    return str(PATHS["models"]["llm"][name]).startswith(REMOTE)


def llm(name: str):
    # the named LLM (paths.yaml models.llm): a LocalLLM for a path, a RemoteLLM for anthropic://<model id>
    if name not in PATHS["models"]["llm"]:
        raise KeyError(f"no LLM {name!r} in config/paths.yaml models.llm: {', '.join(llm_names())}")
    where = str(PATHS["models"]["llm"][name])
    return RemoteLLM(where.removeprefix(REMOTE)) if where.startswith(REMOTE) else LocalLLM(where)


class LocalLLM:
    """A local Hugging Face model, loaded on first use (load_llm): runs whose answers are all cached never load it."""

    def __init__(self, path, load_in_4bit: bool = True, generate_kwargs: dict | None = None, batch_size: int = 4):
        # batch_size: the most prompts to run together on the GPU (bounded by its memory: 4 of our ~2,800-token
        #   pairwise prompts fit a 9B model in 4-bit on 16 GB; 8 don't)
        self.name = str(path)   # what cache keys record as the model
        self.load_in_4bit = load_in_4bit
        self.batch_size = batch_size
        self.batch = False   # by default: batching our prompts isn't faster on the GPU (prompt-heavy, short answers)
        self.settings = generate_kwargs or {   # recorded in cache keys too
            "max_new_tokens": 256,
            "do_sample": False,
            # Turns off "thinking" in models that support it (e.g. Qwen3); ignored by other chat templates.
            "tokenizer_encode_kwargs": {"enable_thinking": False},
        }

    def generate(self, prompts: list[str], prefix: str = "", batch: bool = False) -> list[str]:
        # one response per prompt; batch: through the GPU batch_size at a time (padded together), else one at a time.
        # prefix: unused, locally
        chats = [[{"role": "user", "content": p}] for p in prompts]
        size = self.batch_size if batch else 1
        outputs = load_llm(self.name, self.load_in_4bit)(chats, return_full_text=False, batch_size=size,
                                                         **self.settings)
        return [out[0]["generated_text"] for out in outputs]


class RemoteLLM:
    """A model behind the Anthropic API; costs money per call. Batched, prompts go as one Message Batch (half price,
    usually done within the hour); otherwise one Messages request each."""

    POLL_SECONDS = 30
    batch_size = None   # any number of prompts can go in one batch
    batch = True        # by default: a batch costs half as much

    def __init__(self, model: str, max_tokens: int = 16):
        self.name = model   # e.g. "claude-haiku-4-5"
        self.settings = {"max_tokens": max_tokens}
        self._client = None

    def generate(self, prompts: list[str], prefix: str = "", batch: bool = False) -> list[str]:
        # one response per prompt. prefix: the start every prompt shares (e.g. a template's task, rules and examples),
        # sent as a cached system block so it's paid for in full once per hour, not on every call. It only engages
        # once the prefix is long enough for the model (e.g. 4096 tokens for Haiku 4.5)
        import anthropic
        self._client = self._client or anthropic.Anthropic()
        requests = [self._request(p, prefix) for p in prompts]
        try:
            if batch:
                return self._batch(requests)
            return [_text(self._client.messages.create(**r)) for r in requests]
        except (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError) as e:
            raise ServiceUnavailable(f"Anthropic API ({self.name}): {type(e).__name__}") from e

    def _request(self, prompt: str, prefix: str) -> dict:
        request = {"model": self.name, "max_tokens": self.settings["max_tokens"]}
        if prefix and prompt.startswith(prefix):
            request["system"] = [{"type": "text", "text": prefix, "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
            prompt = prompt[len(prefix):]
        return request | {"messages": [{"role": "user", "content": prompt}]}

    def _batch(self, requests: list[dict], retry: bool = True) -> list[str]:
        # one Message Batch; its id is saved (by the requests' hash) so a run stopped while waiting resumes it rather
        # than paying again. Requests that error or expire are sent once more in a new batch, then the job stops
        key = hashlib.sha1(json.dumps(requests, sort_keys=True).encode("utf-8")).hexdigest()
        saved = _load_batches()
        if key in saved:
            batch_id = saved[key]
            logger.info(f"resuming batch {batch_id} ({len(requests)} requests)")
        else:
            batch_id = self._client.messages.batches.create(
                requests=[{"custom_id": str(i), "params": r} for i, r in enumerate(requests)]).id
            _save_batch(key, batch_id)
            logger.info(f"submitted batch {batch_id} ({len(requests)} requests, {self.name})")
        while (status := self._client.messages.batches.retrieve(batch_id)).processing_status != "ended":
            counts = status.request_counts
            logger.info(f"batch {batch_id}: {counts.succeeded + counts.errored + counts.expired}/{len(requests)} done")
            time.sleep(self.POLL_SECONDS)
        texts, failed = [None] * len(requests), []
        for entry in self._client.messages.batches.results(batch_id):
            i = int(entry.custom_id)
            if entry.result.type == "succeeded":
                texts[i] = _text(entry.result.message)
            else:
                failed.append(i)
        failed += [i for i, t in enumerate(texts) if t is None and i not in failed]
        if failed:
            if not retry:
                raise ServiceUnavailable(f"Anthropic batch {batch_id}: {len(failed)} requests failed twice")
            logger.warning(f"batch {batch_id}: {len(failed)} requests failed; sending them again")
            for i, t in zip(failed, self._batch([requests[i] for i in failed], retry=False)):
                texts[i] = t
        return texts


def _text(message) -> str:
    return "".join(block.text for block in message.content if block.type == "text")


def _load_batches() -> dict[str, str]:
    path = Path(PATHS["service_cache"]) / "anthropic_batches.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _save_batch(key: str, batch_id: str) -> None:
    path = Path(PATHS["service_cache"]) / "anthropic_batches.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_load_batches() | {key: batch_id}, indent=1))
