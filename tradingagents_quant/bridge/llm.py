"""Record and replay model answers; count and cap what a run may spend.

**Cache** (langchain's global LLM cache, stored in the research database):

- ``record`` — an answer already recorded for the same model settings and
  prompt is served from the cache; every new answer is stored.
- ``replay`` — answers come from the cache only. A prompt without a recorded
  answer stops the run (``ReplayMiss``): a replay that quietly asks the model
  after all is not a replay. Replays cost nothing and are deterministic, so an
  agent evaluation can be rerun in tests and CI.
- ``off`` — no cache.

The key is a SHA-256 over langchain's ``llm_string`` (model and parameters)
and the serialized prompt. Entries are append-only.

**Budget**: a callback that counts the tokens of every model call that was not
served from the cache, per run and per day, persisted in ``llm_calls``. When a
run or the day has used its allowance (``[agents]`` in the configuration), the
next call is refused with ``BudgetExceeded`` — the run stops, the process
does not.

Everything from langchain is imported inside functions (see the package
docstring of ``tradingagents_quant``).
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tradingagents_quant.errors import QuantError
from tradingagents_quant.store.db import Database, utc_now

_local = threading.local()  # whether the current thread's last lookup was a hit
_lock = threading.Lock()


class ReplayMiss(QuantError):
    """Replay mode met a prompt with no recorded answer."""


class BudgetExceeded(QuantError):
    """The run or the day has used its token allowance."""


# Message fields that say how an answer was produced, not what was said. A
# cached answer comes back with them changed (langchain marks it
# ``usage_metadata: {total_cost: 0}``), and in an analyst's tool loop that
# answer becomes part of the next prompt: keyed on the raw prompt, every replay
# would miss from its first hit on.
_VOLATILE = ("id", "usage_metadata", "response_metadata")


def _strip(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("lc") == 1 and isinstance(value.get("kwargs"), dict):
            kwargs = {k: _strip(v) for k, v in value["kwargs"].items() if k not in _VOLATILE}
            return {**{k: _strip(v) for k, v in value.items() if k != "kwargs"}, "kwargs": kwargs}
        return {k: _strip(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def canonical_prompt(prompt: str) -> str:
    """The conversation without per-call metadata; the prompt itself if it is
    not langchain-serialized JSON."""
    try:
        parsed = json.loads(prompt)
    except (TypeError, ValueError):
        return prompt
    return json.dumps(_strip(parsed), sort_keys=True, separators=(",", ":"))


def cache_key(prompt: str, llm_string: str) -> str:
    return hashlib.sha256(f"{llm_string}\x00{canonical_prompt(prompt)}".encode()).hexdigest()


def _last_lookup_was_hit() -> bool:
    return bool(getattr(_local, "hit", False))


def make_cache(db_path: Path, mode: str):
    """A langchain ``BaseCache`` over ``llm_cache``, in ``record`` or ``replay`` mode."""
    from langchain_core.caches import BaseCache
    from langchain_core.load import dumps, loads

    if mode not in ("record", "replay"):
        raise ValueError(f"cache mode must be record or replay, not {mode!r}")

    class SqliteLLMCache(BaseCache):
        def __init__(self) -> None:
            self.mode = mode
            self.path = Path(db_path)
            self.hits = 0
            self.misses = 0
            self.stored = 0

        def lookup(self, prompt: str, llm_string: str) -> Sequence[Any] | None:
            key = cache_key(prompt, llm_string)
            with _lock:
                db = Database(self.path)
                try:
                    row = db.query_one("SELECT value_json FROM llm_cache WHERE cache_key = ?", (key,))
                finally:
                    db.close()
            _local.hit = row is not None
            if row is not None:
                self.hits += 1
                return _decode(row[0], loads)
            self.misses += 1
            if self.mode == "replay":
                raise ReplayMiss(
                    f"no recorded answer for this prompt (key {key[:12]}); record the run first "
                    "or switch [agents].llm_cache to record")
            return None

        def update(self, prompt: str, llm_string: str, return_val: Sequence[Any]) -> None:
            if self.mode != "record":
                return
            key = cache_key(prompt, llm_string)
            value = json.dumps([dumps(g) for g in return_val])
            with _lock:
                db = Database(self.path)
                try:
                    db.execute(
                        "INSERT OR IGNORE INTO llm_cache (cache_key, created_at, llm_string, "
                        "prompt_sha256, value_json) VALUES (?, ?, ?, ?, ?)",
                        (key, utc_now(), llm_string,
                         hashlib.sha256(canonical_prompt(prompt).encode()).hexdigest(), value))
                finally:
                    db.close()
            self.stored += 1

        def clear(self, **kwargs: Any) -> None:
            raise QuantError("the recorded answers are append-only and are not cleared")

    return SqliteLLMCache()


def _decode(value_json: str, loads) -> list[Any]:
    """The stored generations: a JSON list of langchain-serialized items."""
    return [loads(item) for item in json.loads(value_json)]


def make_budget(db_path: Path, *, run_id: str, max_run: int, max_day: int,
                enforce: bool = True):
    """A langchain callback that counts tokens and stops a run over its allowance."""
    from langchain_core.callbacks import BaseCallbackHandler

    class TokenBudget(BaseCallbackHandler):
        raise_error = True  # a refusal must stop the run, not be logged and ignored

        def __init__(self) -> None:
            self.path = Path(db_path)
            self.run_id = run_id
            self.max_run, self.max_day = max_run, max_day
            self.enforce = enforce
            self.run_tokens = 0
            self.calls = 0
            self.cached_calls = 0
            self.day = datetime.now(UTC).strftime("%Y-%m-%d")
            with _lock:
                db = Database(self.path)
                try:
                    row = db.query_one("SELECT COALESCE(SUM(input_tokens + output_tokens), 0) "
                                       "FROM llm_calls WHERE day = ? AND cached = 0", (self.day,))
                finally:
                    db.close()
            self.day_tokens = int(row[0])

        def _check(self) -> None:
            if not self.enforce:
                return
            if self.run_tokens >= self.max_run:
                raise BudgetExceeded(f"this run has used {self.run_tokens:,} tokens, its "
                                     f"allowance is {self.max_run:,}")
            if self.day_tokens >= self.max_day:
                raise BudgetExceeded(f"today's model calls have used {self.day_tokens:,} tokens, "
                                     f"the daily allowance is {self.max_day:,}")

        def on_chat_model_start(self, serialized, messages, **kwargs: Any) -> None:
            # The cache lookup comes after this; until it marks a hit, this
            # call is a real one. (Without the reset, a hit from an earlier
            # call would mark every later uncached call as free.)
            _local.hit = False
            self._check()

        def on_llm_start(self, serialized, prompts, **kwargs: Any) -> None:
            _local.hit = False
            self._check()

        def on_llm_end(self, response, **kwargs: Any) -> None:
            cached = _last_lookup_was_hit()
            _local.hit = False
            used_in, used_out, model = _usage(response)
            self.calls += 1
            if cached:
                self.cached_calls += 1
            else:
                self.run_tokens += used_in + used_out
                self.day_tokens += used_in + used_out
            with _lock:
                db = Database(self.path)
                try:
                    db.execute(
                        "INSERT INTO llm_calls (ts, day, run_id, model, input_tokens, "
                        "output_tokens, cached) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (utc_now(), self.day, self.run_id, model, used_in, used_out, int(cached)))
                finally:
                    db.close()

    return TokenBudget()


def _usage(response: Any) -> tuple[int, int, str | None]:
    """Input and output tokens and the model name of an ``LLMResult``."""
    used_in = used_out = 0
    model = None
    for generations in getattr(response, "generations", []) or []:
        for g in generations:
            message = getattr(g, "message", None)
            meta = getattr(message, "usage_metadata", None) or {}
            used_in += int(meta.get("input_tokens", 0) or 0)
            used_out += int(meta.get("output_tokens", 0) or 0)
            info = getattr(message, "response_metadata", None) or {}
            model = model or info.get("model_name") or info.get("model")
    if used_in == used_out == 0:
        usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
        used_in = int(usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0)
        used_out = int(usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0)
    return used_in, used_out, model


def install(db_path: Path, *, mode: str, run_id: str, max_run: int, max_day: int):
    """Set the global cache (unless ``off``) and return (cache, budget)."""
    from langchain_core.globals import set_llm_cache

    db = Database(db_path)
    db.migrate()
    db.close()
    cache = None
    if mode != "off":
        cache = make_cache(db_path, mode)
        set_llm_cache(cache)
    else:
        set_llm_cache(None)
    budget = make_budget(db_path, run_id=run_id, max_run=max_run, max_day=max_day,
                         enforce=mode != "replay")
    return cache, budget


def uninstall() -> None:
    from langchain_core.globals import set_llm_cache

    set_llm_cache(None)
