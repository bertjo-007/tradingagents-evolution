"""Record/replay of model answers and the token budget, with a counting fake model."""

from __future__ import annotations

import sqlite3

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from tradingagents_quant.bridge import llm
from tradingagents_quant.errors import QuantError
from tradingagents_quant.store.db import Database

CALLS: list[str] = []  # outside the model: a field would change its cache key


class Counting(BaseChatModel):
    answer: str = "noted"
    tool: bool = False

    @property
    def _llm_type(self) -> str:
        return "counting"

    @property
    def _identifying_params(self) -> dict:
        # What real models report here (model name, temperature, ...) is
        # what langchain keys the cache on.
        return {"answer": self.answer, "tool": self.tool}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        CALLS.append(messages[-1].content)
        usage = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
        if self.tool:
            message = AIMessage(content="", tool_calls=[{"name": "get_news", "id": "c1",
                                                         "args": {"query": "rates"}}],
                                usage_metadata=usage)
        else:
            message = AIMessage(content=f"{self.answer}: {messages[-1].content}",
                                usage_metadata=usage)
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.fixture(autouse=True)
def _clean():
    CALLS.clear()
    yield
    llm.uninstall()  # the cache is global: never leak it into other tests


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "taq.sqlite3"
    Database(path).migrate()
    return path


def _ask(model, text, budget=None):
    config = {"callbacks": [budget]} if budget else {}
    return model.invoke([HumanMessage(content=text)], config=config)


def test_record_answers_once_and_serves_the_repeat(db_path):
    cache, _ = llm.install(db_path, mode="record", run_id="r1", max_run=10**9, max_day=10**9)
    model = Counting()
    first, second = _ask(model, "what now?"), _ask(model, "what now?")
    assert CALLS == ["what now?"] and first.content == second.content == "noted: what now?"
    assert (cache.hits, cache.misses, cache.stored) == (1, 1, 1)
    _ask(model, "and then?")
    assert len(CALLS) == 2  # another prompt is another answer


def test_replay_serves_the_recording_and_stops_at_a_gap(db_path):
    llm.install(db_path, mode="record", run_id="r1", max_run=10**9, max_day=10**9)
    recorded = _ask(Counting(), "what now?")
    llm.install(db_path, mode="replay", run_id="r2", max_run=10**9, max_day=10**9)
    CALLS.clear()
    replayed = _ask(Counting(), "what now?")
    assert CALLS == [] and replayed.content == recorded.content
    with pytest.raises(llm.ReplayMiss):
        _ask(Counting(), "something never asked")
    with pytest.raises(llm.ReplayMiss):
        _ask(Counting(answer="different settings"), "what now?")  # other model settings: other key


def test_a_tool_call_answer_survives_the_round_trip(db_path):
    llm.install(db_path, mode="record", run_id="r1", max_run=10**9, max_day=10**9)
    first = _ask(Counting(tool=True), "look it up")
    llm.install(db_path, mode="replay", run_id="r2", max_run=10**9, max_day=10**9)
    again = _ask(Counting(tool=True), "look it up")
    assert again.tool_calls == first.tool_calls and again.tool_calls[0]["name"] == "get_news"


def test_an_earlier_cache_hit_does_not_make_later_calls_free(db_path):
    """Found while testing: the hit marker of one call leaked into the next."""
    _, budget = llm.install(db_path, mode="record", run_id="r0", max_run=10**9, max_day=10**9)
    _ask(Counting(), "q", budget)
    _ask(Counting(), "q", budget)  # a hit
    _, uncached = llm.install(db_path, mode="off", run_id="r1", max_run=10**9, max_day=10**9)
    _ask(Counting(), "fresh", uncached)
    assert uncached.run_tokens == 120 and uncached.cached_calls == 0


def test_the_run_budget_stops_the_next_call(db_path):
    _, budget = llm.install(db_path, mode="off", run_id="r1", max_run=200, max_day=10**9)
    model = Counting()
    _ask(model, "one", budget)
    _ask(model, "two", budget)  # 120 used before it: still under 200
    with pytest.raises(llm.BudgetExceeded, match="this run has used 240 tokens"):
        _ask(model, "three", budget)
    assert CALLS == ["one", "two"]


def test_cached_answers_cost_nothing(db_path):
    _, budget = llm.install(db_path, mode="record", run_id="r1", max_run=150, max_day=10**9)
    model = Counting()
    for _ in range(5):
        _ask(model, "same question", budget)
    assert budget.run_tokens == 120 and budget.cached_calls == 4 and budget.calls == 5


def test_the_daily_allowance_carries_across_runs(db_path):
    _, first = llm.install(db_path, mode="off", run_id="r1", max_run=10**9, max_day=100)
    _ask(Counting(), "spend", first)  # 120 of a 100-token day
    _, second = llm.install(db_path, mode="off", run_id="r2", max_run=10**9, max_day=100)
    assert second.day_tokens == 120
    with pytest.raises(llm.BudgetExceeded, match="today's model calls"):
        _ask(Counting(), "more", second)
    rows = Database(db_path).query("SELECT run_id, input_tokens, output_tokens, cached FROM llm_calls")
    assert [tuple(r) for r in rows] == [("r1", 100, 20, 0)]


def test_a_replay_is_not_held_back_by_the_budget(db_path):
    llm.install(db_path, mode="record", run_id="r1", max_run=10**9, max_day=10**9)
    _ask(Counting(), "q")
    _, budget = llm.install(db_path, mode="replay", run_id="r2", max_run=1, max_day=1)
    assert _ask(Counting(), "q", budget).content == "noted: q"


def test_the_key_ignores_how_an_answer_was_produced_but_not_what_it_says():
    """Found by the CLI replay test: a cached answer returns with changed usage
    metadata, enters the next prompt of a tool loop, and broke every replay."""
    from langchain_core.load import dumps

    def prompt(content, **meta):
        return dumps([HumanMessage(content="q"), AIMessage(content=content, **meta)])

    plain = llm.cache_key(prompt("a"), "m")
    assert llm.cache_key(prompt("a", usage_metadata={"input_tokens": 0, "output_tokens": 0,
                                                     "total_tokens": 0}), "m") == plain
    assert llm.cache_key(prompt("a", id="run-123", response_metadata={"x": 1}), "m") == plain
    assert llm.cache_key(prompt("b"), "m") != plain
    assert llm.cache_key(prompt("a"), "other model") != plain
    assert llm.canonical_prompt("not json") == "not json"


def test_recordings_are_kept(db_path):
    cache, _ = llm.install(db_path, mode="record", run_id="r1", max_run=10**9, max_day=10**9)
    _ask(Counting(), "q")
    with pytest.raises(QuantError, match="append-only"):
        cache.clear()
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        Database(db_path).execute("DELETE FROM llm_cache")
