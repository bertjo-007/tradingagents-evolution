"""The Yahoo adapter, offline: yfinance.Ticker is replaced, the network stays blocked."""

from __future__ import annotations

import pandas as pd
import pytest

from tradingagents_quant.data import ingest_yahoo
from tradingagents_quant.errors import DataUnavailableError, NoDataError

from .fixtures import linear_closes, trading_days, yahoo_frame

DAYS = trading_days("2024-01-02", 5)


class _HistoryMetadata:
    """Like yfinance 1.8's HistoryMetadata: has .get(), is not a dict."""

    def __init__(self, values):
        self._values = values

    def get(self, key, default=None):
        return self._values.get(key, default)


class _FakeTicker:
    frame = yahoo_frame(DAYS, linear_closes(5))
    error: Exception | None = None
    seen: list = []

    def __init__(self, symbol):
        self.symbol = symbol
        self.history_metadata = _HistoryMetadata({"currency": "USD"})

    def history(self, **kwargs):
        _FakeTicker.seen.append((self.symbol, kwargs))
        if _FakeTicker.error is not None:
            raise _FakeTicker.error
        return _FakeTicker.frame


@pytest.fixture
def fake_yahoo(monkeypatch):
    import yfinance

    monkeypatch.setattr(yfinance, "Ticker", _FakeTicker)
    _FakeTicker.seen = []
    _FakeTicker.error = None
    _FakeTicker.frame = yahoo_frame(DAYS, linear_closes(5))
    return _FakeTicker


def test_full_history_with_actions_and_an_inclusive_end(fake_yahoo):
    raw = ingest_yahoo.fetch_yahoo("spy", "2024-01-02", "2024-01-08")
    assert raw.canonical == "SPY" and raw.currency == "USD"
    assert raw.frame is fake_yahoo.frame
    symbol, kwargs = fake_yahoo.seen[0]
    assert symbol == "SPY"
    assert kwargs == {"start": "2024-01-02", "end": "2024-01-09",  # yfinance's end is exclusive
                      "auto_adjust": False, "actions": True}


def test_a_failed_request_is_unavailable_not_empty(fake_yahoo):
    fake_yahoo.error = RuntimeError("connection reset")
    with pytest.raises(DataUnavailableError):
        ingest_yahoo.fetch_yahoo("SPY", "2024-01-02", None)


def test_an_empty_answer_from_an_unreachable_host_is_unavailable(fake_yahoo):
    # The network is blocked in tests, so the reachability probe fails: an empty
    # answer then says nothing about the symbol.
    fake_yahoo.frame = pd.DataFrame()
    with pytest.raises(DataUnavailableError):
        ingest_yahoo.fetch_yahoo("SPY", "2024-01-02", None)


def test_an_empty_answer_from_a_reachable_host_is_no_data(fake_yahoo, monkeypatch):
    import tradingagents.dataflows.vendors.yahoo.common as common

    monkeypatch.setattr(common, "vendor_reachable", lambda url, timeout=5.0: True)
    fake_yahoo.frame = pd.DataFrame()
    with pytest.raises(NoDataError):
        ingest_yahoo.fetch_yahoo("SPY", "2024-01-02", None)


def test_a_path_like_symbol_is_refused_before_any_request(fake_yahoo):
    with pytest.raises(NoDataError, match="not a valid symbol"):
        ingest_yahoo.fetch_yahoo("../../etc", "2024-01-02", None)
    assert fake_yahoo.seen == []
