"""Yahoo Finance as a snapshot source, built from the hardened upstream pieces.

Not ``load_ohlcv``: it keeps five years up to today and only adjusted prices
(``auto_adjust=True, actions=False``). A snapshot needs the full history and the
dividends and splits, so a later restatement can be seen. What is reused:
``yf_retry`` (rate-limit backoff, empty-vs-outage), ``raise_for_empty``
(probes the host before calling a symbol empty), ``normalize_symbol`` and
``safe_ticker_component``.

Every upstream import is inside a function: ``tradingagents`` loads a ``.env``
when imported, which must not happen when the research core is merely imported.
"""

from __future__ import annotations

from datetime import date, timedelta

from tradingagents_quant.data.ingest import RawFetch
from tradingagents_quant.errors import DataUnavailableError, NoDataError

SOURCE = "yahoo"


def source_version() -> str:
    import yfinance

    return f"yfinance {yfinance.__version__}"


def fetch_yahoo(symbol: str, start: str, end: str | None) -> RawFetch:
    """Daily bars ``start``..``end`` (inclusive) with dividends and splits."""
    import yfinance as yf

    from tradingagents.dataflows.errors import NoMarketDataError, VendorError
    from tradingagents.dataflows.symbols import normalize_symbol, safe_ticker_component
    from tradingagents.dataflows.vendors.yahoo.common import raise_for_empty, yf_retry

    canonical = normalize_symbol(symbol)
    try:
        safe_ticker_component(canonical)
    except ValueError as exc:
        raise NoDataError(f"{symbol!r} is not a valid symbol: {exc}") from exc
    # yfinance's end is exclusive.
    end_exclusive = (date.fromisoformat(end) + timedelta(days=1)).isoformat() if end else None
    ticker = yf.Ticker(canonical)
    try:
        frame = yf_retry(
            lambda: ticker.history(start=start, end=end_exclusive, auto_adjust=False, actions=True)
        )
        if frame is None or frame.empty:
            raise_for_empty(symbol, canonical, "price rows")
    except NoMarketDataError as exc:
        raise NoDataError(str(exc)) from exc
    except VendorError as exc:
        raise DataUnavailableError(str(exc)) from exc
    # yfinance 1.8 returns a HistoryMetadata object here, not a dict: read it
    # through .get() whatever its type.
    getter = getattr(getattr(ticker, "history_metadata", None), "get", None)
    currency = getter("currency") if callable(getter) else None
    return RawFetch(canonical, frame, currency if isinstance(currency, str) else None)
