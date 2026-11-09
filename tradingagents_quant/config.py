"""Configuration of the research core: defaults, an optional TOML file, then
``TAQ_<SECTION>_<FIELD>`` variables, validated once at startup.

The configuration carries no secrets. LLM keys stay where the agent framework
reads them (``GOOGLE_API_KEY`` and friends) and never pass through here, so a
configuration can be printed, hashed and stored with every run.

Live trading cannot be configured: the two live fields are typed as constants,
and ``assert_no_live_trading`` refuses any variable that asks for it.
"""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from tradingagents_quant import paths
from tradingagents_quant.errors import ConfigError, LiveTradingForbiddenError

# Broad, sector and defensive ETFs with history back to 1998-2004. A default of
# today's best-known single stocks would bake survivorship bias into every
# backtest: they are known because they survived and grew.
US_CORE = (
    "SPY", "QQQ", "IWM", "DIA",
    "TLT", "IEF", "GLD",
    "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
)

# Every variable name that could be read as a request for live trading, with or
# without this package's prefix. Each must be unset or say "off".
_LIVE_VARIABLES = (
    "LIVE_TRADING_MODE",
    "LIVE_ORDER_EXECUTION",
    "TAQ_LIVE_TRADING_MODE",
    "TAQ_LIVE_ORDER_EXECUTION",
    "TAQ_MODES_LIVE_TRADING_MODE",
    "TAQ_MODES_LIVE_ORDER_EXECUTION",
)
_OFF_VALUES = frozenset({"", "0", "false", "no", "off", "disabled"})
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Modes(_Section):
    research_mode: bool = True
    backtesting_mode: bool = True
    paper_trading_mode: bool = True
    # Constants, not switches: validation rejects any other value.
    live_trading_mode: Literal[False] = False
    live_order_execution: Literal["disabled"] = "disabled"


class Universe(_Section):
    name: str = "us-core"
    symbols: tuple[str, ...] = US_CORE
    benchmark: str = "SPY"
    # Trading days are the dates on which this symbol has a bar: a calendar with
    # no dependency, and point-in-time by construction.
    calendar_symbol: str = "SPY"
    currency: str = "USD"

    @field_validator("symbols", mode="before")
    @classmethod
    def _normalize_symbols(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.split(",")
        if isinstance(value, list | tuple):
            value = tuple(str(s).strip().upper() for s in value if str(s).strip())
        return value

    @field_validator("symbols")
    @classmethod
    def _symbols_present_and_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("the universe needs at least one symbol")
        duplicates = sorted({s for s in value if value.count(s) > 1})
        if duplicates:
            raise ValueError(f"symbols listed twice: {', '.join(duplicates)}")
        return value

    @field_validator("benchmark", "calendar_symbol", "currency")
    @classmethod
    def _upper(cls, value: str) -> str:
        value = value.strip().upper()
        if not value:
            raise ValueError("must not be empty")
        return value

    def ingest_symbols(self) -> tuple[str, ...]:
        """Symbols whose history a backtest needs: the universe, its benchmark
        and its calendar, each once."""
        extra = [s for s in (self.benchmark, self.calendar_symbol) if s not in self.symbols]
        return self.symbols + tuple(dict.fromkeys(extra))


class DataSettings(_Section):
    # Earliest bar requested at ingest; each symbol starts at its own first bar.
    history_start: str = "1999-01-01"
    # A latest bar this many calendar days before the date asked for is stale.
    max_stale_days: int = Field(default=10, ge=1, le=60)
    # Fewer bars than this and annualised figures are not reported.
    min_annualization_bars: int = Field(default=252, ge=20)

    @field_validator("history_start")
    @classmethod
    def _iso_date(cls, value: str) -> str:
        from datetime import date

        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"not a YYYY-MM-DD date: {value!r}") from exc
        return value


class CostSettings(_Section):
    """Execution costs of a simulated fill. Defaults: a liquid US ETF through a
    per-share broker (0.005 USD a share, at least 1 USD, at most 1 % of the
    trade), plus spread and slippage. Deliberately not optimistic: SPY's real
    spread is a fraction of a basis point."""

    commission_per_share: float = Field(default=0.005, ge=0)
    commission_min: float = Field(default=1.0, ge=0)
    commission_max_pct: float = Field(default=0.01, ge=0, le=0.1)
    commission_bps: float = Field(default=0.0, ge=0, le=100)  # percentage brokers
    spread_bps: float = Field(default=2.0, ge=0, le=500)  # full spread; a fill pays half
    slippage_bps: float = Field(default=2.0, ge=0, le=500)  # adverse move per fill
    fractional_shares: bool = False


class BacktestSettings(_Section):
    initial_cash: float = Field(default=100_000.0, gt=0)
    risk_free_rate: float = Field(default=0.0, ge=0, le=0.2)  # annual, for Sharpe/Sortino
    # A position is resized only when the target differs from it by more than
    # this fraction; below it, the trade would cost more than it corrects.
    rebalance_band: float = Field(default=0.2, ge=0, le=1)


class RiskSettings(_Section):
    """Limits the deterministic risk engine enforces on every simulated order.

    Read from the configuration only. Nothing at run time — no strategy, no
    research loop, no agent — can change them; changing them means changing
    the configuration, which changes its fingerprint and is recorded.
    """

    max_gross_exposure: float = Field(default=1.0, gt=0, le=1.0)  # all positions / equity
    max_position_weight: float = Field(default=0.35, gt=0, le=1.0)  # one position / equity
    max_open_positions: int = Field(default=20, ge=1)
    max_drawdown: float = Field(default=0.25, gt=0, le=1.0)  # from the peak: no new entries
    max_daily_loss: float = Field(default=0.04, gt=0, le=1.0)  # day's loss: no new entries next day
    max_volatility: float = Field(default=0.80, gt=0)  # annualized, 20 bars, of the instrument
    min_dollar_volume: float = Field(default=5_000_000.0, ge=0)  # 20-bar average traded value
    max_adv_fraction: float = Field(default=0.01, gt=0, le=1.0)  # order / average daily value
    max_correlated_weight: float = Field(default=0.70, gt=0, le=1.0)
    correlation_threshold: float = Field(default=0.90, gt=0, le=1.0)  # 60-bar return correlation
    correlation_window: int = Field(default=60, ge=20, le=504)
    require_stop: bool = False  # True: an entry without a stop level is refused
    max_risk_per_trade: float = Field(default=0.02, gt=0, le=1.0)  # loss at the stop / equity


class QuantConfig(_Section):
    modes: Modes = Modes()
    universe: Universe = Universe()
    data: DataSettings = DataSettings()
    costs: CostSettings = CostSettings()
    backtest: BacktestSettings = BacktestSettings()
    risk: RiskSettings = RiskSettings()

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def fingerprint(self) -> str:
        """A short, stable hash of every setting, stored with each run so a
        result can be traced to the configuration that produced it."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


_SECTIONS: dict[str, type[_Section]] = {
    "modes": Modes,
    "universe": Universe,
    "data": DataSettings,
    "costs": CostSettings,
    "backtest": BacktestSettings,
    "risk": RiskSettings,
}


def assert_no_live_trading(environ: Mapping[str, str]) -> None:
    for name in _LIVE_VARIABLES:
        raw = environ.get(name)
        if raw is not None and raw.strip().lower() not in _OFF_VALUES:
            raise LiveTradingForbiddenError(
                f"{name}={raw!r} asks for live trading. This system runs research, "
                "backtesting and paper trading only; unset the variable."
            )


def _coerce(name: str, raw: str, annotation: Any) -> Any:
    """Parse one variable for a field of the given type; raise on a bad value
    rather than fall back to the default (a typo must not pass silently)."""
    text = raw.strip()
    origin = typing.get_origin(annotation)
    if annotation is bool:
        lowered = text.lower()
        if lowered in _TRUE_VALUES:
            return True
        if lowered in _FALSE_VALUES:
            return False
        raise ConfigError(f"{name}={raw!r} is not a boolean (true/false)")
    if annotation is int:
        try:
            return int(text)
        except ValueError as exc:
            raise ConfigError(f"{name}={raw!r} is not an integer") from exc
    if annotation is float:
        try:
            return float(text)
        except ValueError as exc:
            raise ConfigError(f"{name}={raw!r} is not a number") from exc
    if origin is tuple:
        return [part.strip() for part in text.split(",") if part.strip()]
    return text


def _env_overrides(environ: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    for section, model in _SECTIONS.items():
        hints = typing.get_type_hints(model)
        for field in model.model_fields:
            if field.startswith("live_"):
                continue  # constants; assert_no_live_trading has judged the variable
            name = f"TAQ_{section}_{field}".upper()
            if name in environ and environ[name].strip() != "":
                overrides.setdefault(section, {})[field] = _coerce(
                    name, environ[name], hints[field]
                )
    return overrides


def _read_file(path: Path) -> dict[str, Any]:
    try:
        with open(path, "rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"configuration file {path} is not valid TOML: {exc}") from exc


def _merge(base: dict[str, Any], top: dict[str, dict[str, Any]]) -> dict[str, Any]:
    merged = {k: dict(v) if isinstance(v, dict) else v for k, v in base.items()}
    for section, fields in top.items():
        merged.setdefault(section, {})
        if not isinstance(merged[section], dict):
            raise ConfigError(f"[{section}] must be a table")
        merged[section].update(fields)
    return merged


def config_file(environ: Mapping[str, str], home: Path) -> Path | None:
    """The file to read: ``TAQ_CONFIG`` if set, else ``taq.toml`` in the data
    directory if present, else none (defaults only)."""
    raw = environ.get("TAQ_CONFIG", "").strip()
    if raw:
        return Path(raw).expanduser()
    candidate = home / "taq.toml"
    return candidate if candidate.is_file() else None


def load_config(
    path: Path | None = None, environ: Mapping[str, str] | None = None
) -> QuantConfig:
    """Build and validate the configuration. Raises ``ConfigError`` (or its
    subclass ``LiveTradingForbiddenError``) with a readable message."""
    environ = os.environ if environ is None else environ
    assert_no_live_trading(environ)
    home = paths.data_home(environ)
    source = path if path is not None else config_file(environ, home)
    raw = _read_file(source) if source is not None else {}
    unknown = sorted(set(raw) - set(_SECTIONS))
    if unknown:
        raise ConfigError(f"unknown configuration section(s): {', '.join(unknown)}")
    merged = _merge(raw, _env_overrides(environ))
    try:
        return QuantConfig.model_validate(merged)
    except ValidationError as exc:
        live = [e for e in exc.errors() if str(e["loc"][-1]).startswith("live_")]
        if live:
            raise LiveTradingForbiddenError(
                "the configuration asks for live trading "
                f"({'.'.join(map(str, live[0]['loc']))}); this system runs "
                "research, backtesting and paper trading only."
            ) from exc
        raise ConfigError(f"invalid configuration: {exc}") from exc
