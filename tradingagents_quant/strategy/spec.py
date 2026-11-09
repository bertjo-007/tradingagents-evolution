"""The strategy specification: everything a backtest needs to be reproduced.

A ``StrategySpec`` is frozen. Its ``version_hash`` covers every field, so any
change of rule, parameter, sizing or exit is a new version, stored next to the
results it produced. Nothing — no agent, no optimizer — edits a spec in place.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SizingSpec(_Frozen):
    """How a signal in [0, 1] becomes a target weight of equity.

    - ``fixed_fraction``: weight = signal x ``fraction``.
    - ``vol_target``: scale so the position's annualized volatility is
      ``target_vol``, capped at ``fraction``.
    - ``atr_risk``: size so that a move of ``atr_multiple`` ATRs against the
      position costs ``risk_per_trade`` of equity, capped at ``fraction``.
    """

    method: Literal["fixed_fraction", "vol_target", "atr_risk"] = "fixed_fraction"
    fraction: float = Field(default=1.0, gt=0, le=1.0)  # long only, no leverage
    target_vol: float = Field(default=0.15, gt=0, le=1.0)
    vol_window: int = Field(default=20, ge=5, le=252)
    risk_per_trade: float = Field(default=0.01, gt=0, le=0.1)
    atr_window: int = Field(default=14, ge=2, le=100)
    atr_multiple: float = Field(default=2.0, gt=0, le=20)


class ExitSpec(_Frozen):
    """Exits beyond the rule's own signal. Levels are fixed at entry."""

    stop_loss: Literal["none", "percent", "atr"] = "none"
    stop_value: float = Field(default=0.0, ge=0)  # fraction (percent) or ATR multiple
    take_profit: Literal["none", "percent", "atr"] = "none"
    take_value: float = Field(default=0.0, ge=0)
    atr_window: int = Field(default=14, ge=2, le=100)
    max_holding_bars: int | None = Field(default=None, ge=1)
    # After a stop or target exit, re-enter only once the signal has gone flat
    # and come back: otherwise a still-long signal buys back the next morning.
    reentry: Literal["after_signal_reset", "immediate"] = "after_signal_reset"

    @model_validator(mode="after")
    def _values_present(self) -> ExitSpec:
        for kind, value, name in (
            (self.stop_loss, self.stop_value, "stop_value"),
            (self.take_profit, self.take_value, "take_value"),
        ):
            if kind != "none" and value <= 0:
                raise ValueError(f"{name} must be positive when its exit is enabled")
            if kind == "percent" and value >= 1 and name == "stop_value":
                raise ValueError("a percent stop must be below 100 %")
        return self


class StrategySpec(_Frozen):
    name: str
    rule: str  # key in strategy.library.RULES
    params: dict[str, float | int] = Field(default_factory=dict)
    symbols: tuple[str, ...]
    timeframe: Literal["1d"] = "1d"
    direction: Literal["long_only"] = "long_only"
    # Signals on total-return prices (dividends reinvested, causal) or on
    # quoted prices. Fills always use quoted prices.
    signal_prices: Literal["total_return", "price"] = "total_return"
    sizing: SizingSpec = SizingSpec()
    exits: ExitSpec = ExitSpec()
    benchmark: str = "SPY"
    expected_holding_bars: tuple[int, int] | None = None
    notes: str = ""

    @field_validator("symbols", mode="before")
    @classmethod
    def _symbols(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.split(",")
        return tuple(str(s).strip().upper() for s in value if str(s).strip())

    @field_validator("symbols")
    @classmethod
    def _non_empty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("a strategy needs at least one symbol")
        return value

    @model_validator(mode="after")
    def _rule_known(self) -> StrategySpec:
        from tradingagents_quant.strategy.library import RULES

        if self.rule not in RULES:
            raise ValueError(f"unknown rule {self.rule!r}; known: {', '.join(sorted(RULES))}")
        RULES[self.rule].check_params(self.params)
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def version_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def resolved_params(self) -> dict[str, float | int]:
        from tradingagents_quant.strategy.library import RULES

        return {**RULES[self.rule].defaults, **self.params}

    def describe(self) -> dict[str, Any]:
        """The specification in words, as the brief asks for every strategy."""
        from tradingagents_quant.strategy.library import RULES

        rule = RULES[self.rule]
        params = self.resolved_params()
        return {
            "name": self.name,
            "version": self.version_hash(),
            "instruments": list(self.symbols),
            "timeframe": "daily bars; decision at the close, execution at the next open",
            "entry": rule.entry.format(**params),
            "exit": rule.exit.format(**params) + _exit_words(self.exits),
            "position_sizing": _sizing_words(self.sizing),
            "risk_limits": _exit_words(self.exits).lstrip("; ") or "none beyond the rule",
            "required_data": list(rule.required),
            "benchmark": self.benchmark,
            "expected_holding_bars": self.expected_holding_bars or rule.holding,
        }


def _exit_words(exits: ExitSpec) -> str:
    parts = []
    if exits.stop_loss == "percent":
        parts.append(f"stop loss {exits.stop_value:.1%} below entry")
    elif exits.stop_loss == "atr":
        parts.append(f"stop loss {exits.stop_value:g} x ATR({exits.atr_window}) below entry")
    if exits.take_profit == "percent":
        parts.append(f"take profit {exits.take_value:.1%} above entry")
    elif exits.take_profit == "atr":
        parts.append(f"take profit {exits.take_value:g} x ATR({exits.atr_window}) above entry")
    if exits.max_holding_bars:
        parts.append(f"exit after {exits.max_holding_bars} bars at the latest")
    return "".join(f"; {p}" for p in parts)


def _sizing_words(s: SizingSpec) -> str:
    if s.method == "fixed_fraction":
        return f"{s.fraction:.0%} of equity while the signal is on"
    if s.method == "vol_target":
        return f"volatility target {s.target_vol:.0%} annualized ({s.vol_window} bars), at most {s.fraction:.0%}"
    return (f"risk {s.risk_per_trade:.1%} of equity per {s.atr_multiple:g} x ATR({s.atr_window}), "
            f"at most {s.fraction:.0%}")
