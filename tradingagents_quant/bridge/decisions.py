"""Read the three structured decisions back out of the agent state.

The agent framework keeps only the *rendered* markdown of its structured
outputs in state (``render_research_plan``, ``render_trader_proposal``,
``render_pm_decision`` in ``tradingagents/agents/schemas.py``). Those renders
have fixed field lines. A stage whose text lacks them did not produce a
structured answer — ``agents/structured.py`` fell back to free prose — and
the gate treats that as a failure: a number that came from prose is not one
the system can check.

``tests/quant/test_gate.py`` round-trips the upstream render functions through
this parser, so a change of the render format breaks a test, not the gate.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

FIELDS = {
    "investment_plan": ("Recommendation", "Rationale", "Strategic Actions"),
    "trader_investment_plan": ("Action", "Reasoning", "Entry Price", "Stop Loss",
                               "Position Sizing"),
    "final_trade_decision": ("Rating", "Executive Summary", "Investment Thesis", "Price Target",
                             "Time Horizon"),
}
STAGE_NAMES = {"investment_plan": "Research Manager", "trader_investment_plan": "Trader",
               "final_trade_decision": "Portfolio Manager"}

# Direction of each rating, for the consistency check.
RATING_DIRECTION = {"Buy": 1, "Overweight": 1, "Hold": 0, "Underweight": -1, "Sell": -1}
ACTION_DIRECTION = {"Buy": 1, "Hold": 0, "Sell": -1}


def field_value(text: str, label: str) -> str | None:
    """The value on the line ``**<label>**: <value>``, or None."""
    match = re.search(rf"(?m)^\*\*{re.escape(label)}\*\*: ?(.*)$", text or "")
    return match.group(1).strip() if match else None


def number(text: str | None) -> float | None:
    """A rendered optional float: the number, or None for 'not provided'."""
    if text is None or text == "" or text.lower() in ("not provided", "none", "n/a"):
        return None
    try:
        value = float(text.replace(",", ""))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


@dataclass
class Decisions:
    recommendation: str | None = None
    action: str | None = None
    entry_price: float | None = None
    stop_loss: float | None = None
    rating: str | None = None
    price_target: float | None = None
    time_horizon: str | None = None
    structured: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read(state: dict[str, Any]) -> Decisions:
    d = Decisions()
    for key, labels in FIELDS.items():
        text = state.get(key) or ""
        d.structured[key] = bool(text) and all(field_value(text, lb) is not None for lb in labels)
    plan, trader, pm = (state.get(k) or "" for k in FIELDS)
    d.recommendation = field_value(plan, "Recommendation")
    d.action = field_value(trader, "Action")
    d.entry_price = number(field_value(trader, "Entry Price"))
    d.stop_loss = number(field_value(trader, "Stop Loss"))
    d.rating = state.get("final_rating") or field_value(pm, "Rating")
    d.price_target = number(field_value(pm, "Price Target"))
    d.time_horizon = field_value(pm, "Time Horizon")
    return d
