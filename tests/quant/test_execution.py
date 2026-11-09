"""Fill prices and commissions, checked against hand arithmetic."""

from __future__ import annotations

import pytest

from tradingagents_quant.config import CostSettings
from tradingagents_quant.sim.execution import CostModel


def test_a_buy_pays_half_the_spread_plus_slippage_a_sell_receives_less():
    cm = CostModel(spread_bps=10, slippage_bps=5)  # adverse 5 + 5 = 10 bps
    assert cm.adverse == pytest.approx(0.001)
    assert cm.fill_price("buy", 100.0) == pytest.approx(100.1)
    assert cm.fill_price("sell", 100.0) == pytest.approx(99.9)
    with pytest.raises(ValueError):
        cm.fill_price("short", 100.0)


def test_commission_minimum_per_share_cap_and_percentage():
    cm = CostModel(commission_per_share=0.005, commission_min=1.0, commission_max_pct=0.01)
    assert cm.commission(100, 50.0) == 1.0  # 0.50 per share charge, the minimum applies
    assert cm.commission(1000, 50.0) == pytest.approx(5.0)  # 1000 x 0.005
    assert cm.commission(10, 5.0) == pytest.approx(0.5)  # capped at 1 % of 50
    assert cm.commission(0, 50.0) == 0.0
    pct = CostModel(commission_per_share=0, commission_min=0, commission_max_pct=0, commission_bps=10)
    assert pct.commission(100, 50.0) == pytest.approx(5.0)  # 10 bps of 5000


def test_affordable_never_spends_more_than_the_cash():
    cm = CostModel(commission_per_share=0.005, commission_min=1.0)
    shares = cm.affordable(1000.0, 10.0, wanted=500)
    assert shares == 99  # 100 shares would cost 1000 + 1 commission
    assert shares * 10.0 + cm.commission(shares, 10.0) <= 1000.0
    assert cm.affordable(1000.0, 10.0, wanted=20) == 20
    assert cm.affordable(0.5, 10.0, wanted=1) == 0


def test_fractional_shares_when_allowed():
    cm = CostModel.free()
    assert cm.round_shares(3.7) == 3.0
    frac = CostModel(0, 0, 0, 0, 0, 0, fractional_shares=True)
    assert frac.round_shares(3.7) == pytest.approx(3.7)
    assert frac.affordable(100.0, 30.0, wanted=10) == pytest.approx(3.333333, abs=1e-6)


def test_the_model_is_built_from_the_configuration():
    s = CostSettings(commission_per_share=0.01, spread_bps=4, slippage_bps=1)
    cm = CostModel.from_settings(s)
    assert cm.commission_per_share == 0.01 and cm.adverse == pytest.approx(0.0003)
