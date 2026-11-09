"""Technical indicators as plain, causal pandas computations.

Every value at bar ``t`` depends only on bars ``0..t``; ``tests/quant/
test_indicators.py`` proves it for each indicator by truncation. A value that
cannot be computed yet (warm-up) is NaN, never a partial-window estimate.
"""
