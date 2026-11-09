"""Paper trading: the backtest engine stepped forward one trading day at a
time on simulated money, with the risk engine on every buy, and an
append-only, hash-chained ledger of everything that happens.

PAPER TRADE — NOT REAL MONEY. There is no broker connection and no code path
to one.
"""
