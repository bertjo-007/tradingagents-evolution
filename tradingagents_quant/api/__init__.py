"""The research dashboard: a read-only HTTP view of the system's own state.

Needs the optional extra: ``pip install "tradingagents[dashboard]"``. Every
request opens the database read-only; there is no endpoint that writes,
places an order or changes a setting.
"""
