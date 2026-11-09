"""tradingagents_quant: the deterministic research core beside the agent framework.

Research, backtesting and paper trading only. Nothing in this package places a
real order, and no setting turns that on (see ``config.Modes``).

Import boundary: no module here imports ``tradingagents``, ``yfinance``,
``langchain*`` or ``langgraph`` at module level. ``tradingagents/__init__.py``
loads a ``.env`` on import and mutates ``os.environ``; a simulation must not be
steered by whichever ``.env`` lies above the working directory. The few modules
that need the agent framework or a data vendor import it inside a function.
``tests/quant/test_boundary.py`` enforces this.
"""

__version__ = "0.1.0"

PAPER_LABEL = "PAPER TRADE — NOT REAL MONEY"
