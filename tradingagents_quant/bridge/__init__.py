"""The bridge to the agent framework: the deterministic risk gate in the agent
graph, the record/replay cache for model calls and the token budget.

The only place the research core meets ``tradingagents``. Every import of the
agent framework or of langchain here sits inside a function.
"""
