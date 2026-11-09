"""FastAPI app of the research dashboard. Read-only: GET endpoints only, and
each request opens the database in read-only mode."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from tradingagents_quant import PAPER_LABEL, paths
from tradingagents_quant.api import queries as q
from tradingagents_quant.config import load_config
from tradingagents_quant.errors import SnapshotError

STATIC = Path(__file__).parent / "static"


def create_app(environ: Mapping[str, str] | None = None) -> FastAPI:
    cfg = load_config(environ=environ)
    home = paths.data_home(environ)
    app = FastAPI(title="TradingAgents Evolution — research dashboard",
                  description=f"Read-only view of the research core. {PAPER_LABEL}.",
                  docs_url="/api/docs", redoc_url=None)

    def with_db(fn: Callable[..., Any], *, need_db: bool = True):
        db, store = q.open_store(home)
        try:
            if db is None and need_db:
                return None
            return fn(db, store)
        finally:
            if db is not None:
                db.close()

    def found(value: Any, what: str) -> Any:
        if value is None:
            raise HTTPException(status_code=404, detail=f"{what} not found")
        return value

    @app.middleware("http")
    async def no_write(request, call_next):
        if request.method not in ("GET", "HEAD"):
            return JSONResponse({"detail": "the dashboard is read-only"}, status_code=405)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/status")
    def status() -> Any:
        return with_db(lambda db, store: q.status(db, cfg, home), need_db=False)

    @app.get("/api/snapshot")
    def snapshot() -> Any:
        def run(db, store):
            snap = store.latest()
            return q.snapshot_summary(store, snap.snapshot_id) if snap else None
        return with_db(run) or {}

    @app.get("/api/market/{symbol}")
    def market(symbol: str) -> Any:
        def run(db, store):
            snap = store.latest()
            return q.market(store, snap.snapshot_id, symbol.upper()) if snap else None
        return found(with_db(run), f"market data for {symbol}")

    @app.get("/api/series/{symbol}")
    def series(symbol: str) -> Any:
        def run(db, store):
            snap = store.latest()
            if snap is None:
                return None
            try:
                return q.series(store, snap.snapshot_id, store.get(snap.snapshot_id)
                                .series_info(symbol.upper()).symbol)
            except SnapshotError:
                return None
        return found(with_db(run), f"series {symbol}")

    @app.get("/api/rules")
    def rules() -> Any:
        return q.rules()

    @app.get("/api/strategies")
    def strategies() -> Any:
        return with_db(lambda db, store: q.strategies(db)) or []

    @app.get("/api/backtests")
    def backtests() -> Any:
        return with_db(lambda db, store: q.backtests(db)) or []

    @app.get("/api/backtests/{run_id}")
    def backtest(run_id: str) -> Any:
        return found(with_db(lambda db, store: q.backtest(db, run_id)), f"backtest {run_id}")

    @app.get("/api/experiments")
    def experiments() -> Any:
        return with_db(lambda db, store: q.experiments(db)) or []

    @app.get("/api/experiments/{experiment_id}")
    def experiment(experiment_id: str) -> Any:
        return found(with_db(lambda db, store: q.experiment(db, experiment_id)),
                     f"experiment {experiment_id}")

    @app.get("/api/hypotheses")
    def hypotheses() -> Any:
        return with_db(lambda db, store: q.hypotheses(db), need_db=False)

    @app.get("/api/paper")
    def paper() -> Any:
        return with_db(lambda db, store: q.paper_accounts(db, store)) or []

    @app.get("/api/paper/{name}")
    def paper_account(name: str) -> Any:
        return found(with_db(lambda db, store: q.paper_account(db, store, name)),
                     f"paper account {name}")

    @app.get("/api/risk")
    def risk() -> Any:
        return with_db(lambda db, store: q.risk(db, cfg), need_db=False)

    @app.get("/api/agents")
    def agents() -> Any:
        return with_db(lambda db, store: q.agents(db, cfg), need_db=False)

    @app.get("/api/agents/{decision_id}")
    def agent_decision(decision_id: str) -> Any:
        return found(with_db(lambda db, store: q.agent_decision(db, decision_id)),
                     f"agent decision {decision_id}")

    @app.get("/api/settings")
    def settings() -> Any:
        return q.settings(cfg, home, dict(environ) if environ is not None else None)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
