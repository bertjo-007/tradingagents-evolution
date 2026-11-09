"""Command line of the research core: ``taq`` (or ``python -m tradingagents_quant.cli``)."""

from __future__ import annotations

import json
import os
import sys

import typer

from tradingagents_quant import PAPER_LABEL, __version__, paths
from tradingagents_quant.config import config_file, load_config
from tradingagents_quant.errors import ConfigError

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Deterministic research core: data, backtests, paper trading. No live trading.",
)


def _utf8_output() -> None:
    """Write UTF-8 even when the output is redirected.

    Python on Windows encodes redirected output in the ANSI code page, which
    turns the dash of the paper-trade label into a byte no UTF-8 reader shows.
    The label must arrive intact wherever the output goes.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if (stream.encoding or "").lower().replace("-", "") != "utf8":
                stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError, OSError):
            pass  # a stream that cannot be reconfigured (a test runner's buffer)


@app.callback()
def _main() -> None:
    """Keep ``taq`` a command group even while it has a single command."""
    _utf8_output()


def _fail(exc: Exception) -> None:
    typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=2)


@app.command("config")
def show_config() -> None:
    """Print the effective configuration, where it came from, and its fingerprint."""
    try:
        cfg = load_config()
        home = paths.data_home()
        source = config_file(os.environ, home)
    except ConfigError as exc:
        _fail(exc)
        return
    typer.echo(f"tradingagents_quant {__version__}")
    typer.echo(f"data directory : {home}")
    typer.echo(f"database       : {paths.db_path(home)}")
    typer.echo(f"config file    : {source or '(none, defaults)'}")
    typer.echo(f"fingerprint    : {cfg.fingerprint()}")
    typer.echo(f"orders         : {PAPER_LABEL}")
    typer.echo(json.dumps(cfg.to_dict(), indent=2))


if __name__ == "__main__":
    app()
