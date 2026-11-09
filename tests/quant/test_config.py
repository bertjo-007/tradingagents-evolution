"""Configuration: research-only by construction, strict about bad values."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from tradingagents_quant import PAPER_LABEL
from tradingagents_quant.cli import app
from tradingagents_quant.config import (
    _LIVE_VARIABLES,
    QuantConfig,
    load_config,
)
from tradingagents_quant.errors import ConfigError, LiveTradingForbiddenError

REPO = Path(__file__).resolve().parents[2]


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {"TAQ_DATA_DIR": str(tmp_path / "data"), **extra}


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "taq.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_are_research_backtesting_and_paper_only(tmp_path):
    modes = load_config(environ=_env(tmp_path)).modes
    assert modes.research_mode and modes.backtesting_mode and modes.paper_trading_mode
    assert modes.live_trading_mode is False
    assert modes.live_order_execution == "disabled"


@pytest.mark.parametrize("name", _LIVE_VARIABLES)
@pytest.mark.parametrize("value", ["true", "1", "yes", "enabled", "live"])
def test_a_variable_asking_for_live_trading_stops_the_start(tmp_path, name, value):
    with pytest.raises(LiveTradingForbiddenError, match=name):
        load_config(environ=_env(tmp_path, **{name: value}))


@pytest.mark.parametrize("name", _LIVE_VARIABLES)
@pytest.mark.parametrize("value", ["", "false", "0", "off", "disabled"])
def test_a_live_variable_that_says_off_is_accepted(tmp_path, name, value):
    assert load_config(environ=_env(tmp_path, **{name: value})).modes.live_trading_mode is False


@pytest.mark.parametrize(
    "line", ["live_trading_mode = true", 'live_order_execution = "enabled"']
)
def test_live_trading_in_the_file_stops_the_start(tmp_path, line):
    path = _write(tmp_path, f"[modes]\n{line}\n")
    with pytest.raises(LiveTradingForbiddenError):
        load_config(path, environ=_env(tmp_path))


def test_the_file_is_found_in_the_data_directory(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "taq.toml").write_text('[universe]\nbenchmark = "QQQ"\n', encoding="utf-8")
    assert load_config(environ=_env(tmp_path)).universe.benchmark == "QQQ"


def test_variables_override_the_file(tmp_path):
    path = _write(tmp_path, '[universe]\nbenchmark = "QQQ"\n')
    env = _env(
        tmp_path,
        TAQ_UNIVERSE_BENCHMARK="iwm",
        TAQ_UNIVERSE_SYMBOLS=" spy, qqq ,tlt",
        TAQ_DATA_MAX_STALE_DAYS="5",
        TAQ_MODES_PAPER_TRADING_MODE="false",
    )
    cfg = load_config(path, environ=env)
    assert cfg.universe.benchmark == "IWM"
    assert cfg.universe.symbols == ("SPY", "QQQ", "TLT")
    assert cfg.data.max_stale_days == 5
    assert cfg.modes.paper_trading_mode is False


@pytest.mark.parametrize(
    "env",
    [
        {"TAQ_MODES_RESEARCH_MODE": "ja"},
        {"TAQ_DATA_MAX_STALE_DAYS": "ten"},
        {"TAQ_DATA_MAX_STALE_DAYS": "0"},
        {"TAQ_DATA_HISTORY_START": "1.1.2000"},
        {"TAQ_UNIVERSE_SYMBOLS": "SPY,spy"},
        {"TAQ_UNIVERSE_SYMBOLS": " , "},
    ],
)
def test_a_bad_value_is_an_error_not_a_silent_default(tmp_path, env):
    with pytest.raises(ConfigError):
        load_config(environ=_env(tmp_path, **env))


@pytest.mark.parametrize(
    "text",
    [
        "[universe]\nsymbol = ['SPY']\n",  # misspelt field
        "[nonsense]\nx = 1\n",  # unknown section
        "[universe\n",  # not TOML
    ],
)
def test_a_malformed_file_is_an_error(tmp_path, text):
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, text), environ=_env(tmp_path))


def test_a_missing_file_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.toml", environ=_env(tmp_path))


def test_the_example_file_matches_the_defaults(tmp_path):
    example = load_config(REPO / "config" / "taq.example.toml", environ=_env(tmp_path))
    assert example == QuantConfig()


def test_the_fingerprint_is_stable_and_follows_every_setting(tmp_path):
    a = load_config(environ=_env(tmp_path))
    b = load_config(environ=_env(tmp_path / "other"))  # another data directory
    c = load_config(environ=_env(tmp_path, TAQ_UNIVERSE_BENCHMARK="QQQ"))
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


def test_the_configuration_cannot_be_changed_after_loading(tmp_path):
    cfg = load_config(environ=_env(tmp_path))
    with pytest.raises(ValidationError):
        cfg.universe.benchmark = "QQQ"


def test_the_benchmark_and_calendar_are_ingested_once():
    cfg = QuantConfig.model_validate(
        {"universe": {"symbols": ["QQQ", "TLT"], "benchmark": "SPY", "calendar_symbol": "SPY"}}
    )
    assert cfg.universe.ingest_symbols() == ("QQQ", "TLT", "SPY")


def test_the_configuration_has_no_place_for_a_secret():
    def names(model) -> list[str]:
        found = []
        for field, info in model.model_fields.items():
            found.append(field)
            annotation = info.annotation
            if isinstance(annotation, type) and hasattr(annotation, "model_fields"):
                found += names(annotation)
        return found

    # Whole words: "api_key", "auth_token", "secret" are suspicious;
    # "max_tokens_per_run" is a budget, not a credential.
    import re

    secret = re.compile(r"(^|_)(key|token|secret|password|credential)s?($|_)")
    suspicious = [n for n in names(QuantConfig) if secret.search(n) and n != "max_tokens_per_run"
                  and n != "max_tokens_per_day"]
    assert suspicious == []
    assert secret.search("api_key") and secret.search("auth_token") and secret.search("secret")


def test_cli_prints_the_configuration_and_the_paper_label(tmp_path):
    result = CliRunner().invoke(app, ["config"], env={"TAQ_DATA_DIR": str(tmp_path / "data")})
    assert result.exit_code == 0, result.output
    assert PAPER_LABEL in result.output
    assert "fingerprint" in result.output


def test_cli_refuses_to_start_when_live_trading_is_asked_for(tmp_path):
    result = CliRunner().invoke(
        app, ["config"], env={"TAQ_DATA_DIR": str(tmp_path / "data"), "LIVE_TRADING_MODE": "true"}
    )
    assert result.exit_code == 2
