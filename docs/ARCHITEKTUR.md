# Architektur — TradingAgents Evolution

Dieser Fork von [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)
(Basis: v0.7.1, Commit `9fa1196`) erweitert das Multi-Agenten-Framework um
einen deterministischen Forschungskern: Daten-Snapshots, Indikatoren,
Strategien, einen Portfolio-Backtester, Walk-Forward-Validierung, eine
Risiko-Engine und Paper-Trading.

**Grenze des Systems:** Research, Backtesting und Paper-Trading. Es gibt kein
Modul, das eine echte Order platziert, und keine Einstellung, die das
einschaltet. Jede simulierte Order trägt `PAPER TRADE — NOT REAL MONEY`.

Die Leitfrage ist nicht «Kann die KI den nächsten Kurs vorhersagen?», sondern:
**Lässt sich mit verlässlichen Daten und realistischen Tests zeigen, dass eine
Strategie nach Risiko und Kosten einen wiederholbaren Vorteil hat?**

---

## 1. Ist-Zustand des Upstreams

Das Repo ist weiter, als man aus dem Paper schliessen würde. Diese Teile sind
solide und werden weiterverwendet, nicht ersetzt.

### Agentengraph

`tradingagents/graph/setup.py: GraphSetup.setup_graph` baut einen LangGraph:

```
START ─┬─ Market Analyst ──┐
       ├─ News Analyst ────┤
       ├─ Sentiment Analyst┼─► Bull ⇄ Bear ─► Research Manager ─► Trader
       ├─ Fundamentals ────┤                                         │
       └─ Memory Log ──────┘    Aggressive ⇄ Conservative ⇄ Neutral ◄┘
                                              │
                                     Portfolio Manager ─► END
```

- Jeder Analyst ist ein eigener Subgraph mit einem Ein-Schlüssel-`output_schema`,
  damit parallele Analysten nie denselben State-Schlüssel schreiben.
- Zwei LLM-Stufen: *quick* (Analysten, Debatten, Trader) und *deep*
  (Research Manager, Portfolio Manager).
- Symbol und Datum kommen per `InjectedState` aus dem State; ein Modell kann
  weder einen anderen Ticker noch ein späteres Datum anfragen
  (`tradingagents/agents/tools.py`).
- Research Manager, Trader, Portfolio Manager und Sentiment liefern
  pydantic-Modelle (`tradingagents/agents/schemas.py`). Das endgültige Signal
  ist ein Fünf-Stufen-Rating oder `REVIEW` (`tradingagents/agents/rating.py`).

### Datenschicht

- `dataflows/router.py` mit konfigurierbaren Vendor-Ketten; ein nicht genannter
  Vendor wird nie stillschweigend als Ersatz verwendet.
- Fehler-Taxonomie in `dataflows/errors.py` (`NoMarketDataError`,
  `VendorUnavailableError`, `VendorNotConfiguredError`), Sentinels
  `DATA_UNAVAILABLE:` statt erfundener Werte.
- Staleness-Guard (`MAX_OHLCV_STALE_DAYS = 10`), Unterscheidung «Vendor
  ausgefallen» vs. «Symbol hat keine Daten», Point-in-Time-Klemme in
  `dataflows/date_window.py`.
- Secret-Scrubbing in Fehlermeldungen (`dataflows/net.py`), Windows-feste
  Datei-Locks (`dataflows/files.py`).

### Tests

104 Testdateien, 1407 Tests (Referenzlauf vom 09.11.2026 unter Windows,
Python 3.13: 1407 bestanden, 6 übersprungen, alle Überspringungen POSIX-only
oder optionales `langchain_aws`). `tests/conftest.py` leert alle
`TRADINGAGENTS_*`-Variablen vor dem Import und blockiert jedes Netz per
autouse-Fixture. CI: ruff + pytest auf Linux, Python 3.11–3.14, plus ein Lauf
in `TZ=America/New_York`.

## 2. Lücken

| # | Lücke | Fundstelle |
|---|---|---|
| 1 | **Keine deterministische Risiko-Engine.** Risikomanagement sind drei LLM-Personas, die in Prosa debattieren. Kein Positionslimit, keine Konzentrationsgrenze, kein Drawdown-Stopp. | `agents/risk_mgmt/*` |
| 2 | **Keine Positionsgrössen-Mathematik.** `TraderProposal.position_sizing` ist ein String. `PortfolioContext` wird nur in Prompts gerendert. | `agents/schemas.py`, `portfolio.py` |
| 3 | **Kein Portfolio-Backtester.** `backtest.py` bewertet Entscheidungsqualität (Trefferquote, mittleres Alpha) und verbietet im Docstring, daraus einen Simulator zu machen. Keine Equity-Kurve, keine Kosten, kein Drawdown. | `backtest.py` |
| 4 | **Keine Datenbank.** Gedächtnis ist eine regex-geparste Markdown-Datei. SQLite nur als LangGraph-Checkpoint. | `memory/log.py` |
| 5 | **Kein Replay.** Alles live; ein vergangener Lauf ist nicht reproduzierbar, News und Social sind nicht point-in-time archiviert. | `dataflows/*` |
| 6 | **Keine Konsistenzprüfung** Research-Empfehlung → Trader-Action → PM-Rating. | `graph/setup.py` |
| 7 | **Keine Prüfung der Modellzahlen.** Niemand prüft, ob `stop_loss` auf der richtigen Seite von `entry_price` liegt oder ob `entry_price` zum letzten Kurs passt. | `agents/trader/trader.py` |
| 8 | **Strukturierte Ausgabe ist best effort.** `agents/structured.py` verschluckt jede Exception und fällt still auf Prosa zurück; das Rating kommt dann aus einem Regex. | `agents/structured.py` |
| 9 | **Keine Kosten- oder Token-Budgets.** | — |
| 10 | **Kein Windows-CI**, obwohl es Windows-spezifische Codepfade gibt. | `.github/workflows/ci.yml` |

Nebenbefund aus M0: `load_ohlcv` lädt **nur fünf Jahre bis heute** und **nur
adjustierte Kurse** (`auto_adjust=True, actions=False`). Für Walk-Forward über
viele Fenster ist das zu kurz, und ohne Dividenden und Splits lässt sich eine
rückwirkende Restatierung nicht erkennen. Der Ingest des Forschungskerns
verwendet deshalb die gehärteten Bausteine (`yf_retry`, `raise_for_empty`,
`normalize_symbol`, `safe_ticker_component`), lädt aber die volle Historie samt
Corporate Actions.

## 3. Zielarchitektur

### Ein zweites Paket neben `tradingagents/`

```
tradingagents_quant/
  config.py  paths.py  errors.py  cli.py              M0
  store/      migrations.py · db.py                   M1  sqlite3, PRAGMA user_version
  data/       normalize · quality · snapshot · ingest · ingest_yahoo
  indicators/ core.py · registry.py                   M2
  regime.py
  strategy/   spec.py · library.py                    M3/M4
  sim/        events · execution · position · engine · metrics
              portfolio · benchmark · walkforward · optimize · deflate
  risk/       engine.py                               M6  Grenzen in config.RiskSettings
  paper/      ledger · account · runner                   Hauptbuch mit Hash-Kette
  bridge/     gate.py · consistency.py                M7  einzige Brücke zur Agentenschicht
  loop/       research_loop.py · review.py            M9
  eval/       calibration.py · agents.py
  api/        app.py · routes.py · static/            M8  nur mit Extra [dashboard]
tests/quant/                                          erbt tests/conftest.py
```

Der Name ist absichtlich lang: `quant/backtest/` neben `tradingagents/backtest.py`
verwechselt man schnell. Das Simulationspaket heisst deshalb `sim/`, und
`tradingagents/backtest.py` bleibt, was es ist — ein Bewerter der
Entscheidungsqualität.

### Abhängigkeitsregel

Kein Modul in `tradingagents_quant` importiert beim Laden `tradingagents`,
`yfinance`, `langchain*`, `langgraph`, `dotenv` oder `curl_cffi`. Wer sie
braucht (`data/ingest_yahoo.py`, `bridge/*`, `loop/*`), importiert sie in der
Funktion, die sie braucht.

Der Grund ist nicht Sauberkeit, sondern Reproduzierbarkeit:
`tradingagents/__init__.py` führt beim Import `load_dotenv(find_dotenv(usecwd=True))`
aus und verändert damit `os.environ`. Ein Backtest, der irgendetwas aus
`tradingagents` importiert, würde sich je nach der `.env`, die zufällig über
dem Arbeitsverzeichnis liegt, anders verhalten.

Umgekehrt darf genau ein Upstream-Modul in den Kern greifen:
`tradingagents/graph/setup.py`, für das Risiko-Tor (M7).

`tests/quant/test_boundary.py` erzwingt beides: ein AST-Scan nennt die
verletzende Zeile, ein frischer Interpreter importiert jedes Modul des Kerns und
findet auch transitive Importe.

### Die Naht zur Agentenschicht (M7)

Ein neuer Node `Risk Gate` zwischen `Portfolio Manager` und `END`. Reiner
Python-Code, kein LLM-Aufruf. Er prüft die Konsistenz der drei strukturierten
Entscheide, prüft `entry_price`/`stop_loss`/`price_target` gegen die Kurse im
eigenen Snapshot-Store, ruft Risiko-Engine und Sizing und schreibt
`risk_verdict`, `sim_order` und `data_quality` in den State.

Vertragsbedingungen, die aus den bestehenden Tests folgen:

1. **Strikt additiv** — `final_rating` und `final_trade_decision` bleiben
   unverändert; eine Ablehnung lebt in `risk_verdict` und wirkt beim
   Paper-Broker.
2. **Kein Router-Aufruf** — das Tor liest Kurse nur aus dem eigenen Store.
3. **Wirft nie** — jede Störung wird zum Verdikt.

Abgelehnt wird bei `REVIEW`, bei stillem Rückfall auf Prosa, bei Inkonsistenz,
bei fehlenden Daten, bei verletzter Grenze und wenn eine Risikokennzahl nicht
berechenbar ist. Kein LLM kann das Tor überstimmen.

### Speicherorte

| Was | Wo | Warum |
|---|---|---|
| Datenbank, Snapshots | `%LOCALAPPDATA%\tradingagents-quant\` (bzw. `$XDG_DATA_HOME`), verschiebbar mit `TAQ_DATA_DIR` | SQLite und ein Sync-Client (OneDrive, Dropbox) ergeben gesperrte oder korrupte Datenbanken. `paths.py` verweigert solche Orte — auch über Junctions und Symlinks. |
| Konfiguration | `taq.toml` im Datenverzeichnis oder `TAQ_CONFIG`; Vorlage `config/taq.example.toml` | Enthält nie Zugangsdaten. |
| LLM-Schlüssel | Umgebungsvariablen (`GOOGLE_API_KEY`) | Laufen nie durch die Konfiguration des Kerns und landen deshalb weder in Logs noch in Reports. |

### Sicherheitsvorgaben

`research_mode`, `backtesting_mode`, `paper_trading_mode` sind schaltbar.
`live_trading_mode` und `live_order_execution` sind als Konstanten typisiert
(`Literal[False]`, `Literal["disabled"]`): jeder andere Wert in Datei oder
Umgebung bricht den Start mit `LiveTradingForbiddenError` ab, ebenso jede der
Variablen `LIVE_TRADING_MODE`, `LIVE_ORDER_EXECUTION` und ihre `TAQ_`-Varianten.

### Keine schweren Abhängigkeiten im Kern

Die CI testet bis Python 3.14. Fehlt dort ein Wheel, baut pip aus den Quellen —
und unter Windows scheitert MSVC an Pfaden mit Umlaut. Deshalb: kein Parquet
(SQLite und CSV), kein scipy (Normalverteilung aus `statistics.NormalDist`),
kein TA-Lib (Indikatoren selbst in pandas/numpy). FastAPI und uvicorn gibt es
nur mit dem Extra `[dashboard]`.

## 4. Roadmap

| M | Inhalt | Status |
|---|---|---|
| M0 | Fork, Umgebung, Paketnaht, Konfiguration, Speicherorte, Grenztests, dieser Bericht | erledigt |
| M1 | Snapshot-Store (unveränderlich, gehasht), Ingest mit Corporate Actions, Qualitätsregeln, CLI `ingest`/`verify`/`report` — siehe [DATEN.md](DATEN.md) | erledigt |
| M2 | Indikatoren, Regime-Klassifikation; Paritäts- und Kausalitätstests — siehe [INDIKATOREN.md](INDIKATOREN.md) | erledigt |
| M3 | Single-Asset-Engine mit Kosten, Slippage, Stops, Sizing, Kennzahlen; Oracle-, Shift- und Trunkierungstest — siehe [BACKTESTING.md](BACKTESTING.md) | erledigt |
| M4 | Portfolio-Ebene, Strategie-Bibliothek, Benchmarks — siehe [STRATEGIEN.md](STRATEGIEN.md) | erledigt |
| M5 | Walk-Forward (embargoed), robuste Optimierung, Deflated Sharpe, Sharpe-Differenztest, Experiment-Register, Präregistrierung — siehe [VALIDIERUNG.md](VALIDIERUNG.md) | erledigt |
| M6 | Risiko-Engine, Paper-Trading als dieselbe Engine (Stepper) mit Hash-Ketten-Hauptbuch — siehe [RISIKO.md](RISIKO.md), [PAPER_TRADING.md](PAPER_TRADING.md) | erledigt |
| M7 | Risiko-Tor im Agentengraphen, LLM-Replay-Cache, Token-Budget | offen |
| M8 | API und Dashboard | offen |
| M9 | Research-Loop, Evaluation und Kalibrierung, Doku, Windows-CI | offen |

## 5. Wie ein Vorteil nachgewiesen wird

**Deterministische Strategien** sind rückwärts ehrlich prüfbar: Walk-Forward mit
purged/embargoed Splits, Kosten- und Parametersensitivität, Null-Modelle,
Regimesplit, mindestens 100 nicht überlappende Trades, Deflated Sharpe mit der
tatsächlich gezählten Versuchszahl.

**Die Agentenschicht** ist rückwärts nicht ehrlich prüfbar: News und Social sind
nicht point-in-time archiviert, und das Modell kennt die Backtest-Periode aus
dem Pretraining. Für sie gibt es deshalb keine Rendite-Aussage aus
historischen Läufen. Sie wird vorwärts gemessen — als Veto oder Size-Modifier
auf dem deterministischen Signal, gepaart gegen das Signal ohne sie — und
rückwärts nur auf Kalibrierung (Rating gegen realisiertes 5-Tage-Alpha).

**Was das System nie behauptet:** dass ein positiver Backtest künftigen Gewinn
bedeutet. Das Standarduniversum besteht aus ETFs, weil eine Liste heute
bekannter Einzelaktien den Survivorship Bias schon in der Auswahl trägt; wo er
trotzdem auftritt, wird er im Report ausgewiesen.
