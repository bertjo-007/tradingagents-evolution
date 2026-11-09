# Daten — Snapshots, Quelle, Qualitätsregeln

Der Forschungskern rechnet **nie** auf Live-Daten. Jeder Backtest, jede
Walk-Forward-Validierung und jeder Paper-Trading-Schritt liest aus einem
**Snapshot**: einer unveränderlichen, gehashten Kopie der Marktdaten, so wie
der Anbieter sie zu einem bestimmten Zeitpunkt geliefert hat.

## Warum Snapshots

Kursanbieter ändern ihre Historie rückwirkend. Ein reales Beispiel aus dem
ersten Ingest (09.11.2026): XLB, XLE, XLK, XLU und XLY hatten am 05.12.2025
einen 2:1-Split. Weil Yahoo den `Close` split-adjustiert liefert, sind seitdem
**alle** älteren Schlusskurse dieser ETFs halbiert. Ein Backtest, der live
lädt, hätte vor und nach dem Split mit anderen Zahlen gerechnet — ohne dass es
jemand merkt. Mit Snapshots bleibt ein Ergebnis reproduzierbar, und
`taq verify --drift` zeigt, was der Anbieter seither geändert hat.

## Quelle: Yahoo Finance

| | |
|---|---|
| Modul | `tradingagents_quant/data/ingest_yahoo.py` |
| Schlüssel | keiner |
| Abruf | `Ticker.history(auto_adjust=False, actions=True)`, volle Historie ab `data.history_start` |
| Art der Daten | **historische Tagesbars**, keine Echtzeitdaten. Die laufende Sitzung wird nie gespeichert. |
| Wiederverwendet | `yf_retry` (Backoff bei Rate-Limits), `raise_for_empty` (prüft die Erreichbarkeit, bevor ein Symbol «leer» heisst), `normalize_symbol`, `safe_ticker_component` aus `tradingagents/dataflows` |

Warum nicht `load_ohlcv` aus dem Upstream: es lädt nur fünf Jahre bis heute und
nur adjustierte Kurse ohne Dividenden und Splits. Für Walk-Forward über viele
Fenster ist das zu kurz, und eine Restatierung wäre nicht erkennbar.

### Was gespeichert wird

| Spalte | Bedeutung |
|---|---|
| `date` | Börsenlokales Datum `YYYY-MM-DD` (nicht nach UTC verschoben) |
| `open`, `high`, `low`, `close` | **split-adjustiert**, nicht dividenden-adjustiert (Yahoos Konvention) |
| `adj_close` | split- und dividenden-adjustiert |
| `volume` | wie geliefert |
| `corporate_actions` | Dividenden, Splits, Kapitalgewinn-Ausschüttungen mit Datum und Wert |

Nichts wird aufgefüllt. Ein Wert, den der Anbieter nicht liefert, bleibt leer
(`NULL`); ein Bar ohne Schlusskurs wird nicht gespeichert, sondern als
Qualitätsbefund vermerkt. Fällt ein Symbol ganz aus, erscheint es unter den
Ausfällen des Snapshots — es bekommt keine Bars, schon gar nicht die eines
anderen Anbieters.

### Herkunft jedes Werts

Jeder Bar hängt an einem Snapshot, und der Snapshot nennt Quelle
(`yahoo`), Bibliotheksversion (`yfinance 1.8.0`), Abrufzeit (UTC),
angefragten Zeitraum und den Fingerprint der Konfiguration. Damit ist für jede
Zahl in einem Report nachvollziehbar, woher sie stammt.

## Unveränderlichkeit

Ein Snapshot wird in **einer** Transaktion geschrieben und versiegelt. Danach
verweigert die Datenbank selbst — per Trigger, nicht nur per Konvention —
jedes `UPDATE`, `DELETE` und jedes nachträgliche `INSERT`. Bricht ein Ingest
mittendrin ab, bleibt nichts zurück.

Jede Serie trägt einen SHA-256 über ihre exakten Werte, der Snapshot einen
Hash über alle Serien. `taq verify` rechnet beides aus den gespeicherten Zeilen
nach.

## Qualitätsregeln

Die Regeln melden, sie reparieren nie. Eine Serie mit einem Befund der Stufe
`error` ist für Backtests gesperrt; `warning` wird bei jedem Ergebnis
mitgezeigt.

| Regel | Stufe | Bedeutung |
|---|---|---|
| `empty_series` | error | kein abgerechneter Bar mit Schlusskurs |
| `non_positive_price` | error | ein Preis ≤ 0 |
| `ohlc_inconsistent` | error | High/Low widersprechen Open/Close um mehr als 1 % des Schlusskurses |
| `ohlc_rounding` | warning | dasselbe, aber unter 1 % (Rundungsrauschen) |
| `negative_volume` | error | Volumen < 0 |
| `zero_volume` | warning | Volumen = 0 |
| `duplicate_conflicting` | error | dasselbe Datum zweimal mit verschiedenen Werten (der erste wird behalten) |
| `duplicate_identical` | info | dasselbe Datum zweimal, identisch |
| `missing_close` | warning | Bar ohne Schlusskurs, nicht gespeichert |
| `unsettled_session` | info | heutige Sitzung, nicht gespeichert |
| `missing_adj_close` | warning | kein adjustierter Schlusskurs |
| `adj_close_above_close` | warning | adjustierter über unadjustiertem Kurs |
| `extreme_move` | warning | Tagesbewegung über 25 % ohne Split an diesem Tag |
| `stale_series` | error | letzter Bar älter als `data.max_stale_days` vor dem Stichtag |
| `calendar_gap` | warning | Handelstag des Kalendersymbols ohne Bar |
| `calendar_extra` | warning | Bar an einem Tag, der kein Handelstag des Kalendersymbols ist |
| `calendar_unavailable` | warning | Kalendersymbol fehlt im Snapshot, Lücken ungeprüft |

**Handelskalender:** die Handelstage sind die Daten, an denen das
Kalendersymbol (Standard: SPY) einen Bar hat. Keine Abhängigkeit, und
automatisch point-in-time.

Erster echter Ingest am 09.11.2026: 16 Serien, 108 412 Bars (1999–2026), ein
einziger Befund — `zero_volume` bei XLI am 26.01.1999.

## Befehle

```powershell
taq ingest                                  # ganzes Universum, volle Historie
taq ingest --symbols SPY,QQQ --start 2010-01-01 --end 2025-12-31
taq snapshots                               # alle versiegelten Snapshots
taq report --details                        # Qualität je Serie, alle Befunde
taq verify                                  # Hashes nachrechnen (offline)
taq verify --drift                          # neu abrufen und vergleichen (Netz)
```

## Speicherort

`%LOCALAPPDATA%\tradingagents-quant\taq.sqlite3`, verschiebbar mit
`TAQ_DATA_DIR`. Nie in OneDrive oder einem anderen Sync-Ordner — `paths.py`
verweigert das, auch über Junctions. Ein Snapshot des Standarduniversums
belegt rund 12 MB.

## Grenzen

- **Survivorship Bias:** Yahoo kennt keine delisteten Titel. Das
  Standarduniversum besteht deshalb aus breiten und Sektor-ETFs mit Historie
  ab 1998–2004, nicht aus heute bekannten Einzelaktien. Für ein eigenes
  Universum gilt: wer nur heutige Gewinner testet, testet die Auswahl, nicht
  die Strategie.
- **Keine Echtzeitdaten.** Paper-Trading arbeitet mit dem letzten
  abgerechneten Tagesbar.
- **Yahoo ist eine Endkundenquelle**, keine geprüfte Börsendatenquelle.
  Die Qualitätsregeln fangen grobe Fehler, keine subtilen.
- **Fundamentaldaten und News** sind (noch) nicht Teil der Snapshots; die
  Agentenschicht holt sie weiterhin live über den Upstream-Router.
