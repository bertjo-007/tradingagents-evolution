# Strategien

Eine Strategie ist **Daten, kein Prompt**: eine eingefrorene `StrategySpec`
(`tradingagents_quant/strategy/spec.py`) plus eine registrierte,
deterministische Regel (`strategy/library.py`). Jede Änderung an Regel,
Parameter, Sizing oder Ausstieg ergibt einen neuen `version_hash` und wird mit
den Ergebnissen gespeichert, die sie erzeugt hat. Niemand — kein Agent, kein
Optimierer — ändert eine Spezifikation an Ort und Stelle.

## Was eine Spezifikation enthält

| Briefing verlangt | Feld |
|---|---|
| Instrumente | `symbols` |
| Zeitrahmen | `timeframe` (`1d`: Entscheid am Close, Ausführung am nächsten Open) |
| Einstieg / Ausstieg | die Regel (`rule`, `params`) plus `exits` |
| Positionsgrösse | `sizing` (`fixed_fraction`, `vol_target`, `atr_risk`) |
| Risikogrenzen | `exits` (Stop, Ziel, maximale Haltedauer); Portfolio-Grenzen kommen in M6 |
| Benötigte Daten | aus der Regel (`required`) |
| Benchmark | `benchmark` (Standard: das Index-Symbol der Konfiguration) |
| Erwartete Haltedauer | `expected_holding_bars` oder der Richtwert der Regel |

`spec.describe()` gibt all das in Worten aus.

## Die Regeln

Jede Regel liefert pro Bar ein Signal in [0, 1], entschieden am Close aus den
Bars bis dahin. In der Aufwärmphase ist das Signal `NaN` — keine Position, und
sichtbar warum.

| Regel | Familie | Einstieg | Ausstieg | Standard |
|---|---|---|---|---|
| `buy_and_hold` | Baseline | sofort | nie | — |
| `cash` | Baseline | nie | — (Nullrendite; freie Kasse wird nicht verzinst) | — |
| `sma_cross` | MA-Kreuzung | SMA(fast) > SMA(slow) | SMA(fast) ≤ SMA(slow) | 50 / 200 |
| `trend_following` | Trendfolge | Close > SMA(window) | Close ≤ SMA(window) | 200 |
| `momentum` | Momentum | Rendite von vor *lookback* bis vor *skip* Bars > *threshold* | darunter | 252 / 21 / 0 % |
| `mean_reversion` | Mean Reversion | RSI(n) < *entry* und Close über SMA(*trend*) | RSI(n) > *exit* | RSI(2), 10 / 70, SMA 200 |
| `breakout` | Breakout | Close über dem höchsten Hoch der **vorherigen** *entry* Bars | Close unter dem tiefsten Tief der vorherigen *exit* Bars | 55 / 20 |
| `volatility_filter` | Volatilität | 20-Tage-Volatilität ≤ ihr *q*-Quantil der letzten *lookback* Bars | darüber | 20 / 252 / 50 % |
| `regime_switch` | Regime-abhängig | Trend-Regime `up` oder `strong_up` (optional: nicht bei hoher Volatilität) | jedes andere Regime | — |

Regeln mit Zustand (Mean Reversion, Breakout) halten ihre Position zwischen
Ein- und Ausstiegssignal über eine Vorwärtsschleife (`hold_between`), die nur
vergangene Bars liest. Widersprüchliche Parameter (Ausstieg über Einstieg,
schnelle über langsamer Linie) werden abgewiesen.

**Jede Regel** muss den Trunkierungstest in
`tests/quant/test_engine_leakage.py` bestehen: abgeschnitten bei Tag *t* trifft
sie dieselben Entscheidungen wie auf allen Daten. Eine neue Regel, die dort
fehlt, lässt die Testsuite scheitern.

## Mehrere Instrumente

Mit *N* Instrumenten bekommt jedes höchstens 1/*N* des Kapitals: das Signal
und die Grösse jedes Instruments werden durch *N* geteilt. Ein einzelnes
aktives Signal konzentriert so nie das ganze Konto; der Rest bleibt in Kasse.
Die Summe aller Gewichte übersteigt nie 100 %. Eine gemeinsame Kasse bedient
alle; Verkäufe werden vor Käufen ausgeführt, und reicht das Geld nicht für alle
Käufe, werden sie anteilig gekürzt (und das vermerkt). Ein Instrument wird vor
seinem ersten Handelstag nicht gehandelt.

## Benchmarks

Jeder Backtest läuft gegen:

| Benchmark | Was |
|---|---|
| Buy & Hold | dieselben Instrumente, gleich gewichtet, nie umgeschichtet, **Dividenden reinvestiert** |
| Index | Buy & Hold des Index-Symbols (Standard SPY), Dividenden reinvestiert |
| SMA 50/200 | die einfache Kreuzungsstrategie auf dem Index |
| Cash | Nullrendite |

Warum Dividenden reinvestieren: Ohne sie lägen die Dividenden des Buy-and-Hold
jahrzehntelang als unverzinste Kasse herum, während eine umschichtende
Strategie ihre eigenen nebenbei wieder anlegt. Das liess SPY Buy-and-Hold
2005–2026 bei +589 % statt +805 % enden und hätte jede Strategie besser
aussehen lassen, als sie ist. Gegenprobe: der unabhängig aus Kursen und
Dividenden gerechnete Total-Return-Index ergibt +807 %; die Differenz zur
Simulation (Kosten, ganze Aktien, Reinvestition am nächsten Open) beträgt
0,45 % des Endwerts.

## Stand der Befunde

Alle Regeln mit Standardparametern auf SPY, 2005-01-03 bis 2026-11-06,
Standardkosten, Snapshot `snap-20261109T103320Z-a900e8b8`:

| Regel | CAGR | Sharpe | Max. Drawdown |
|---|---|---|---|
| Buy & Hold (DRIP) | **10,6 %** | 0,63 | −55 % |
| `trend_following` | 7,4 % | 0,68 | −24 % |
| `regime_switch` | 7,0 % | 0,68 | −20 % |
| `sma_cross` | 8,4 % | 0,66 | −33 % |
| `momentum` | 8,4 % | 0,63 | −34 % |
| `volatility_filter` | 5,2 % | 0,61 | −21 % |
| `mean_reversion` | 3,8 % | 0,58 | −15 % |
| `breakout` | 2,6 % | 0,34 | −24 % |

Neun Sektor-ETFs mit `trend_following` (2005–2026): CAGR 6,0 % gegen 9,0 %
für Buy & Hold gleich gewichtet, Sharpe 0,63 gegen 0,62, Drawdown −24 %
gegen −48 %, bei 859 Trades und rund 15 700 USD Kosten.

**Was das heisst:** Keine Regel schlägt Buy-and-Hold bei der Rendite. Die
höchsten Sharpe-Werte liegen 0,05 darüber — bei einem Standardfehler der
Sharpe-Ratio von rund 0,2 über 22 Jahre ist das Rauschen, kein Vorteil. Was
alle Regeln liefern, ist ein kleinerer Drawdown gegen weniger Rendite: ein
Tausch, kein Edge.

Und all das ist **in-sample**: Standardparameter, auf derselben Periode
gemessen, auf der sie seit Jahrzehnten bekannt sind. Ob irgendetwas davon
ausserhalb der Stichprobe hält, prüft die Walk-Forward-Validierung (M5).
