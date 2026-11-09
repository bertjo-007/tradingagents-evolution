# Backtesting

Der Backtester (`tradingagents_quant/sim/`) ist deterministischer Code ohne
LLM. Er simuliert eine Strategie Bar für Bar auf einem Snapshot, mit Kosten,
Slippage, Stops, Dividenden und Kassenbuch, und vergleicht sie immer mit
Buy-and-Hold desselben Instruments.

Nicht zu verwechseln mit `tradingagents/backtest.py` des Upstreams: das
bewertet die Entscheidungsqualität der Agenten (Trefferquote, mittleres Alpha)
und ist bewusst kein Portfolio-Simulator.

Alle Ergebnisse sind **simuliert** und tragen `PAPER TRADE — NOT REAL MONEY`.

## Zeitmodell

Das Zeitmodell ist der Kern der Look-ahead-Freiheit.

| Zeitpunkt | Was geschieht |
|---|---|
| Morgen von Tag *t* | Wer am Close von *t*−1 hielt, erhält die Dividende mit Ex-Datum *t*. Dann wird die am Close von *t*−1 entschiedene Order **zum Open von *t*** ausgeführt. |
| Während Tag *t* | Stop und Ziel werden gegen Tief und Hoch des Bars geprüft. Eine über Nacht gehaltene Position, die unter ihren Stop eröffnet, wird zum (schlechteren) Open verkauft. |
| Close von Tag *t* | Bewertung zum Schlusskurs. Die Strategie entscheidet aus den Bars bis *t*; das ergibt die Order für den Open von *t*+1. |

Weitere Regeln:

- **Stop und Ziel im selben Bar:** der Stop wird angenommen. Tagesbars
  verraten die Reihenfolge nicht, und der Simulator wählt nicht die
  schmeichelhafte.
- **Grösse der Order** aus dem Close von *t* (dem letzten bekannten Preis).
  Öffnet *t*+1 höher, wird die Order auf das verfügbare Geld verkleinert und
  das als `reduced_for_cash` vermerkt.
- **Kein Open-Kurs** am Ausführungstag: keine Ausführung (`no_open_price`),
  die Order wird am nächsten Tag neu entschieden.
- **Nach einem Stop** wird erst wieder eingestiegen, wenn das Signal einmal auf
  flat gefallen ist (`reentry = after_signal_reset`, umstellbar).
- **Nicht berechenbarer Stop** (z. B. ATR in der Aufwärmphase): kein Einstieg
  (`stop_not_computable`). Eine Risikogrösse, die nicht feststeht, ist kein
  Trade.
- **Nur long, ohne Hebel.** Zielgewichte ausserhalb von [0, 1] werden
  abgewiesen; die Kasse wird nie negativ.

### Preise

Ausgeführt wird zu den notierten Kursen. Signale laufen standardmässig auf
einem **Total-Return-Index**, der vom ersten Bar vorwärts gerechnet wird
(`tr[t] = tr[t−1] · (close[t] + dividende[t]) / close[t−1]`) und deshalb nie
einen späteren Bar liest. Yahoos `adj_close` wird nicht verwendet: es ist
rückwärts adjustiert (eine Dividende von 2020 verändert einen Kurs von 2010 —
Look-ahead) und rauscht zwischen zwei Abrufen.

## Kosten

Konfigurierbar im Abschnitt `[costs]`. Standard für einen liquiden US-ETF:

| Posten | Standard |
|---|---|
| Kommission | 0,005 USD je Aktie, mindestens 1 USD, höchstens 1 % des Ordervolumens |
| Spread | 2 Basispunkte (jede Ausführung zahlt die Hälfte) |
| Slippage | 2 Basispunkte je Ausführung, immer gegen die Order |

Bewusst nicht optimistisch: der echte Spread von SPY liegt unter einem
Basispunkt. Ein Vorteil, der unter diesen Kosten verschwindet, war keiner.

## Positionsgrösse

`sim/sizing.py` ist die **einzige** Stelle, die Positionsgrössen rechnet.

| Verfahren | Gewicht |
|---|---|
| `fixed_fraction` | Signal × Anteil |
| `vol_target` | Ziel-Volatilität ÷ realisierte 20-Tage-Volatilität, gedeckelt beim Anteil |
| `atr_risk` | so, dass eine Bewegung von *k* ATR gegen die Position *r* % des Kapitals kostet, gedeckelt |

Ohne Volatilitätsschätzung (Aufwärmphase) ist die Grösse null, keine Schätzung.
Eine Position wird nur angepasst, wenn sie um mehr als `rebalance_band`
(Standard 20 %) vom Ziel abweicht.

## Kennzahlen

| Kennzahl | Definition |
|---|---|
| Gesamtrendite | Endkapital ÷ Startkapital − 1 |
| CAGR | annualisiert, nur ab `min_annualization_bars` (252) Bars |
| Volatilität | Standardabweichung der Tagesrenditen × √252 |
| Sharpe | mittlere Überrendite ÷ Standardabweichung × √252 |
| Sortino | mittlere Überrendite ÷ Abwärtsabweichung × √252 |
| Max. Drawdown | grösster Rückgang vom vorherigen Höchststand, mit Datum und längster Dauer |
| Trefferquote, Profit-Faktor | über abgeschlossene Trades; ohne Verlusttrade kein Profit-Faktor (statt «unendlich») |
| Exposure | Anteil der Tage mit Position |
| Turnover | gehandeltes Volumen ÷ mittleres Kapital (gesamt und pro Jahr) |
| Kosten | Kommission sowie Spread und Slippage, getrennt |
| nach Regime | Rendite eines Tages gehört zum Regime des Vortags (das beim Halten bekannt war) |

Annualisierte Kennzahlen gibt es unter 252 Bars nicht — sie wären eine
Aussage über die drei Monate, nicht über die Strategie.

## Wie Look-ahead ausgeschlossen wird

`tests/quant/test_engine_leakage.py` prüft auf drei Wegen:

1. **Trunkierung.** Ein Backtest auf Daten, die bei Tag *t* abgeschnitten sind,
   trifft bis *t* exakt dieselben **Entscheidungen** und Ausführungen wie einer
   auf allen Daten. Für jede Regel der Bibliothek, auch mit Stops und
   Volatilitäts-Sizing.
2. **Orakel.** Eine Strategie, die den morgigen Kurs *kennt*, muss absurd gut
   aussehen (gemessen: Sharpe 10,8). Sähe sie das nicht, würde die Engine das
   Timing verwischen — und damit auch ein echtes Leck verbergen können.
3. **Verspätetes Orakel.** Dasselbe Orakel einen Tag später hat keinen Vorteil
   mehr (gemessen: Sharpe −0,08).

**Ein Befund beim Bau:** Ein Trunkierungstest, der nur die Ausführungen
vergleicht, übersieht einen Blick um genau einen Tag. Die verfälschte
Entscheidung ist dann die letzte vor dem Schnitt, und ihre Ausführung läge am
Morgen danach — sie wird nie verglichen. Der Test vergleicht deshalb auch die
Entscheidungen, und `test_a_one_day_peek_is_caught` pflanzt genau dieses Leck
ein und verlangt, dass es auffällt.

## Nachvollziehbarkeit

Jeder gespeicherte Lauf (`backtest_runs`) verweist auf die Strategie-Version
(Hash der vollständigen Spezifikation), den Snapshot und den Hash der Serie,
die Konfiguration (Fingerprint), die Codeversion (Git-Commit) und seinen
Buy-and-Hold-Vergleich. Trades und Equity-Kurve liegen daneben. Die Tabellen
sind append-only: die Datenbank verweigert jede Änderung.

## Befehl

```powershell
taq backtest --strategy sma_cross --symbol SPY --start 2005-01-01
taq backtest --param fast=20 --param slow=100 --sizing vol_target --stop atr:3 --take percent:0.2
taq backtest --max-hold 60 --signal-prices price --no-save
```

## Benchmarks

Jeder Lauf wird gegen Buy & Hold derselben Instrumente (Dividenden
reinvestiert), den Index, die SMA-50/200-Baseline und Cash gerechnet; siehe
[STRATEGIEN.md](STRATEGIEN.md#benchmarks). Mehrere Instrumente:
`--symbol XLK,XLE,XLV`; jedes erhält höchstens 1/*N* des Kapitals.

## Ergebnisse

Die aktuellen Befunde aller Regeln stehen in
[STRATEGIEN.md](STRATEGIEN.md#stand-der-befunde). Kurz: keine Regel schlägt
Buy-and-Hold bei der Rendite, die Sharpe-Unterschiede liegen im Rauschen, alle
verkleinern den Drawdown. In-sample. Positiv ist ein Backtest erst, wenn er die
Walk-Forward-Validierung besteht — und auch dann ist er keine Garantie für die
Zukunft.

Eine frühere Fassung dieses Abschnitts verglich mit einem Buy-and-Hold ohne
Dividenden-Reinvestition (CAGR 9,8 % statt 10,6 %); dieser Benchmark war zu
schwach und ist ersetzt.
