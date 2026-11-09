# Validierung — Walk-Forward, Optimierung, Mehrfachtests

Ein positiver Backtest beweist wenig. Er kann Glück sein, Überanpassung an
genau diese Jahre, oder das Ergebnis davon, dass man so lange probiert hat, bis
etwas gut aussah. Dieser Teil des Systems soll genau das auseinanderhalten.

## Walk-Forward

Jede Faltung hat drei aufeinanderfolgende Abschnitte, getrennt durch ein
Embargo (Standard 21 Bars, damit weder eine offene Position noch das Gedächtnis
eines Indikators über die Grenze reicht):

```
train ──embargo── validation ──embargo── test
```

| Abschnitt | Was geschieht |
|---|---|
| train | Jeder Parametersatz des Gitters wird simuliert und nach seiner **Nachbarschaft** im Gitter bewertet (Mittel aus sich selbst und den direkten Nachbarn). Ein Plateau schlägt eine Spitze. |
| validation | Die besten `top_k` werden simuliert; der beste hier wird gewählt. |
| test | Der gewählte Satz wird **genau einmal** simuliert, Buy-and-Hold (Dividenden reinvestiert) daneben. |

Die Faltungen rücken um eine Testlänge vor; die Testabschnitte kacheln die
Historie lückenlos und ohne Überlappung zu einem Out-of-Sample-Ergebnis.
Jeder Abschnitt beginnt flach mit frischem simuliertem Kapital — Strategie und
Buy-and-Hold zahlen beide einen Einstieg.

**Nachweis, dass das Testfenster die Wahl nicht berührt**
(`test_the_test_data_cannot_touch_the_choice`): Alle Kurse nach dem
Validierungsende einer Faltung werden verändert (ein späterer Crash). Die Wahl
der Faltung und ihre Trainings- und Validierungswerte bleiben bit-genau gleich.
Zusätzlich werden jedem Abschnitt nur Daten bis zu seinem Ende übergeben.

## Das Urteil

Nach vorab festgelegten Kriterien (`WalkForwardDesign`, idealerweise aus einer
präregistrierten Hypothese):

| Urteil | Bedingung |
|---|---|
| `validated` | Out-of-Sample-Sharpe über der von Buy-and-Hold (plus Marge) **und** die Wahrscheinlichkeit, dass sie wirklich höher ist, über der Schwelle (Standard 0,95), nach allen Blicken auf diese Testdaten |
| `failed_validation` | sonst — mit jedem Grund, auch «Leistung vor allem im Training» (Out-of-Sample-Sharpe unter der Hälfte der In-Sample-Sharpe) |
| `inconclusive` | zu wenige Out-of-Sample-Trades oder eine zu kurze Reihe für eine Sharpe-Ratio |

### Welcher Test

Die Hypothesen behaupten eine **höhere Sharpe-Ratio als Buy-and-Hold**.
Geprüft wird deshalb die Differenz der beiden Sharpe-Ratios an denselben Tagen,
mit dem Test von Jobson und Korkie (1981) in der Korrektur von Memmel (2003),
der die Korrelation der beiden Renditereihen berücksichtigt. Monte-Carlo in
`tests/quant/test_deflate.py`: bei gleicher wahrer Sharpe hält er die
Strategie in rund 5 % der Fälle für besser — wie es bei einer Schwelle von
95 % sein muss.

Nicht geprüft wird:

- **Die Sharpe-Ratio gegen null.** Die besteht jede Strategie, die ungefähr im
  Aktienmarkt investiert ist — und Buy-and-Hold selbst. Sie wird nur zur
  Information gezeigt.
- **Die Information Ratio** (Rendite über Buy-and-Hold). Sie prüft mehr
  *Rendite*, nicht mehr Rendite *pro Risiko*, und würde eine Strategie
  verwerfen, die besser ist, aber weniger investiert. Auch sie wird nur zur
  Information gezeigt.

## Optimierung und Deflated Sharpe

Wer viele Varianten probiert, findet die beste aus Glück. Die **Deflated Sharpe
Ratio** (Bailey & López de Prado 2014, `sim/deflate.py`) gibt an, wie
wahrscheinlich eine Sharpe-Ratio echt ist, gegeben die Zahl der Versuche, die
Länge der Reihe sowie deren Schiefe und Wölbung.

Wie gross der Effekt ist (Monte-Carlo, `test_deflation_is_calibrated_under_the_null`):
Nimmt man die beste von 20 wertlosen Strategien, hält ein naiver Test sie in
**63 %** der Fälle für signifikant, die Deflated Sharpe in **0,3 %**.

Für jede Faltung zeigt der Bericht, wie viel der Trainings-Sharpe auf die
Auswahl aus dem Gitter zurückgeht.

## Das Experiment-Register

Jedes Experiment wird gespeichert (`experiments`), mit jedem bewerteten
Parametersatz auf jedem Abschnitt (`experiment_trials`) und jeder Testperiode
(`experiment_test_periods`). Alles append-only.

Ein späteres Experiment auf denselben Instrumenten und überlappenden
Testperioden wird abgewiesen, ausser es nimmt die Wiederverwendung
ausdrücklich in Kauf (`--allow-reuse`) — **auch mit einer anderen Regel**:
Out-of-Sample-Daten, die man schon angesehen hat, sind es nicht mehr, egal mit
welcher Strategie. Jeder frühere Blick zählt als Versuch und verschärft die
Schwelle (Šidák: die Wahrscheinlichkeit hoch die Zahl der Blicke).

## Präregistrierung

Siehe [hypothesen/README.md](hypothesen/README.md). H-001 bis H-003 wurden in
Commit `91c7f73` (2026-11-09 12:35:10 +01:00) festgeschrieben, bevor ein
Walk-Forward auf echten Daten lief.

## Ergebnisse der präregistrierten Hypothesen

Snapshot `snap-20261109T103320Z-a900e8b8`, Standardkosten, Faltungen train 6 /
validation 2 / test 2 Jahre, neun Faltungen, Out-of-Sample 2008-03-12 bis
2026-03-23 (18 Jahre):

| | H-001 Trendfolge SPY | H-002 Trendfolge 9 Sektoren | H-003 RSI(2) SPY |
|---|---|---|---|
| Varianten | 4 | 4 | 9 |
| OOS CAGR (Buy & Hold) | 7,0 % (11,2 %) | 5,8 % (10,3 %) | 2,4 % (11,2 %) |
| OOS Sharpe (Buy & Hold) | 0,64 (0,64) | 0,61 (0,63) | 0,31 (0,64) |
| OOS max. Drawdown (Buy & Hold) | −20 % (−51 %) | −18 % (−50 %) | −33 % (−51 %) |
| OOS-Trades | 72 | 691 | 155 |
| P(höhere Sharpe) | 0,50 | 0,47 | 0,08 |
| Blicke auf die Testdaten | 3 | 1 | 4 |
| nach allen Blicken | 0,13 | 0,47 | 0,00 |
| **Urteil** | **failed_validation** | **failed_validation** | **failed_validation** |
| Experiment | `wf-20261109T114214-738d4613` | `wf-20261109T114233-b12b7001` | `wf-20261109T114246-f409ae64` |

**Was das heisst:** Keine der drei Strategien hat out-of-sample eine höhere
Sharpe-Ratio als Buy-and-Hold. Die beiden Trendfolger sind risikobereinigt
gleichauf — sie halbieren den Drawdown und kosten dafür rund 4 Prozentpunkte
Rendite pro Jahr. Das ist ein Tausch, den man auch einfacher haben kann: weniger
Aktien halten. H-003 zeigt das klassische Bild der Überanpassung: Sharpe 0,65
im Training, 0,31 ausserhalb.

Für das ganze System heisst das: die Antwort auf die Leitfrage ist bisher
**nein** — mit verlässlichen Daten, realistischen Kosten und ehrlicher
Zählung der Versuche zeigt keine der geprüften Strategien einen
wiederholbaren Vorteil. Das ist ein Ergebnis, kein Versagen.

## Zwei Fehler in der eigenen Methode — und warum sie sichtbar bleiben

Das Register ist append-only; auch fehlerhafte Läufe bleiben darin.

1. **Erster Lauf von H-001** (`wf-20261109T113523-738d4613`, Urteil
   `validated`): Das Kriterium prüfte die Sharpe der Strategie **gegen null**
   und liess eine Sharpe-Differenz in der dritten Nachkommastelle als «höher»
   gelten. Buy-and-Hold selbst hätte diesen Test bestanden. Korrigiert in
   `b1ee20c`; der Regressionstest `test_buy_and_hold_can_never_validate_against_itself`
   verhindert, dass es wiederkommt.
2. **Zweiter Lauf von H-001** (`wf-20261109T113848-738d4613`): Die erste
   Korrektur testete die Information Ratio, also mehr *Rendite*, während die
   Hypothese eine höhere *Sharpe-Ratio* behauptet. Korrigiert in `a7b03f7`,
   vor den Läufen von H-002 und H-003.

Beide Korrekturen wurden committet, **bevor** der jeweils nächste Lauf
stattfand. Das endgültige Urteil über H-001 ist `wf-20261109T114214-738d4613`.
Die drei Blicke auf die SPY-Testdaten sind gezählt und haben die Schwelle für
H-003 verschärft — das ist der Preis der Fehler, und er wird bezahlt statt
versteckt. Am Ergebnis ändert er nichts: jede der drei Strategien scheitert
auch beim ersten Blick.

## Befehle

```powershell
taq walkforward --hypothesis docs/hypothesen/H-001.toml
taq walkforward --strategy sma_cross --symbol SPY --grid fast=20,50 --grid slow=100,200 `
                --train-years 6 --validation-years 2 --test-years 2
taq experiments
```
