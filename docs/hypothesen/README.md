# Präregistrierte Hypothesen

Eine Hypothese wird **vor** dem ersten Lauf festgeschrieben und committet:
was behauptet wird, wie es geprüft wird, wie viele Varianten probiert werden
dürfen und was als Erfolg zählt. Der Commit-Zeitstempel belegt, dass das vor
dem Ergebnis feststand. Danach gilt die Datei; sie wird nicht mehr geändert.
Wer nach einem Ergebnis etwas anderes prüfen will, schreibt eine neue
Hypothese — und die zählt als weiterer Versuch auf denselben Daten.

## Warum

Wer zuerst rechnet und dann entscheidet, was ein Erfolg ist, findet fast immer
einen. Die beste von 20 wertlosen Strategien hält ein naiver Signifikanztest in
63 % der Fälle für echt (Monte-Carlo in `tests/quant/test_deflate.py`).
Festgelegte Kriterien und die Deflated Sharpe Ratio verhindern das.

## Format

TOML mit drei Abschnitten:

```toml
[hypothesis]
id = "H-001"                      # gleich dem Dateinamen
statement = "…"                   # die Behauptung in einem Satz
created = "2026-11-09"

[design]                          # Felder von WalkForwardDesign
rule = "trend_following"
symbols = ["SPY"]
grid = { window = [100, 150, 200, 250] }
train_years = 6
validation_years = 2
test_years = 2
embargo_bars = 21
objective = "sharpe"
top_k = 2

[criteria]                        # was als Erfolg zählt
margin_vs_benchmark = 0.0         # OOS-Sharpe muss Buy-and-Hold um so viel übertreffen
min_deflated_sharpe = 0.95
min_oos_trades = 15
max_variants = 4                  # mehr Varianten verweigert das Design
```

Ausführen:

```powershell
taq walkforward --hypothesis docs/hypothesen/H-001.toml
```

Das Experiment speichert den SHA-256 der Datei. Eine Hypothese, deren Datei
nach dem Lauf verändert wurde, ist an einem anderen Hash zu erkennen.

## Mehrfaches Testen

Hypothesen auf denselben Instrumenten und überlappenden Testperioden sehen
dieselben Out-of-Sample-Daten. Die zweite muss das ausdrücklich in Kauf
nehmen (`--allow-reuse`), und ihre Deflated Sharpe zählt jede frühere
Betrachtung als Versuch. Die Reihenfolge der Läufe ist deshalb Teil des
Protokolls: H-001, H-002, H-003, in dieser Reihenfolge.
