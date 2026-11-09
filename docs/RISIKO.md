# Risikomanagement

Die Risiko-Engine (`tradingagents_quant/risk/engine.py`) ist deterministischer
Code. Jede simulierte Kauforder — Einstieg, Aufstockung, Dividenden-
Reinvestition — läuft durch sie, bevor sie platziert wird. Verkäufe (Abbau von
Risiko) laufen nie durch sie; sie werden nie blockiert.

## Grundsätze

1. **Unbekanntes Risiko ist kein Trade.** Lässt sich ein Wert nicht berechnen
   (fehlende Historie, keine Volatilität, keine Korrelation), scheitert die
   Prüfung, und die Order wird abgelehnt.
2. **Nur ablehnen oder verkleinern, nie vergrössern.** Die Engine liefert die
   grösste Stückzahl, die alle Grenzen einhält; die Order bekommt höchstens
   das, was die Strategie wollte.
3. **Keine Tür für eine Einschätzung von aussen.** `RiskEngine.evaluate` nimmt
   nur den Auftrag und den Kontostand entgegen — keinen Konfidenzwert, keine
   Agentenmeinung, keinen Override. Ein Test prüft die Signatur.
4. **Grenzen nur aus der Konfiguration.** Kein Lauf, keine Strategie, kein Agent
   kann sie zur Laufzeit ändern. Wer sie ändert, ändert die Konfiguration —
   deren Fingerprint steht bei jedem Ergebnis.
5. **Jeder Entscheid ist nachvollziehbar.** Im Paper-Trading steht jedes
   Verdikt mit allen Prüfwerten im Hauptbuch.

## Die Prüfungen

| Prüfung | lehnt ab oder kürzt, wenn | Standard |
|---|---|---|
| `data` | kein brauchbarer Preis oder Kontowert | — |
| `drawdown` | das Konto weiter unter seinem Höchststand liegt als erlaubt | 25 % |
| `daily_loss` | der letzte Tag mehr verloren hat als erlaubt | 4 % |
| `open_positions` | eine neue Position die Höchstzahl überschreiten würde | 20 |
| `volatility` | die 20-Tage-Volatilität des Instruments zu hoch ist | 80 % p. a. |
| `liquidity` | der 20-Tage-Durchschnittsumsatz zu klein ist; grosse Orders werden auf einen Anteil davon gekürzt | 5 Mio. USD; 1 % |
| `position_weight` | die Position mehr als den erlaubten Anteil am Kapital hätte (gekürzt) | 35 % |
| `gross_exposure` | alle Positionen zusammen zu viel hätten (gekürzt) | 100 % |
| `correlation` | stark korrelierte Positionen (60-Tage-Korrelation ≥ 0,9) zusammen zu viel hätten (gekürzt) | 70 % |
| `stop` | ein Stop verlangt ist und fehlt, oder der Stop nicht unter dem Preis liegt | nicht verlangt |
| `risk_per_trade` | der Verlust bis zum Stop zu gross wäre (gekürzt) | 2 % |

Alle Grenzen stehen im Abschnitt `[risk]` der Konfiguration
(`config/taq.example.toml`).

Zwei Eigenschaften, die man kennen muss:

- **Eine einzelne Position ist ihre eigene korrelierte Gruppe.** Die wirksame
  Obergrenze einer Position ist darum das Minimum aus `max_position_weight`
  und `max_correlated_weight`.
- **Die Grenzen gelten zum Entscheidungskurs.** Entschieden wird am Close,
  ausgeführt am nächsten Open. Öffnet der Kurs höher, liegt der ausgeführte
  Wert leicht über der Grenze — so wie bei einer Lücke auch das Geld knapper
  wird.

## Folge des Standards für Ein-Instrument-Strategien

Mit `max_position_weight = 0.35` hält ein Paper-Konto mit einer
Ein-Instrument-Strategie höchstens 35 % in dieser Position, der Rest bleibt
Kasse — auch bei einem breiten ETF wie SPY. Will die Strategie mehr, versucht
die Engine an jedem Tag aufzustocken, und die Risiko-Engine lehnt an jedem Tag
ab; beides steht im Hauptbuch. Das ist gewollt: die Strategie sagt, was sie
will, das Risiko sagt, was sie darf. Wer für breite ETFs eine höhere
Einzelgrenze will, setzt sie in der Konfiguration.

## Konto in Schwierigkeiten

Erreicht der Drawdown eines Paper-Kontos `max_drawdown`, wird es
**suspendiert**: Ausstiege, Stops und Ziele laufen weiter, neue Positionen
nicht. Die Suspendierung hebt nur ein Mensch auf, mit Begründung, die ins
Hauptbuch kommt:

```powershell
taq paper resume --name trend-spy --reason "Crash geprüft, Grenzen unverändert"
```

## Im Backtest

Backtests laufen standardmässig ohne Risiko-Engine — sie messen die Strategie
selbst. Die Engine nimmt einen Filter entgegen (`order_filter`), und der
Paper-Replay-Test rechnet einen Backtest mit derselben Risiko-Engine, um zu
zeigen, dass Paper-Trading und Backtest dieselben Ausführungen ergeben.

## Dieselbe Arithmetik

Die Stückzahl zu einem Zielgewicht rechnet genau eine Funktion,
`sim.sizing.shares_for`. Engine und Risiko-Engine rufen beide sie auf; ein Test
prüft die Gleichheit.
