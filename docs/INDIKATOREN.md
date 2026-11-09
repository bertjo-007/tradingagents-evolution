# Indikatoren und Marktregime

Alle Indikatoren sind reiner, deterministischer pandas-Code
(`tradingagents_quant/indicators/core.py`). Kein LLM berechnet oder schätzt
einen Indikatorwert; die Agentenschicht bekommt die gerechneten Werte und darf
sie nur interpretieren.

## Drei Garantien

1. **Kausal.** Der Wert an Tag *t* hängt nur von den Tagen bis *t* ab. Für
   jeden Indikator der Registry beweist das ein Trunkierungstest: der Wert,
   berechnet auf den Daten bis *t*, ist **bit-identisch** zum Wert an *t*,
   berechnet auf der ganzen Reihe.
2. **Keine Schätzung in der Aufwärmphase.** Solange das Fenster nicht voll
   ist, ist der Wert `NaN`. (`stockstats` liefert dort Werte aus halben
   Fenstern.)
3. **Nichts wird aufgefüllt.** Fehlt ein Hoch oder Tief, ist der True Range
   dieses Tages unbekannt — nicht der Range dessen, was übrig bleibt. Ein
   undefinierter Wert (z. B. %B bei Bandbreite null) bleibt `NaN`.

## Gleiche Zahlen wie die Agenten

Der Market Analyst des Upstreams bekommt seine Indikatoren über `stockstats`
(Whitelist in `tradingagents/dataflows/vendors/yahoo/market.py`). Für diese
Indikatoren gelten hier dieselben Formeln, und ein Test prüft die Gleichheit
auf 10⁻¹² relativ ab dem Ende der Aufwärmphase:

| Indikator | Schlüssel | Formel |
|---|---|---|
| SMA | `sma` | gleitender Mittelwert des Schlusskurses |
| EMA | `ema` | `ewm(span=n, adjust=True)` |
| RSI | `rsi` | Wilder-Glättung `ewm(alpha=1/n, adjust=True)` der Gewinne/Verluste; 50 ohne Bewegung |
| MACD | `macd` | EMA 12 − EMA 26, Signal = EMA 9 der Linie, Histogramm |
| Bollinger | `boll` | SMA 20 ± 2 · Standardabweichung (ddof = 1), dazu %B und Bandbreite |
| ATR | `atr` | Wilder-Glättung des True Range; erster Bar = eigener Range |
| VWMA | `vwma` | volumengewichteter Schnitt des typischen Preises (H+L+C)/3 |

Weitere, die die Agenten nicht sehen:

| Indikator | Schlüssel | Bedeutung |
|---|---|---|
| Historische Volatilität | `hv` | annualisierte Standardabweichung der Log-Renditen |
| Rate of Change | `roc` | Rendite über *n* Bars |
| Momentum 12-1 | `mom` | Rendite von vor 252 bis vor 21 Bars (lässt den letzten Monat aus) |
| On-Balance Volume | `obv` | kumuliertes, vorzeichenbehaftetes Volumen (nur Veränderungen vergleichen) |
| Volumen-Verhältnis | `volratio` | Volumen relativ zum 20-Bar-Schnitt |
| DMI / ADX | `dmi` | +DI, −DI und Wilders ADX = SMMA(DX, 14) |
| Donchian | `donchian` | höchstes Hoch / tiefstes Tief der letzten *n* Bars **inklusive heute** — ein Breakout vergleicht mit dem Kanal von gestern |

Der ADX folgt bewusst dem Lehrbuch (Wilder), nicht `stockstats` (dort
`EMA(DX, 6)`): die Agenten sehen ihn nicht, also muss nichts übereinstimmen.

Spaltennamen folgen den Parametern: `rsi_14`, `macd_12_26_9_signal`,
`boll_20_2_upper`. Abgerufen wird immer über
`tradingagents_quant.indicators.registry` (`compute`, `compute_many`,
`values_at`), nie direkt.

## Marktregime

`tradingagents_quant/regime.py` ordnet jedem Tag zwei unabhängige Labels zu:

**Trend:** `strong_up`, `up`, `strong_down`, `down`, `sideways`, `uncertain`

- *Struktur:* Kurs auf einer Seite des 200er-Durchschnitts **und** der 50er
  mindestens 2 Tages-Standardabweichungen vom 200er entfernt, auf derselben
  Seite. In der eigenen Volatilität gemessen, damit dieselbe Schwelle für TLT
  und QQQ gilt.
- *Stärke:* Struktur mit ADX ≥ 25 (Richtung aus den DI-Linien) ist `strong_*`,
  sonst `up`/`down`.
- Ohne Struktur: ADX < 20 ist `sideways`, alles andere `uncertain`.

**Volatilität:** `high`, `normal`, `low` — die heutige 20-Tage-Volatilität als
Perzentil ihrer eigenen letzten 252 Werte (≥ 80 % hoch, ≤ 20 % tief).

Vor genügend Historie (272 Bars) heisst das Label `unknown`.

### Warum nicht ADX allein

Ein ruhiger, stetiger Anstieg hat kleine Tagesranges und damit einen tiefen
ADX. Mit ADX allein galt SPY im Jahr 2017 (+18,5 %, eines der ruhigsten
Bullenjahre) an 56 % der Tage als «seitwärts». Mit der Strukturbedingung:

| Jahr | SPY | strong_up | up | strong_down | down | sideways | uncertain |
|---|---|---|---|---|---|---|---|
| 2008 | −37,7 % | 0 % | 0 % | 40 % | 51 % | 2 % | 8 % |
| 2013 | +26,4 % | 21 % | 75 % | 0 % | 0 % | 4 % | 0 % |
| 2015 | −0,8 % | 0 % | 57 % | 1 % | 7 % | 15 % | 19 % |
| 2017 | +18,5 % | 25 % | 75 % | 0 % | 0 % | 0 % | 0 % |
| 2022 | −19,9 % | 0 % | 10 % | 12 % | 46 % | 6 % | 26 % |
| 2024 | +24,0 % | 40 % | 60 % | 0 % | 0 % | 0 % | 0 % |

(Snapshot `snap-20261109T103320Z-a900e8b8`.) Die Richtung ist in keinem
klaren Jahr falsch.

### Was das Regime nicht ist

- **Keine Prognose.** Es beschreibt, was bis heute geschehen ist.
- **Es hinkt nach**, weil es auf gleitenden Durchschnitten beruht. Die erste
  Hälfte von 2015 gilt noch als `up`, weil die Struktur des Bullenmarkts 2014
  nachwirkt; der Bruch im August zeigt sich erst danach.
- **Ein Random Walk ohne Drift ist kein Seitwärtsmarkt.** Er bildet zufällig
  echte Trendphasen, und das Regime benennt sie zu Recht als Trend. Seitwärts
  heisst: keine Struktur und kein Richtungsdruck.

Jeder Backtest und jeder simulierte Trade speichert das Regime, in dem er lief,
damit sich Ergebnisse nach Regime aufteilen lassen.

## Befehl

```powershell
taq regime --symbol SPY            # heutiges Regime, Indikatorwerte, Anteile je Jahr
taq regime --symbol TLT --years 10
```
