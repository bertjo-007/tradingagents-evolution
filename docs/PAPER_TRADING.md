# Paper-Trading

**PAPER TRADE — NOT REAL MONEY.** Es gibt keine Broker-Anbindung und keinen
Codepfad zu einer. Jede Ausführung, jeder Tagesabschluss und jede Kontoeröffnung
trägt dieses Label im Hauptbuch und in jeder Anzeige.

## Eine Engine, nicht zwei

Ein Paper-Konto ist das `Book` des Backtesters mit gespeichertem Zustand.
`taq paper run` lädt den Zustand, schreitet mit **derselben** `Book.step`-
Funktion über die Handelstage, die das Konto noch nicht gesehen hat, und
speichert den neuen Zustand. Zielgewichte, Kosten, Ausstiege und
Positionsgrössen kommen aus denselben Funktionen wie im Backtest; jede
Kauforder läuft zusätzlich durch die [Risiko-Engine](RISIKO.md).

Der Nachweis (`test_day_by_day_paper_trading_makes_the_backtests_fills`): eine
Historie wird in 32 Läufen durch ein Paper-Konto geschoben, davon 19 einzelne
Tage hintereinander — der Zustand wird also immer wieder gespeichert und
geladen. Die Ausführungen sind **bit-genau** die eines Backtests über dieselben
Tage mit derselben Risiko-Engine.

## Ablauf

```powershell
taq ingest                                   # frische Daten (Netz)
taq paper open --name trend-spy --strategy trend_following --symbol SPY --param window=200
taq paper run                                # alle Konten bis zum letzten Handelstag
taq paper status                             # Positionen, Kasse, Kapital, Drawdown
taq paper history --name trend-spy           # Ausführungen, Ablehnungen, Status
taq paper verify                             # Hash-Kette prüfen
```

Ein neues Konto entscheidet erstmals am Close des Starttags (Standard: der
letzte Handelstag im Snapshot) und handelt ab dem nächsten Open. Es arbeitet
mit dem letzten **abgerechneten** Tagesbar; Echtzeitdaten gibt es nicht.

Täglicher Rhythmus: nach Börsenschluss `taq ingest` und `taq paper run`.

## Das Hauptbuch

Alles, was einem Konto geschieht, steht in `paper_ledger`: Eröffnung,
Ausführungen, Ablehnungen, Risikoentscheide mit allen Prüfwerten, Dividenden,
Tagesabschlüsse, Statuswechsel, Umrechnungen. Jeder Eintrag enthält den Hash
des vorherigen:

```
hash = sha256(vorheriger hash | seq | Zeitpunkt | Art | Datum | Inhalt)
```

Die Datenbank verweigert jedes `UPDATE` und `DELETE`. Wer trotzdem etwas ändert
(etwa nach Entfernen eines Triggers), bricht die Kette, und `taq paper verify`
nennt den Eintrag. Jeder gespeicherte Zustand verweist auf den Kopf der Kette,
die ihn erzeugt hat. Alte Zustände bleiben erhalten; eine Strategieänderung
öffnet ein neues Konto, statt die Historie eines alten zu überschreiben.

## Status

| Status | Bedeutung |
|---|---|
| `active` | handelt |
| `suspended` | Drawdown hat `risk.max_drawdown` erreicht: Ausstiege laufen, keine neuen Positionen, bis ein Mensch es mit Begründung aufhebt (`taq paper resume`) |
| `halted` | die Daten passen nicht mehr zum Konto: nichts wird verarbeitet |

## Splits nach dem letzten Lauf

Yahoo passt bei einem Split die ganze Kurshistorie an. Ein Konto, das vor dem
Split zuletzt lief, hält dann die alte Stückzahl zu den alten Kursen, der neue
Snapshot hat halbierte Kurse. Beim nächsten Lauf vergleicht das Konto seinen
letzten Schlusskurs mit dem des neuen Snapshots am selben Tag:

- **Erklärt ein Split seither die Abweichung**, werden Stückzahl, Kurse, Stop,
  Ziel und offene Order auf die neue Basis umgerechnet, und das Hauptbuch hält
  die Umrechnung fest (`rebase`). Das Kapital bleibt stetig.
- **Erklärt nichts die Abweichung**, wird das Konto angehalten (`halted`).
  Weiterrechnen auf einer falschen Basis würde die Historie verfälschen.

## Grenzen

- Tagesbars, keine Intraday-Ausführung, keine Echtzeitdaten.
- Dividenden werden am Ex-Tag gutgeschrieben (nicht am Zahltag).
- Freie Kasse wird nicht verzinst.
- Die Ausführung zum nächsten Open mit Spread und Slippage ist eine Annahme;
  echte Ausführungen könnten besser oder schlechter sein.
