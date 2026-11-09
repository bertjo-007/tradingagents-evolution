# Das Forschungs-Dashboard

Eine Leseansicht des Forschungskerns im Browser. Jeder Wert, den das Dashboard
zeigt, steht in der Forschungsdatenbank oder wird beim Aufruf aus einem
gespeicherten Snapshot berechnet. Wo nichts gespeichert ist, sagt die Ansicht
das und nennt den Befehl, mit dem sich das ändert. Es gibt keine Beispielwerte,
keine Platzhalterzahlen und keine Demo-Konten.

## Starten

```powershell
pip install -e ".[dashboard]"     # FastAPI und uvicorn, nur für das Dashboard
taq dashboard                      # http://127.0.0.1:8000/
taq dashboard --port 8080
```

`taq dashboard` lauscht nur auf Loopback (`127.0.0.1`, `::1`, `localhost`). Das
Dashboard hat **keine Anmeldung**: Wer es erreicht, sieht alles, auch
Datenverzeichnis, Konfiguration und Agentenberichte. Eine andere Adresse
verlangt deshalb ausdrücklich `--allow-remote`.

Beim Start bringt der Befehl eine vorhandene Datenbank auf das aktuelle Schema.
Gibt es noch keine Datenbank, legt er auch keine an; die Übersicht zeigt dann
das Datenverzeichnis und `taq ingest` als ersten Schritt.

## Was es nicht kann

Das Dashboard **liest nur**:

- Die API kennt nur `GET`. Jede andere Methode, ob `POST`, `PUT`, `PATCH` oder
  `DELETE`, beantwortet sie mit 405, auch auf Pfaden, die es nicht gibt.
- Jede Anfrage öffnet die Datenbank im SQLite-Modus `mode=ro`. Ein Test liest
  jeden Endpunkt und prüft, dass die Datei danach Byte für Byte gleich ist.
- Es gibt keinen Knopf für einen Lauf, eine Order oder eine Änderung der
  Grenzen. Läufe startet die Befehlszeile (`taq ingest`, `taq backtest`,
  `taq walkforward`, `taq paper run`, `taq agent`).
- Antworten tragen `Cache-Control: no-store`, damit der Browser keinen alten
  Kontostand zeigt.

## Die Bereiche

| Bereich | Inhalt | Quelle |
|---|---|---|
| Überblick | Modi (Live-Handel aus, Order-Ausführung deaktiviert), Snapshot, Anzahl Backtests, Experimente, Validierungen, Paper-Konten und Agentenentscheide; die Leitfrage mit der ehrlichen Antwort; Kurs, Regime und Indikatoren des Benchmarks am letzten Bar | `status` |
| Research | Präregistrierte Hypothesen mit ihrem letzten Urteil; Marktregime je Instrument mit Anteilen je Jahr; die Entscheide der Agenten mit Bull gegen Bear, allen Berichten und dem Urteil des Risk Gate | `hypotheses`, `market`, `series`, `agents` |
| Strategien | Die Regelbibliothek und jede gespeicherte Strategie-Version mit der Zahl ihrer Backtests, Walk-Forward-Versuche und Paper-Konten | `rules`, `strategies` |
| Backtests | Walk-Forward-Experimente mit Urteil, Faltungen, Design und gezählten Versuchen; Backtests mit Kapitalkurve gegen jeden Benchmark, Drawdown, Vergleichstabelle, Regime, Trades und Herkunft (Snapshot, Serien-Hash, Strategie-Version, Konfiguration, Code) | `experiments`, `backtests` |
| Paper-Trading | Konten mit Kapitalverlauf, Positionen, Orders für den nächsten Open, jedem Ereignis des Hauptbuchs und dem Ergebnis der Hash-Ketten-Prüfung | `paper` |
| Risiko | Die Grenzen der Risiko-Engine, Exposure je Konto, jede abgewiesene Kauforder und jeder vom Risk Gate blockierte Agentenentscheid | `risk` |
| Agenten | Modelle, Aufzeichnungsmodus, Token-Budgets, Verbrauch je Tag (aufgezeichnete Antworten zählen nicht), alle Entscheide | `agents` |
| Einstellungen | Die wirksame Konfiguration mit Fingerprint, das Datenverzeichnis, die Datenquellen; für jeden API-Schlüssel nur, **ob** er gesetzt ist | `settings` |

Jede Ansicht mit simuliertem Geld trägt die Kennzeichnung
`PAPER TRADE — NOT REAL MONEY`.

## Die API

Unter `/api/` liegen dieselben Daten als JSON, beschrieben unter `/api/docs`.

| Pfad | Inhalt |
|---|---|
| `/api/status` | Modi, Snapshot, Zählungen, Markt des Benchmarks, Paper-Konten |
| `/api/snapshot` | Der neueste Snapshot mit Serien, Qualität und Fehlschlägen |
| `/api/market/{symbol}` | Regime und Indikatoren am letzten Bar, Regime-Anteile je Jahr |
| `/api/series/{symbol}` | Schlusskurse des Snapshots, auf höchstens 1500 Punkte ausgedünnt (der letzte bleibt immer) |
| `/api/rules`, `/api/strategies` | Regelbibliothek, Strategie-Versionen |
| `/api/backtests`, `/api/backtests/{run_id}` | Backtests, einer mit Kurven, Benchmarks und Trades |
| `/api/experiments`, `/api/experiments/{id}` | Walk-Forward-Experimente |
| `/api/hypotheses` | Die Hypothesen aus `docs/hypothesen/` mit ihrem letzten Experiment |
| `/api/paper`, `/api/paper/{name}` | Paper-Konten, eines mit Hauptbuch |
| `/api/risk` | Grenzen, Exposure, Abweisungen |
| `/api/agents`, `/api/agents/{decision_id}` | Agentenentscheide, einer mit allen Berichten |
| `/api/settings` | Konfiguration, Schlüssel nur als ja/nein |

Fehlt etwas, antwortet die API mit 404 oder einer leeren Liste. Zahlen, die
sich nicht berechnen lassen (NaN, ±∞), werden zu `null`.

## Sicherheit

- **Schlüssel:** `/api/settings` meldet für `GOOGLE_API_KEY`, `GEMINI_API_KEY`,
  `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `ALPHA_VANTAGE_API_KEY` und
  `FRED_API_KEY` nur `true` oder `false`. Ein Test setzt einen Schlüssel und
  prüft, dass sein Wert in keiner Antwort vorkommt.
- **Text der Agenten** ist Modellausgabe und damit fremder Text. Die Seite
  setzt ihn nie als HTML ein, sondern maskiert jedes Zeichen. Ein Bericht mit
  `<img onerror=…>` erscheint als Text und nicht als Element; das ist im
  Browser geprüft.
- **Nichts von aussen:** Seite, Stylesheet und Skript liegen im Paket. Es gibt
  kein CDN, keine Webfonts und keine Analytics. Ein Test sucht in den drei
  Dateien nach `http://` und `https://`.

## Aufbau

```
tradingagents_quant/api/
  app.py        FastAPI: Routen, nur GET, Datenbank je Anfrage read-only
  queries.py    Was angezeigt wird, aus Datenbank und Snapshots — sonst nichts
  static/
    index.html  Gerüst und Navigation
    app.css     Ein dunkles Thema, ein Akzent, Systemschriften
    app.js      Hash-Router, Ansichten, SVG-Diagramme; kein Build-Schritt
```

Es gibt keinen Build-Schritt und kein `node_modules`. Vite scheitert in
OneDrive (siehe Projektnotiz), und für eine Leseansicht braucht es kein
Framework. Die Diagramme sind eigenes SVG: Kapitalkurven logarithmisch, die
Strategie im Akzent, die Benchmarks grau und gestrichelt. Der Drawdown wird im
Browser aus der gespeicherten Kapitalkurve gerechnet, also aus denselben
Zahlen wie die Kennzahlen.

## Tests

`tests/quant/test_api.py` füllt ein temporäres Datenverzeichnis mit den echten
Befehlen: Ingest, Backtest, Walk-Forward, zwei Paper-Konten (eines davon von
der Risiko-Engine abgewiesen) und zwei Entscheide des Risk Gate. Dagegen prüfen
die Tests:

- Ohne Datenbank ist jede Ansicht leer, und ausserhalb der Konfiguration
  enthält sie keine einzige Zahl.
- Zählungen, Kurse, Kapital und Kurven stimmen mit Datenbank und Snapshot
  überein, und jede Kurve endet am letzten Bar.
- Ein Paper-Konto zeigt dasselbe Kapital wie `taq paper status`, und seine
  Hash-Kette ist intakt.
- Berichte der Agenten kommen unverändert an, auch feindseliger Text.
- Kein Schlüsselwert erscheint in einer Antwort, und jede schreibende Methode
  ergibt 405.
- Die Datenbankdatei ist nach allen Lesezugriffen unverändert.
- `taq dashboard` lauscht standardmässig auf Loopback, verweigert andere
  Adressen ohne `--allow-remote`, hebt ein altes Schema an und legt keine
  Datenbank an.

Die Tests laufen nur mit dem Extra `[dashboard]`. Ohne FastAPI werden sie
übersprungen. Die CI installiert das Extra.
