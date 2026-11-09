# Die Agentenschicht und ihre Prüfung

Die Agenten des Upstreams (Analysten, Bull/Bear-Debatte, Research Manager,
Trader, Risiko-Debatte, Portfolio Manager) bleiben, wie sie sind. Der Fork fügt
drei Dinge hinzu, damit aus ihrer Arbeit nichts ungeprüft folgt und ihre Läufe
reproduzierbar und bezahlbar bleiben.

## 1. Das Risk Gate

Ein neuer Knoten nach dem Portfolio Manager
(`tradingagents/graph/setup.py` → `tradingagents_quant/bridge/gate.py`):

```
… Risiko-Debatte → Portfolio Manager → Risk Gate → Ende
```

Reiner Python-Code, **kein Modell**. Er prüft:

| Prüfung | Was |
|---|---|
| `data` | Es gibt einen Snapshot mit dem Instrument, mit einem Bar am oder vor dem Analysedatum, nicht veraltet, ohne Datenfehler. Nichts wird nachgeladen; das Tor ruft keinen Datenanbieter. |
| `structured` | Research Manager, Trader und Portfolio Manager haben **strukturiert** geantwortet. Ist einer still auf Prosa zurückgefallen (`agents/structured.py` fängt jeden Fehler ab), sind seine Zahlen nicht prüfbar: abgelehnt. |
| `rating` | Das Rating ist eines der fünf; `REVIEW` nimmt keine Position. |
| `consistency` | Empfehlung, Aktion und Rating zeigen in dieselbe Richtung (Buy/Overweight ↔ Buy, Hold ↔ Hold, Underweight/Sell ↔ Sell). |
| `entry_price` | Der Einstiegskurs des Traders liegt höchstens ±10 % (`[agents].max_entry_deviation`) vom letzten Schlusskurs. |
| `stop_loss` | Der Stop liegt bei einem Kauf unter dem Einstieg. |
| `price_target` | Das Kursziel liegt auf der Seite, die das Rating behauptet. |
| `risk_engine` | Die [Risiko-Engine](RISIKO.md) akzeptiert die simulierte Order — und sagt, in welcher Grösse. |

Ergebnis in neuen State-Feldern: `risk_verdict` (mit jeder Prüfung),
`sim_order` (die erlaubte simulierte Order, Label `PAPER TRADE — NOT REAL
MONEY`), `data_quality`, `risk_gate_report`; im Bericht als Abschnitt
«VI. Deterministic Risk Gate». Jeder Entscheid wird in `agent_decisions`
gespeichert (mit einem Hash über alle Berichte und Pläne), sofern eine
Forschungsdatenbank existiert.

Drei Zusagen, durch Tests gesichert:

- **Strikt additiv.** `final_rating` und `final_trade_decision` bleiben, was die
  Agenten geschrieben haben. Eine Ablehnung wirkt dort, wo Orders entstehen.
- **Kein Modell, kein Datenanbieter.** Der Upstream-Test, der die Zahl der
  LLM-Aufrufe und die Menge der Router-Aufrufe festschreibt, bleibt grün.
- **Wirft nie.** Jede Störung — auch unsinnige Eingaben oder eine verlangte
  Live-Ausführung — wird zur Ablehnung mit Grund.

Am echten Graphen geprüft (`tests/quant/test_gate.py`): ein strukturierter
Entscheid wird freigegeben, ein Prosa-Rückfall blockiert, ohne
Forschungsdaten wird zurückgehalten.

## 2. Aufzeichnen und Wiederholen

Jede Modellantwort kann in der Forschungsdatenbank aufgezeichnet werden
(`[agents].llm_cache`):

| Modus | Verhalten |
|---|---|
| `record` (Standard) | Eine schon aufgezeichnete Antwort auf denselben Prompt mit denselben Modelleinstellungen kommt aus dem Cache; jede neue wird gespeichert. |
| `replay` | Nur aus dem Cache. Fehlt eine Antwort, **stoppt der Lauf** (`ReplayMiss`), statt doch das Modell zu fragen. Kostet nichts, ist deterministisch. |
| `off` | Kein Cache. |

Ein Fund beim Bau: LangChain gibt eine Antwort aus dem Cache leicht verändert
zurück (`usage_metadata: {total_cost: 0}`). In der Werkzeugschleife eines
Analysten wird diese Antwort Teil des nächsten Prompts — mit einem Schlüssel
über den rohen Prompt wäre jeder Replay ab dem ersten Treffer gescheitert. Der
Schlüssel wird deshalb über den Gesprächsinhalt ohne die Metadaten der
Nachrichten gebildet (`canonical_prompt`).

Grenze: Ein Replay wiederholt die *Antworten* exakt. Ändert sich, was in einen
Prompt eingeht — andere Daten, ein neuer Eintrag im Gedächtnis-Log der Agenten
—, ist es ein anderer Prompt, und der Replay stoppt. Das ist gewollt.

## 3. Das Token-Budget

Ein Callback zählt die Tokens jedes Modellaufrufs, der nicht aus dem Cache
kam, pro Lauf und pro Tag (`llm_calls`). Ist das Kontingent erschöpft
(`max_tokens_per_run`, Standard 600 000; `max_tokens_per_day`, Standard
3 000 000), wird der **nächste** Aufruf abgewiesen (`BudgetExceeded`) — der Lauf
endet, das Programm nicht. Replays laufen unabhängig vom Budget.

## Einen Lauf starten

```powershell
taq ingest --symbols SPY                         # das Tor braucht Daten
taq agent --ticker SPY --date 2026-11-06 --yes   # kostet Geld: --yes ist Pflicht
taq agent --ticker SPY --date 2026-11-06 --cache replay   # derselbe Lauf, gratis
```

Modelle (`[agents]`, aus `tradingagents/llm_clients/model_catalog.py`):
`gemini-3.5-flash-lite` für Analysten, Debatten und Trader,
`gemini-3.8-flash` für Research und Portfolio Manager. Der Schlüssel wird als
`GOOGLE_API_KEY` erwartet; ist nur `GEMINI_API_KEY` gesetzt, übernimmt `taq agent`
ihn für die Dauer des Prozesses (er wird nirgends gespeichert oder ausgegeben).

## Warum die Agenten nicht rückwärts auf Rendite geprüft werden

Siehe [ARCHITEKTUR.md](ARCHITEKTUR.md#5-wie-ein-vorteil-nachgewiesen-wird):
News und Social sind nicht point-in-time archiviert, und ein Modell von 2026
kennt den Ausgang jedes historischen Jahres aus seinem Training. Ein
historischer Agenten-Backtest wäre Leckage durch die Gewichte. Die
Agentenschicht wird deshalb vorwärts gemessen; das gespeicherte Register
(`agent_decisions`) und die Aufzeichnungen (`llm_cache`) bauen das Material
dafür ab dem ersten Lauf auf.
