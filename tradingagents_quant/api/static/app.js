// TradingAgents Evolution, research dashboard. Read-only view of /api.
// Every value shown comes from the API; where the API has nothing, the view
// says so and how to change it. Text from the database (agent reports above
// all) is escaped, never interpreted as HTML.

const PAPER = "PAPER TRADE — NOT REAL MONEY"; // wording required by the brief
const view = document.getElementById("view");

// -- helpers ---------------------------------------------------------------

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const isNum = (v) => typeof v === "number" && Number.isFinite(v);
const nf = (d) => new Intl.NumberFormat("de-CH", { minimumFractionDigits: d, maximumFractionDigits: d });
const fmt = {
  pct: (v, signed = true) => isNum(v) ? (signed && v > 0 ? "+" : "") + nf(2).format(v * 100) + " %" : "n/a",
  share: (v) => isNum(v) ? nf(1).format(v * 100) + " %" : "n/a",
  num: (v, d = 2) => isNum(v) ? nf(d).format(v) : "n/a",
  money: (v) => isNum(v) ? nf(0).format(v) : "n/a",
  int: (v) => isNum(v) ? nf(0).format(v) : "n/a",
  qty: (v) => isNum(v) ? (Number.isInteger(v) ? nf(0) : nf(4)).format(v) : "n/a",
  any: (v) => isNum(v) ? new Intl.NumberFormat("de-CH", { maximumFractionDigits: 4 }).format(v) : "n/a",
};
const sign = (v) => (isNum(v) ? (v > 0 ? "pos" : v < 0 ? "neg" : "") : "");
const n = (v, kind = "num", cls = "") => `<span class="num ${cls}">${esc(fmt[kind](v))}</span>`;
const signed = (v, kind = "pct") => n(v, kind, sign(v));
const loss = (dd) => (isNum(dd) && dd > 0 ? n(-dd, "pct", "neg") : n(isNum(dd) ? 0 : dd, "pct"));
const short = (s, k = 12) => esc(String(s ?? "").slice(0, k));

const VERDICT = { validated: ["validiert", "good"], failed_validation: ["nicht bestanden", "bad"],
  inconclusive: ["unentschieden", "warn"] };
const STATUS = { active: ["aktiv", "good"], suspended: ["suspendiert", "warn"], halted: ["angehalten", "bad"],
  approved: ["freigegeben", "good"], blocked: ["blockiert", "bad"], ok: ["ok", "good"],
  warning: ["Warnungen", "warn"], error: ["Fehler", "bad"], complete: ["abgeschlossen", ""],
  failed: ["fehlgeschlagen", "bad"], partial: ["teilweise", "warn"] };
const tag = (key, map = STATUS) => {
  const [label, cls] = map[key] || [key ?? "n/a", ""];
  return `<span class="tag ${cls}">${esc(label)}</span>`;
};
const TREND = { strong_up: "starker Aufwärtstrend", up: "Aufwärtstrend", strong_down: "starker Abwärtstrend",
  down: "Abwärtstrend", sideways: "seitwärts", uncertain: "unklar", unknown: "unbekannt" };
const VOL = { high: "hohe Volatilität", normal: "normale Volatilität", low: "tiefe Volatilität", unknown: "unbekannt" };
const SIDE = { buy: "Kauf", sell: "Verkauf" };
const REASON = { entry: "Einstieg", exit_signal: "Ausstiegssignal", stop_loss: "Stop", take_profit: "Kursziel",
  time_stop: "Haltedauer", rebalance: "Rebalancing", open: "offen", dividend_reinvest: "Dividende reinvestiert" };
const reason = (r) => esc(REASON[r] || r);
const SEGMENT = { train: "Training", validation: "Validierung", test: "Test", benchmark: "Benchmark" };
const ORDER = { side: "Seite", shares: "Stück", reference_price: "Referenzkurs", weight: "Gewicht",
  reference_equity: "Referenzkapital", stop: "Stop", price_target: "Kursziel", fill: "Ausführung" };
const QUALITY = { symbol: "Instrument", trade_date: "Handelstag", snapshot_id: "Snapshot", series_quality: "Serienqualität",
  source: "Quelle", last_bar: "letzter Bar", last_close: "letzter Schluss", stale_days: "Tage alt" };
const show = (v) => (isNum(v) ? `<span class="num">${esc(fmt.any(v))}</span>` : val(v));
const labelled = (obj, names) => kv(Object.entries(obj || {}).filter(([k]) => k !== "label")
  .map(([k, v]) => [names[k] || k, k === "side" ? esc(SIDE[v] || v) : k === "weight" ? n(v, "share")
    : k === "fill" && v === "next open" ? "nächster Open" : show(v)]));
const KIND = { opened: "eröffnet", fill: "Ausführung", rejection: "abgewiesen", dividend: "Dividende",
  status: "Status", rebase: "Umrechnung" };
const BENCH = { buy_and_hold: "Buy & Hold", index: "Index", sma_baseline: "SMA 50/200", cash: "Cash" };

async function api(path) {
  const res = await fetch(`/api/${path}`, { headers: { Accept: "application/json" } });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
}

const empty = (text, cmd) =>
  `<div class="empty">${esc(text)}${cmd ? ` <code>${esc(cmd)}</code>` : ""}</div>`;
const header = (title, lead, paper = true) => `
  <div class="head"><h1>${esc(title)}</h1>${paper ? `<span class="paper-label">${esc(PAPER)}</span>` : ""}</div>
  ${lead ? `<p class="lead">${esc(lead)}</p>` : ""}`;
const figure = (k, v, s = "") =>
  `<div class="figure"><div class="k">${esc(k)}</div><div class="v">${v}</div>${s ? `<div class="s">${s}</div>` : ""}</div>`;
const table = (cols, rows, opts = {}) => {
  if (!rows.length) return empty(opts.empty || "Keine Einträge.", opts.cmd);
  const head = cols.map((c) => `<th class="${c.r ? "r" : ""}">${esc(c.h)}</th>`).join("");
  const body = rows.map((row) => {
    const href = opts.href ? opts.href(row) : null;
    const cells = cols.map((c) => `<td class="${c.r ? "r" : ""}">${c.v(row)}</td>`).join("");
    return `<tr${href ? ` class="link" data-href="${esc(href)}"` : ""}>${cells}</tr>`;
  }).join("");
  return `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
};
const kv = (pairs) => `<dl class="kv">${pairs.map(([k, v]) =>
  `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>`;
const val = (v) => esc(Array.isArray(v) ? v.join(", ")
  : typeof v === "object" && v !== null ? JSON.stringify(v) : String(v ?? "n/a"));

view.addEventListener("click", (e) => {
  const row = e.target.closest("tr.link");
  if (row && !e.target.closest("a")) location.hash = row.dataset.href;
});

// -- charts ------------------------------------------------------------------
// A line chart in SVG: shared time axis, optional log scale, hover readout.

function niceTicks(lo, hi, count = 5) {
  const span = hi - lo || Math.abs(hi) || 1;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) || 10 * mag;
  const ticks = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-12; t += step) ticks.push(t);
  return ticks;
}

function logTicks(lo, hi) {
  const ticks = [];
  for (let e = Math.floor(Math.log10(lo)) - 1; e <= Math.ceil(Math.log10(hi)) + 1; e++) {
    for (const m of [1, 2, 5]) {
      const t = m * 10 ** e;
      if (t >= lo && t <= hi) ticks.push(t);
    }
  }
  return ticks.length >= 3 ? ticks : niceTicks(lo, hi, 4);
}

function lineChart(series, opts = {}) {
  const W = 1000, H = opts.height || 260, L = 70, R = 14, T = 10, B = 26;
  const live = series.filter((s) => s.points && s.points.length > 1);
  const wrap = document.createElement("div");
  wrap.className = "chart";
  if (!live.length) { wrap.innerHTML = empty("Keine Datenpunkte."); return wrap; }
  const parsed = live.map((s) => ({ ...s, xs: s.points.map((p) => Date.parse(p[0])), ys: s.points.map((p) => p[1]) }));
  const allX = parsed.flatMap((s) => s.xs), allY = parsed.flatMap((s) => s.ys).filter(isNum);
  const x0 = Math.min(...allX), x1 = Math.max(...allX);
  let y0 = Math.min(...allY), y1 = Math.max(...allY);
  if (opts.zero) { y0 = Math.min(y0, 0); y1 = Math.max(y1, 0); }
  const log = opts.log && y0 > 0;
  const tf = log ? Math.log : (v) => v;
  const pad = (tf(y1) - tf(y0)) * 0.05 || 1;
  const ty0 = tf(y0) - pad, ty1 = tf(y1) + pad;
  const sx = (x) => L + ((x - x0) / (x1 - x0 || 1)) * (W - L - R);
  const sy = (y) => T + (1 - (tf(y) - ty0) / (ty1 - ty0)) * (H - T - B);
  const yfmt = opts.format || ((v) => fmt.num(v, 0));
  const yTicks = (log ? logTicks(Math.exp(ty0), Math.exp(ty1)) : niceTicks(ty0, ty1, 4));
  const years = [];
  const first = new Date(x0).getUTCFullYear(), last = new Date(x1).getUTCFullYear();
  if (x1 - x0 > 2 * 365.25 * 864e5) {
    const every = Math.max(1, Math.ceil((last - first + 1) / 9));
    for (let y = first; y <= last; y += every) {
      const t = Date.UTC(y, 0, 1);
      if (t >= x0 && t <= x1) years.push([t, y]);
    }
  } else {  // a short span: every month or every second or third one
    const months = (last - first) * 12 + new Date(x1).getUTCMonth() - new Date(x0).getUTCMonth();
    const every = months > 12 ? 3 : months > 6 ? 2 : 1;
    for (let m = 0; m <= months + 1; m++) {
      const d = new Date(Date.UTC(first, new Date(x0).getUTCMonth() + m, 1));
      if (d.getUTCMonth() % every === 0 && +d >= x0 && +d <= x1) years.push([+d, d.toISOString().slice(0, 7)]);
    }
  }
  if (!years.length) years.push([x0, new Date(x0).toISOString().slice(0, 10)]);
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(opts.label || "Diagramm")}">`;
  for (const t of yTicks) {
    svg += `<line class="grid-line" x1="${L}" x2="${W - R}" y1="${sy(t)}" y2="${sy(t)}"/>`;
    svg += `<g class="axis"><text x="${L - 8}" y="${sy(t) + 4}" text-anchor="end">${esc(yfmt(t))}</text></g>`;
  }
  for (const [t, label] of years) {
    svg += `<g class="axis"><text x="${sx(t)}" y="${H - 6}" text-anchor="middle">${esc(label)}</text></g>`;
  }
  for (const s of parsed.slice().reverse()) {
    const d = s.xs.map((x, i) => (isNum(s.ys[i]) ? `${i ? "L" : "M"}${sx(x).toFixed(1)},${sy(s.ys[i]).toFixed(1)}` : "")).join("");
    if (s.area) {
      svg += `<path d="${d}L${sx(s.xs[s.xs.length - 1])},${sy(0)}L${sx(s.xs[0])},${sy(0)}Z" fill="${s.color}" opacity="0.18"/>`;
    }
    svg += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.width || 1.6}"
             stroke-dasharray="${s.dash || ""}" vector-effect="non-scaling-stroke"/>`;
  }
  svg += `<line class="hover-line" x1="0" x2="0" y1="${T}" y2="${H - B}" stroke="var(--line-strong)" visibility="hidden"/>`;
  svg += `<rect x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="transparent"/></svg>`;
  wrap.innerHTML = svg + `<div class="tip"></div>`;
  const svgEl = wrap.querySelector("svg"), tip = wrap.querySelector(".tip"), vline = wrap.querySelector(".hover-line");
  const nearest = (xs, x) => { let lo = 0, hi = xs.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; xs[m] < x ? (lo = m) : (hi = m); }
    return Math.abs(xs[lo] - x) < Math.abs(xs[hi] - x) ? lo : hi; };
  svgEl.addEventListener("pointermove", (e) => {
    const box = svgEl.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * W;
    if (px < L || px > W - R) { tip.style.display = "none"; vline.setAttribute("visibility", "hidden"); return; }
    const x = x0 + ((px - L) / (W - L - R)) * (x1 - x0);
    const lines = parsed.map((s) => { const i = nearest(s.xs, x);
      return `<div><span style="color:${s.color}">${esc(s.name)}</span> ${esc(yfmt(s.ys[i]))}</div>`; });
    tip.innerHTML = `<div class="muted">${esc(new Date(x).toISOString().slice(0, 10))}</div>${lines.join("")}`;
    tip.style.display = "block";
    const left = ((px / W) * box.width);
    tip.style.left = `${Math.min(left + 12, box.width - tip.offsetWidth - 4)}px`;
    tip.style.top = "8px";
    vline.setAttribute("x1", px); vline.setAttribute("x2", px); vline.setAttribute("visibility", "visible");
  });
  svgEl.addEventListener("pointerleave", () => { tip.style.display = "none"; vline.setAttribute("visibility", "hidden"); });
  return wrap;
}

const legend = (series) => `<div class="legend">${series.filter((s) => s.points && s.points.length > 1)
  .map((s) => `<span><i style="border-color:${s.color};border-top-style:${s.dash ? "dashed" : "solid"}"></i>${esc(s.name)}</span>`)
  .join("")}</div>`;
const benchStyle = { buy_and_hold: ["var(--bench-1)", ""], index: ["var(--bench-1)", "6 4"],
  sma_baseline: ["var(--bench-2)", "2 3"], cash: ["var(--bench-2)", "8 3 2 3"] };
const drawdown = (points) => { let peak = -Infinity;
  return points.map(([d, v]) => { peak = Math.max(peak, v); return [d, v / peak - 1]; }); };
const mount = (id, node) => { const el = document.getElementById(id); if (el) el.replaceChildren(node); };

// -- views ---------------------------------------------------------------------

async function overview() {
  const s = await api("status");
  if (!s.database) {
    return header("Überblick", "Noch keine Forschungsdatenbank im Datenverzeichnis.") +
      empty(`Datenverzeichnis: ${s.data_dir}. Erster Schritt:`, "taq ingest");
  }
  const c = s.counts || {}, snap = s.snapshot, m = s.market;
  const answer = c.validated > 0
    ? `${c.validated} von ${c.experiments} Experiment(en) validiert.`
    : `Bisher nein: ${c.experiments ?? 0} Experiment(e), keines validiert.`;
  let html = header("Überblick", "Der Zustand des Systems, gelesen aus der Forschungsdatenbank. Kontostände und Orders sind simuliert, die Kurse stammen aus dem Snapshot.");
  html += `<div class="figures">
    ${figure("Snapshot", snap ? esc(snap.created_at.slice(0, 10)) : "keiner", snap ? `${snap.series.length} Serien, ${fmt.int(snap.bars)} Bars` : "")}
    ${figure("Backtests", n(c.backtests, "int"))}
    ${figure("Walk-Forward-Experimente", n(c.experiments, "int"), `${fmt.int(c.validated)} validiert`)}
    ${figure("Paper-Konten", n(c.paper_accounts, "int"))}
    ${figure("Agentenentscheide", n(c.agent_decisions, "int"))}
  </div>`;
  html += `<div class="grid g-wide" style="margin-top:16px">
    <div class="panel"><h3>Leitfrage</h3>
      <p style="margin:0 0 6px">Zeigt eine Strategie nach Kosten und Risiko einen wiederholbaren Vorteil gegenüber Buy-and-Hold?</p>
      <p style="margin:0"><b>${esc(answer)}</b> <a href="#/backtests">Experimente ansehen</a></p></div>
    <div class="panel"><h3>System</h3>${kv([
      ["Live-Handel", s.modes.live_trading_mode ? `<span class="neg">an</span>` : "aus"],
      ["Order-Ausführung", s.modes.live_order_execution === "disabled" ? "deaktiviert" : `<span class="neg">${esc(s.modes.live_order_execution)}</span>`],
      ["Paper-Trading", s.modes.paper_trading_mode ? "an" : "aus"],
      ["Konfiguration", `<span class="num">${esc(s.config_fingerprint)}</span>`],
      ["Code", `<span class="num">${esc(s.code_version)}</span>`],
      ["Schema", s.schema_current ? `v${esc(s.schema_version)}` : `<span class="warn">v${esc(s.schema_version)}, veraltet</span>`],
    ])}</div></div>`;
  if (m) {
    const ind = m.indicators || {};
    html += `<h2>Markt: ${esc(m.symbol)} am ${esc(m.as_of)}</h2><div class="grid g-wide">
      <div class="panel"><div id="ov-chart"><div class="skeleton tall"></div></div></div>
      <div class="panel">${kv([
        ["Schlusskurs", n(m.close)], ["Trend", esc(TREND[m.trend] || m.trend)],
        ["Volatilität", esc(VOL[m.volatility] || m.volatility)],
        ["SMA 50 / 200", `${n(ind.sma_50)} / ${n(ind.sma_200)}`], ["RSI 14", n(ind.rsi_14, "num")],
        ["ATR 14", n(ind.atr_14)], ["Volatilität 20T", n(ind.hv_20, "share")], ["ADX 14", n(ind.dmi_14_adx, "num")],
      ])}<p class="muted" style="margin:10px 0 0;font-size:12px">Regime und Indikatoren beschreiben, was war. Sie sind keine Prognose.</p></div></div>`;
  }
  html += `<h2>Paper-Konten</h2>` + table([
    { h: "Konto", v: (a) => esc(a.name) }, { h: "Status", v: (a) => tag(a.status) },
    { h: "Kapital", r: true, v: (a) => n(a.equity, "money") },
    { h: "seit Start", r: true, v: (a) => signed(a.equity / a.initial_cash - 1) },
    { h: "Drawdown", r: true, v: (a) => loss(a.drawdown) },
    { h: "Stand", v: (a) => esc(a.as_of || "noch nicht gelaufen") },
  ], s.paper || [], { empty: "Noch kein Paper-Konto. Keine Strategie hat die Validierung bestanden; ein Konto eröffnet", cmd: "taq paper open --name ...", href: (a) => `#/paper/${encodeURIComponent(a.name)}` });
  html += `<h2>Laufende Research-Aufgaben</h2>` + empty("Keine. Ein geplanter Research-Lauf ist noch nicht eingerichtet; Läufe werden von Hand gestartet.");
  view.innerHTML = html;
  if (m) {
    const ser = await api(`series/${encodeURIComponent(m.symbol)}`);
    mount("ov-chart", lineChart([{ name: m.symbol, points: ser.points, color: "var(--accent)" }],
      { log: true, label: `Schlusskurs ${m.symbol}`, format: (v) => fmt.num(v, 0) }));
  }
  return null;
}

async function research() {
  const [hyps, agents, snap] = await Promise.all([api("hypotheses"), api("agents"), api("snapshot")]);
  let html = header("Research", "Präregistrierte Hypothesen, Marktregime je Instrument und die Analysen der Agenten mit dem Urteil des Risk Gate.");
  html += `<h2>Präregistrierte Hypothesen</h2>` + table([
    { h: "ID", v: (h) => `<span class="num nowrap">${esc(h.id)}</span>` },
    { h: "Behauptung", v: (h) => esc(h.statement) },
    { h: "Urteil", v: (h) => h.latest ? tag(h.latest.verdict, VERDICT) : `<span class="muted nowrap">nicht gelaufen</span>` },
    { h: "P(höher), korr.", r: true, v: (h) => n(h.latest && h.latest.p_higher_sharpe) },
  ], hyps, { empty: "Keine Hypothese in docs/hypothesen/.", href: (h) => h.latest ? `#/experiments/${h.latest.experiment_id}` : null });
  const symbols = (snap && snap.series || []).map((x) => x.symbol);
  html += `<h2>Marktregime</h2>`;
  html += symbols.length ? `<div class="panel"><label class="muted" for="sym">Instrument </label>
      <select id="sym">${symbols.map((x) => `<option${x === "SPY" ? " selected" : ""}>${esc(x)}</option>`).join("")}</select>
      <div id="regime" style="margin-top:12px"></div></div>` : empty("Kein Snapshot.", "taq ingest");
  html += `<h2>Analysen der Agenten</h2>` + decisionsTable(agents.decisions);
  view.innerHTML = html;
  const sel = document.getElementById("sym");
  if (sel) { const show = () => regimePanel(sel.value); sel.addEventListener("change", show); show(); }
  return null;
}

async function regimePanel(symbol) {
  const box = document.getElementById("regime");
  box.innerHTML = `<div class="skeleton tall"></div>`;
  const [m, ser] = await Promise.all([api(`market/${encodeURIComponent(symbol)}`), api(`series/${encodeURIComponent(symbol)}`)]);
  if (!m) { box.innerHTML = empty("Keine Daten für dieses Instrument."); return; }
  const years = Object.entries(m.trend_by_year || {});
  const trends = ["strong_up", "up", "sideways", "uncertain", "down", "strong_down"];
  box.innerHTML = `<div class="grid g-wide"><div id="regime-chart"></div><div>${kv([
      ["Stand", esc(m.as_of)], ["Trend", esc(TREND[m.trend] || m.trend)], ["Volatilität", esc(VOL[m.volatility] || m.volatility)],
      ...Object.entries(m.indicators || {}).map(([k, v]) => [k, n(v, k.startsWith("hv_") ? "share" : "num")]) ])}</div></div>
    <h3 style="margin-top:16px">Anteil der Handelstage je Trend-Regime</h3>` +
    table([{ h: "Jahr", v: (r) => esc(r[0]) }, ...trends.map((t) => ({ h: TREND[t], r: true, v: (r) => n(r[1][t] || 0, "share") }))], years);
  mount("regime-chart", lineChart([{ name: symbol, points: ser.points, color: "var(--accent)" }],
    { log: true, label: `Schlusskurs ${symbol}`, height: 220 }));
}

function decisionsTable(decisions) {
  return table([
    { h: "Erstellt (UTC)", v: (d) => `<span class="num">${esc(d.created_at.slice(0, 16).replace("T", " "))}</span>` },
    { h: "Instrument", v: (d) => esc(d.symbol) }, { h: "Datum", v: (d) => `<span class="num">${esc(d.trade_date)}</span>` },
    { h: "Rating", v: (d) => esc(d.rating || "n/a") }, { h: "Risk Gate", v: (d) => tag(d.gate) },
    { h: "Grund", v: (d) => esc((d.reasons || [])[0] || "") },
  ], decisions || [], { empty: "Noch kein Agentenlauf. Ein Lauf kostet Geld und verlangt eine Bestätigung:",
    cmd: "taq agent --ticker SPY --date JJJJ-MM-TT --yes", href: (d) => `#/research/decision/${d.decision_id}` });
}

async function decision(id) {
  const d = await api(`agents/${encodeURIComponent(id)}`);
  if (!d) return header("Agentenentscheid", "") + empty("Nicht gefunden.");
  const r = d.reports || {};
  const section = (title, key) => r[key] ? `<details><summary>${esc(title)}</summary><pre class="report">${esc(r[key])}</pre></details>` : "";
  const checks = (d.verdict && d.verdict.checks || []).map((c) =>
    `<li><span class="tag ${c.passed ? "good" : "bad"}">${c.passed ? "ok" : "fail"}</span><span>${esc(c.name)}</span><span class="d">${esc(c.detail)}</span></li>`).join("");
  const order = d.sim_order && Object.keys(d.sim_order).length ? labelled(d.sim_order, ORDER) : empty("Keine simulierte Order.");
  return `<a class="back" href="#/research">Zurück zu Research</a>` +
    header(`${d.symbol} am ${d.trade_date}`, `Rating der Agenten: ${d.rating || "n/a"}. Entscheid ${d.decision_id}.`) +
    `<div class="grid g2"><div class="panel"><h3>Risk Gate ${tag(d.gate)}</h3><ul class="checks">${checks}</ul></div>
     <div class="panel"><h3>Simulierte Order</h3>${order}<h3 style="margin-top:14px">Daten</h3>${labelled(d.data_quality, QUALITY)}</div></div>
     <h2>Bull gegen Bear</h2><div class="grid g2">
       <div class="panel"><h3>Bull</h3>${r.bull_history ? `<pre class="report">${esc(r.bull_history)}</pre>` : empty("Nicht aufgezeichnet.")}</div>
       <div class="panel"><h3>Bear</h3>${r.bear_history ? `<pre class="report">${esc(r.bear_history)}</pre>` : empty("Nicht aufgezeichnet.")}</div></div>
     <h2>Berichte und Pläne</h2><div class="panel">
       ${section("Technische Analyse", "market_report")}${section("Sentiment", "sentiment_report")}
       ${section("News", "news_report")}${section("Fundamentaldaten", "fundamentals_report")}
       ${section("Research Manager", "investment_plan")}${section("Trader", "trader_investment_plan")}
       ${section("Risiko-Debatte", "risk_debate")}${section("Portfolio Manager", "final_trade_decision")}
       ${Object.keys(r).length ? "" : empty("Zu diesem Entscheid sind keine Berichte gespeichert.")}</div>`;
}

async function strategies() {
  const [rules, versions] = await Promise.all([api("rules"), api("strategies")]);
  let html = header("Strategien", "Jede Strategie ist eine deterministische Regel und eine eingefrorene Spezifikation. Jede Änderung ergibt eine neue Version.", false);
  html += `<h2>Regeln</h2>` + table([
    { h: "Regel", v: (r) => `<span class="num">${esc(r.rule)}</span>` }, { h: "Familie", v: (r) => esc(r.family) },
    { h: "Einstieg", v: (r) => esc(r.entry) }, { h: "Ausstieg", v: (r) => esc(r.exit) },
    { h: "Standard", v: (r) => `<span class="num">${esc(Object.entries(r.defaults).map(([k, v]) => `${k}=${v}`).join(", ") || "-")}</span>` },
  ], rules);
  html += `<h2>Gespeicherte Versionen</h2>` + table([
    { h: "Name", v: (v) => esc(v.name) }, { h: "Regel", v: (v) => esc(v.rule) },
    { h: "Version", v: (v) => `<span class="num">${short(v.version, 16)}</span>` },
    { h: "Parameter", v: (v) => `<span class="num">${esc(Object.entries(v.spec.params || {}).map(([k, x]) => `${k}=${x}`).join(", ") || "Standard")}</span>` },
    { h: "Instrumente", v: (v) => esc((v.spec.symbols || []).join(", ")) },
    { h: "Backtests", r: true, v: (v) => n(v.backtests, "int") }, { h: "Versuche", r: true, v: (v) => n(v.trials, "int") },
    { h: "Paper", r: true, v: (v) => n(v.paper_accounts, "int") },
  ], versions, { empty: "Noch keine Strategie-Version gespeichert.", cmd: "taq backtest" });
  view.innerHTML = html;
  return null;
}

async function backtests() {
  const [exps, runs] = await Promise.all([api("experiments"), api("backtests")]);
  let html = header("Backtests und Validierung", "Walk-Forward-Experimente beantworten, ob etwas ausserhalb der Stichprobe hält. Einzelne Backtests sind in-sample.");
  html += `<h2>Walk-Forward-Experimente</h2>` + table([
    { h: "Erstellt (UTC)", v: (e) => `<span class="num">${esc(e.created_at.slice(0, 16).replace("T", " "))}</span>` },
    { h: "Hypothese", v: (e) => esc(e.hypothesis || "-") }, { h: "Regel", v: (e) => esc(e.rule) },
    { h: "Instrumente", v: (e) => esc(e.symbols.length > 24 ? e.symbols.slice(0, 22) + "..." : e.symbols) },
    { h: "Urteil", v: (e) => e.status === "complete" ? tag(e.verdict, VERDICT) + (e.superseded_test ? ` <span class="muted">ersetzter Test</span>` : "") : tag(e.status) },
    { h: "OOS-Sharpe", r: true, v: (e) => n(e.oos_sharpe) }, { h: "Buy & Hold", r: true, v: (e) => n(e.benchmark_sharpe) },
    { h: "P(höher), korr.", r: true, v: (e) => n(e.p_higher_sharpe) }, { h: "Trades", r: true, v: (e) => n(e.oos_trades, "int") },
  ], exps, { empty: "Noch kein Experiment.", cmd: "taq walkforward --hypothesis docs/hypothesen/H-001.toml", href: (e) => `#/experiments/${e.experiment_id}` });
  html += `<h2>Backtests</h2>` + table([
    { h: "Erstellt (UTC)", v: (r) => `<span class="num">${esc(r.created_at.slice(0, 16).replace("T", " "))}</span>` },
    { h: "Strategie", v: (r) => esc(r.strategy) }, { h: "Instrumente", v: (r) => esc(r.symbol) },
    { h: "Zeitraum", v: (r) => `<span class="num">${esc(r.metrics.start)} bis ${esc(r.metrics.end)}</span>` },
    { h: "CAGR", r: true, v: (r) => signed(r.metrics.cagr) }, { h: "Sharpe", r: true, v: (r) => n(r.metrics.sharpe) },
    { h: "Max. DD", r: true, v: (r) => signed(r.metrics.max_drawdown) },
    { h: "vs Buy & Hold", r: true, v: (r) => signed(r.vs_benchmark.total_return_vs_benchmark) },
  ], runs, { empty: "Noch kein gespeicherter Backtest.", cmd: "taq backtest --strategy sma_cross --symbol SPY", href: (r) => `#/backtests/${r.run_id}` });
  view.innerHTML = html;
  return null;
}

async function backtestDetail(id) {
  const b = await api(`backtests/${encodeURIComponent(id)}`);
  if (!b) return header("Backtest", "") + empty("Nicht gefunden.");
  const m = b.metrics;
  const series = [{ name: b.strategy, points: b.equity, color: "var(--accent)", width: 2 },
    ...b.benchmarks.map((x) => ({ name: `${BENCH[x.name] || x.name} ${x.symbol}`, points: x.equity,
      color: (benchStyle[x.name] || ["var(--bench-1)"])[0], dash: (benchStyle[x.name] || [, ""])[1] }))];
  const bh = b.benchmarks.find((x) => x.name === "buy_and_hold");
  const ddSeries = [{ name: b.strategy, points: drawdown(b.equity), color: "var(--accent)", area: true },
    ...(bh ? [{ name: "Buy & Hold", points: drawdown(bh.equity), color: "var(--bench-1)", dash: "4 3" }] : [])];
  const cols = [{ name: b.strategy, m }, ...b.benchmarks.map((x) => ({ name: BENCH[x.name] || x.name, m: x.metrics }))];
  const rows = [["Gesamtrendite", "total_return", "pct"], ["CAGR", "cagr", "pct"], ["Sharpe", "sharpe", "num"],
    ["Sortino", "sortino", "num"], ["Max. Drawdown", "max_drawdown", "pct"], ["Trades", "trades", "int"],
    ["Investiert (Tage)", "exposure", "share"], ["Kosten", "total_costs", "money"]];
  const regimes = Object.entries(b.regimes || {}).filter(([, r]) => r.exposure > 0).sort((a, c) => c[1].bars - a[1].bars);
  let html = `<a class="back" href="#/backtests">Zurück zu den Backtests</a>` +
    header(`${b.strategy} auf ${b.symbol}`, `${m.start} bis ${m.end}, ${fmt.int(m.bars)} Handelstage. In-sample: ein Parametersatz, auf derselben Periode gemessen.`);
  html += `<div class="figures">${figure("Gesamtrendite", signed(m.total_return))}${figure("CAGR", signed(m.cagr))}
    ${figure("Sharpe", n(m.sharpe))}${figure("Sortino", n(m.sortino))}${figure("Max. Drawdown", signed(m.max_drawdown))}
    ${figure("Trades", n(m.trades, "int"), `Treffer ${fmt.share(m.win_rate)}`)}${figure("Kosten", n(m.total_costs, "money"))}
    ${figure("Dividenden", n(m.dividends, "money"))}</div>`;
  html += `<h2>Kapital gegen Benchmarks</h2><div class="panel">${legend(series)}<div id="eq"></div></div>`;
  html += `<h2>Drawdown</h2><div class="panel">${legend(ddSeries)}<div id="dd"></div></div>`;
  html += `<h2>Vergleich</h2><div class="panel">` + table([{ h: "", v: (r) => esc(r[0]) },
    ...cols.map((c) => ({ h: c.name, r: true, v: (r) => r[2] === "pct" ? signed(c.m[r[1]]) : n(c.m[r[1]], r[2]) }))], rows) + `</div>`;
  html += `<div class="grid g2"><div><h2>Nach Regime (während investiert)</h2>` + table([
      { h: "Regime", v: ([k]) => esc(k.split("/").map((p, i) => (i ? VOL : TREND)[p] || p).join(", ")) },
      { h: "Tage", r: true, v: ([, r]) => n(r.share_of_bars, "share") }, { h: "investiert", r: true, v: ([, r]) => n(r.exposure, "share") },
      { h: "Rendite", r: true, v: ([, r]) => signed(r.compounded_return) }], regimes.slice(0, 10)) + `</div>
    <div><h2>Herkunft</h2><div class="panel">${kv([["Snapshot", short(b.snapshot_id, 40)], ["Serien-Hash", short(b.series_sha256, 16)],
      ["Strategie-Version", short(b.version, 16)], ["Konfiguration", short(b.config_fingerprint, 16)], ["Code", esc(b.code_version)], ["Lauf", esc(b.run_id)]])}
      ${(b.notes || []).map((x) => `<p class="muted" style="font-size:12px;margin:10px 0 0">${esc(x)}</p>`).join("")}</div></div></div>`;
  const trades = (b.trades || []).slice(-40).reverse();
  html += `<h2>Letzte Trades <span class="muted">(${fmt.int((b.trades || []).length)} insgesamt)</span></h2>` + table([
    { h: "Einstieg", v: (t) => `<span class="num">${esc(t.entry_date)}</span>` }, { h: "Ausstieg", v: (t) => `<span class="num">${esc(t.exit_date || "offen")}</span>` },
    { h: "Grund", v: (t) => reason(t.exit_reason) }, { h: "Tage", r: true, v: (t) => n(t.holding_bars, "int") },
    { h: "Rendite", r: true, v: (t) => signed(t.return_pct) }, { h: "Ergebnis", r: true, v: (t) => n(t.pnl, "money", sign(t.pnl)) },
  ], trades);
  view.innerHTML = html;
  mount("eq", lineChart(series, { log: true, label: "Kapital", format: (v) => fmt.money(v) }));
  mount("dd", lineChart(ddSeries, { zero: true, height: 180, label: "Drawdown", format: (v) => fmt.pct(v, false) }));
  return null;
}

async function experimentDetail(id) {
  const e = await api(`experiments/${encodeURIComponent(id)}`);
  if (!e) return header("Experiment", "") + empty("Nicht gefunden.");
  const s = e.summary || {}, d = (e.design || {}).design || {}, h = (e.design || {}).hypothesis;
  const defl = s.deflated || {};
  let html = `<a class="back" href="#/backtests">Zurück zur Validierung</a>` +
    header(`${d.rule} auf ${(d.symbols || []).join(", ")}`, h ? `${h.id}: ${h.statement}` : "Ohne präregistrierte Hypothese.", false);
  html += `<div class="panel"><h3>Urteil ${e.status === "complete" ? tag(e.verdict, VERDICT) : tag(e.status)}</h3>
    ${(s.reasons || []).map((r) => `<p style="margin:4px 0">${esc(r)}</p>`).join("") || `<p class="muted" style="margin:4px 0">Alle Kriterien erfüllt.</p>`}
    ${defl.p_deflated === undefined ? `<p class="warn" style="margin:8px 0 0">Mit einem inzwischen ersetzten Test gerechnet (${esc(defl.test || "Sharpe gegen null")}); siehe docs/VALIDIERUNG.md.</p>` : ""}</div>`;
  const o = s.oos || {}, b = s.benchmark_oos || {};
  html += `<h2>Ausserhalb der Stichprobe</h2><div class="figures">
    ${figure("Sharpe", n(o.sharpe), `Buy & Hold ${fmt.num(b.sharpe)}`)}${figure("CAGR", signed(o.cagr), `Buy & Hold ${fmt.pct(b.cagr)}`)}
    ${figure("Max. Drawdown", signed(o.max_drawdown), `Buy & Hold ${fmt.pct(b.max_drawdown)}`)}
    ${figure("P(höhere Sharpe)", n(defl.p), `nach ${fmt.int(defl.trials)} Blick(en): ${fmt.num(defl.p_deflated)}`)}
    ${figure("Trades", n(s.oos_trades, "int"))}${figure("Jahre", n(o.years, "num"))}</div>`;
  html += `<h2>Faltungen</h2>` + table([
    { h: "", v: (f) => n(f.n + 1, "int") }, { h: "Testperiode", v: (f) => `<span class="num">${esc(f.periods.test[0])} bis ${esc(f.periods.test[1])}</span>` },
    { h: "Gewählt", v: (f) => `<span class="num">${esc(Object.entries(f.chosen).map(([k, v]) => `${k}=${v}`).join(", ") || "Standard")}</span>` },
    { h: "Training", r: true, v: (f) => n(f.train_objective) }, { h: "Validierung", r: true, v: (f) => n(f.validation_objective) },
    { h: "Test", r: true, v: (f) => n(f.test.sharpe) }, { h: "Buy & Hold", r: true, v: (f) => n(f.benchmark.sharpe) },
  ], s.folds || []);
  html += `<div class="grid g2"><div><h2>Design</h2><div class="panel">${kv([
      ["Gitter", val(d.grid)], ["fest", val(d.fixed_params)], ["Fenster", esc(`${d.train_years} / ${d.validation_years} / ${d.test_years} Jahre, Embargo ${d.embargo_bars} Tage`)],
      ["Ziel", esc(d.objective)], ["Marge", n(d.margin_vs_benchmark)], ["Schwelle", n(d.min_deflated_sharpe)],
      ["Hypothesen-Hash", e.hypothesis_sha256 ? short(e.hypothesis_sha256, 16) : "keiner"]])}</div></div>
    <div><h2>Versuche</h2><div class="panel">${kv(Object.entries(e.trials || {}).map(([k, v]) => [SEGMENT[k] || k, n(v, "int")]))}
      ${(s.notes || []).map((x) => `<p class="muted" style="font-size:12px;margin:10px 0 0">${esc(x)}</p>`).join("")}</div></div></div>`;
  return html;
}

async function paper() {
  const accounts = await api("paper");
  let html = header("Paper-Trading", "Konten mit simuliertem Geld. Dieselbe Engine wie im Backtest, jede Kauforder durch die Risiko-Engine, jedes Ereignis im Hauptbuch.");
  html += table([
    { h: "Konto", v: (a) => esc(a.name) }, { h: "Strategie", v: (a) => esc(a.strategy) },
    { h: "Instrumente", v: (a) => esc(a.symbols.join(", ")) }, { h: "Status", v: (a) => tag(a.status) },
    { h: "Kapital", r: true, v: (a) => n(a.equity, "money") }, { h: "seit Start", r: true, v: (a) => signed(a.equity / a.initial_cash - 1) },
    { h: "Kasse", r: true, v: (a) => n(a.cash, "money") }, { h: "Stand", v: (a) => esc(a.as_of || "noch nicht gelaufen") },
  ], accounts || [], { empty: "Noch kein Paper-Konto. Keine Strategie hat bisher die Validierung bestanden. Ein Konto eröffnen und weiterführen:",
    cmd: "taq paper open --name ... && taq paper run", href: (a) => `#/paper/${encodeURIComponent(a.name)}` });
  view.innerHTML = html;
  return null;
}

async function paperDetail(name) {
  const a = await api(`paper/${encodeURIComponent(name)}`);
  if (!a) return header("Paper-Konto", "") + empty("Nicht gefunden.");
  const positions = Object.entries(a.positions || {});
  const events = (a.events || []).slice().reverse();
  let html = `<a class="back" href="#/paper">Zurück zum Paper-Trading</a>` +
    header(a.name, `${a.strategy} auf ${a.symbols.join(", ")}, Stand ${a.as_of || "noch nicht gelaufen"}.`);
  html += `<div class="figures">${figure("Kapital", n(a.equity, "money"), `Start ${fmt.money(a.initial_cash)}`)}
    ${figure("seit Start", signed(a.equity / a.initial_cash - 1))}${figure("Kasse", n(a.cash, "money"))}
    ${figure("Drawdown", loss(a.drawdown))}${figure("Status", tag(a.status))}
    ${figure("Hauptbuch", a.ledger_intact ? `<span class="pos">intakt</span>` : `<span class="neg">Kette gebrochen</span>`)}</div>`;
  html += `<h2>Kapitalverlauf</h2><div class="panel" id="paper-eq"></div>`;
  html += `<div class="grid g2"><div><h2>Positionen</h2>` + table([
      { h: "Instrument", v: ([s]) => esc(s) }, { h: "Stück", r: true, v: ([, p]) => n(p.shares, "qty") },
      { h: "Kurs", r: true, v: ([, p]) => n(p.last_close) }, { h: "Wert", r: true, v: ([, p]) => n(p.value, "money") },
      { h: "Gewicht", r: true, v: ([, p]) => n(p.weight, "share") }, { h: "Stop", r: true, v: ([, p]) => n(p.stop) }],
    positions, { empty: "Keine offene Position." }) +
    `</div><div><h2>Offene Orders</h2>` + table([{ h: "Instrument", v: ([s]) => esc(s) }, { h: "Seite", v: ([, o]) => esc(SIDE[o.side] || o.side) },
      { h: "Stück", r: true, v: ([, o]) => n(o.shares, "qty") }, { h: "Grund", v: ([, o]) => reason(o.reason) }],
    Object.entries(a.pending || {}), { empty: "Keine Order für den nächsten Open." }) + `</div></div>`;
  html += `<h2>Ereignisse</h2>` + table([
    { h: "Datum", v: (e) => `<span class="num">${esc(e.date || "")}</span>` }, { h: "Art", v: (e) => esc(KIND[e.kind] || e.kind) },
    { h: "", v: (e) => e.kind === "fill" ? esc(`${SIDE[e.side] || e.side} ${fmt.qty(e.shares)} ${e.symbol} zu ${fmt.num(e.price, 4)} (${REASON[e.reason] || e.reason})`)
      : e.kind === "rejection" ? esc(`${e.symbol}: ${e.reason}, ${e.detail}`) : e.kind === "dividend" ? esc(`Dividenden ${fmt.money(e.amount)}`)
      : e.kind === "opened" ? esc(`eröffnet mit ${fmt.money(e.initial_cash)} simulierter Kasse`) : esc(e.reason || e.status || "") },
  ], events.slice(0, 120), { empty: "Noch keine Ereignisse." });
  view.innerHTML = html;
  mount("paper-eq", lineChart([{ name: a.name, points: a.equity_curve, color: "var(--accent)" }],
    { label: "Kapital", format: (v) => fmt.money(v) }));
  return null;
}

const LIMITS = { max_gross_exposure: ["Brutto-Exposure", "share"], max_position_weight: ["Einzelposition", "share"],
  max_open_positions: ["offene Positionen", "int"], max_drawdown: ["Drawdown, ab dem nichts Neues kommt", "share"],
  max_daily_loss: ["Tagesverlust", "share"], max_volatility: ["Volatilität des Instruments", "share"],
  min_dollar_volume: ["Tagesumsatz mindestens (USD)", "money"], max_adv_fraction: ["Order höchstens vom Tagesumsatz", "share"],
  max_correlated_weight: ["korrelierte Positionen zusammen", "share"], correlation_threshold: ["Korrelation ab", "num"],
  correlation_window: ["Korrelationsfenster (Tage)", "int"], require_stop: ["Stop verlangt", "bool"],
  max_risk_per_trade: ["Verlust bis zum Stop je Trade", "share"] };

async function risk() {
  const r = await api("risk");
  let html = header("Risikomanagement", "Die Grenzen der deterministischen Risiko-Engine und was sie abgewiesen hat. Kein Lauf und kein Agent kann diese Grenzen ändern.");
  html += `<div class="grid g2"><div><h2>Grenzen</h2><div class="panel">${kv(Object.entries(r.limits).map(([k, v]) => {
    const [label, kind] = LIMITS[k] || [k, "num"];
    return [label, kind === "bool" ? (v ? "ja" : "nein") : n(v, kind)]; }))}</div></div>
    <div><h2>Konten</h2>` + table([{ h: "Konto", v: (a) => esc(a.name) }, { h: "Status", v: (a) => tag(a.status) },
      { h: "Exposure", r: true, v: (a) => n(a.gross_exposure, "share") }, { h: "grösste Position", r: true, v: (a) => n(a.largest_position, "share") },
      { h: "Drawdown", r: true, v: (a) => loss(a.drawdown) }],
      r.accounts, { empty: "Kein Paper-Konto." }) + `</div></div>`;
  html += `<h2>Abgewiesene Kauforders</h2>` + table([{ h: "Datum", v: (x) => `<span class="num">${esc(x.date)}</span>` },
    { h: "Konto", v: (x) => esc(x.account) }, { h: "Instrument", v: (x) => esc(x.symbol) },
    { h: "Grund", v: (x) => esc((x.reasons || []).join("; ")) }], r.refusals, { empty: "Keine Abweisung." });
  html += `<h2>Vom Risk Gate blockierte Agentenentscheide</h2>` + table([{ h: "Datum", v: (x) => `<span class="num">${esc(x.date)}</span>` },
    { h: "Instrument", v: (x) => esc(x.symbol) }, { h: "Rating", v: (x) => esc(x.rating || "n/a") },
    { h: "Grund", v: (x) => esc((x.reasons || [])[0] || "") }], r.gate_refusals,
    { empty: "Keine.", href: (x) => `#/research/decision/${x.decision_id}` });
  view.innerHTML = html;
  return null;
}

async function agents() {
  const a = await api("agents");
  const s = a.settings;
  let html = header("Agenten", "Die LLM-Agentenschicht: Modelle, Aufzeichnung, Verbrauch und jedes Urteil des Risk Gate.", false);
  html += `<div class="grid g2"><div><h2>Einstellungen</h2><div class="panel">${kv([
      ["Anbieter", esc(s.llm_provider)], ["schnelle Stufe", `<span class="num">${esc(s.quick_model)}</span>`],
      ["tiefe Stufe", `<span class="num">${esc(s.deep_model)}</span>`], ["Aufzeichnung", esc(s.llm_cache)],
      ["Tokens je Lauf", n(s.max_tokens_per_run, "int")], ["Tokens je Tag", n(s.max_tokens_per_day, "int")],
      ["Analysten", esc((s.analysts || []).join(", "))], ["aufgezeichnete Antworten", n(a.cache_entries, "int")]])}</div></div>
    <div><h2>Verbrauch je Tag</h2>` + table([{ h: "Tag", v: (u) => `<span class="num">${esc(u.day)}</span>` },
      { h: "Aufrufe", r: true, v: (u) => n(u.calls, "int") }, { h: "aus Aufzeichnung", r: true, v: (u) => n(u.cached, "int") },
      { h: "Tokens", r: true, v: (u) => n(u.tokens, "int") }], a.usage, { empty: "Noch kein Modellaufruf." }) + `</div></div>`;
  html += `<h2>Entscheide</h2>` + decisionsTable(a.decisions);
  view.innerHTML = html;
  return null;
}

async function settings() {
  const s = await api("settings");
  const c = s.config;
  const section = (title, obj) => `<div class="panel"><h3>${esc(title)}</h3>${kv(Object.entries(obj).map(([k, v]) => [k, val(v)]))}</div>`;
  let html = header("Einstellungen", "Die wirksame Konfiguration. Ändern lässt sie sich nur in der Konfigurationsdatei; jede Änderung ändert den Fingerprint.", false);
  html += `<div class="panel">${kv([["Fingerprint", `<span class="num">${esc(s.fingerprint)}</span>`], ["Datenverzeichnis", esc(s.data_dir)],
    ["Kursdaten", esc(s.data_provider.prices)], ["Daten der Agenten", esc(s.data_provider.agents)], ["Research-Frequenz", esc(s.research_frequency)]])}</div>`;
  html += `<h2>Zugangsdaten</h2><div class="panel">${kv(Object.entries(s.credentials).map(([k, v]) =>
    [k, v ? `<span class="pos">gesetzt</span>` : `<span class="muted">nicht gesetzt</span>`]))}
    <p class="muted" style="font-size:12px;margin:10px 0 0">Gezeigt wird nur, ob ein Schlüssel gesetzt ist, nie sein Wert.</p></div>`;
  html += `<h2>Konfiguration</h2><div class="grid g2">${section("Modi", c.modes)}${section("Universum", c.universe)}
    ${section("Daten", c.data)}${section("Kosten", c.costs)}${section("Backtest", c.backtest)}${section("Risiko", c.risk)}${section("Agenten", c.agents)}</div>`;
  return html;
}

// -- router --------------------------------------------------------------------

const ROUTES = [
  [/^#\/overview$/, overview, "overview"], [/^#\/research$/, research, "research"],
  [/^#\/research\/decision\/(.+)$/, decision, "research"], [/^#\/strategies$/, strategies, "strategies"],
  [/^#\/backtests$/, backtests, "backtests"], [/^#\/backtests\/(.+)$/, backtestDetail, "backtests"],
  [/^#\/experiments\/(.+)$/, experimentDetail, "backtests"], [/^#\/paper$/, paper, "paper"],
  [/^#\/paper\/(.+)$/, paperDetail, "paper"], [/^#\/risk$/, risk, "risk"], [/^#\/agents$/, agents, "agents"],
  [/^#\/settings$/, settings, "settings"],
];

async function route() {
  const hash = location.hash || "#/overview";
  const match = ROUTES.map(([re, fn, nav]) => [hash.match(re), fn, nav]).find(([m]) => m);
  if (!match) { location.hash = "#/overview"; return; }
  const [m, fn, nav] = match;
  document.querySelectorAll(".nav a").forEach((a) => {
    if (a.getAttribute("href") === `#/${nav}`) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  view.innerHTML = `<div class="skeleton" style="width:28%"></div><div class="skeleton tall"></div><div class="skeleton" style="width:60%"></div>`;
  try {
    const html = await fn(...m.slice(1).map(decodeURIComponent));
    if (typeof html === "string") view.innerHTML = html;
  } catch (err) {
    view.innerHTML = `<div class="error">Konnte nicht geladen werden: ${esc(err.message)}. Läuft <code>taq dashboard</code> noch?</div>`;
  }
  view.focus({ preventScroll: true });
}

async function modes() {
  try {
    const s = await api("status");
    document.getElementById("modes").innerHTML = s.modes.live_trading_mode
      ? `Live-Handel: <span class="neg">an</span>` : `Live-Handel: <b>aus</b>`;
  } catch { /* the view shows the error */ }
}

window.addEventListener("hashchange", route);
modes();
route();
