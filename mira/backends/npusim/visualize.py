"""Interactive HTML view of an MNPU-1 simulation.

    mira emit prog.mira -t sim -s html -o prog.html                     # one run
    mira emit prog.mira -t sim -s html --compare no-double-buffer       # side by side with a variant

The page is self-contained (inline CSS/JS, no network) and shows, per run:
  * a stats card: time, engine utilization, MAC throughput, DRAM traffic;
  * a timeline with one lane per IR op and per engine (DMA, MXU, VPU). Each instruction is a bar;
    DMA bars have a faint tail for their latency, which overlaps other work;
  * optionally, every SRAM and accumulator bank, showing when it's written and read.
Scroll to zoom, drag to pan, double-click to reset, hover for details.
"""
from __future__ import annotations

import html
import json
from typing import Optional

LANES = ("dma", "mxu", "vpu")


def _sim_executables(prog) -> list:
    """The simulator segments of a compiled program, in execution order (incl. control-flow bodies)."""
    out = []

    def walk(runner):
        for seg in runner.segments:
            ex = seg.executable
            if hasattr(ex, "last_trace"):
                out.append((seg.graph.name, ex))
            for child in getattr(ex, "sub", {}).values():
                walk(child)
    walk(prog.runner)
    return out


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    spans.sort()
    out: list[list[int]] = []
    for s, e in spans:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def collect(label: str, prog) -> dict:
    """Everything the page needs about one (already executed) program, as plain JSON-able data."""
    segs = _sim_executables(prog)
    if not segs:
        raise ValueError("this program has no npu-sim segments (compile it with target npu-sim)")
    if not all(ex.last_trace for _, ex in segs):
        raise ValueError("run the program before visualizing it")
    cfg = segs[0][1].cfg
    texts, ins, opnames, op_index = [], [], [], {}
    op_spans: dict[int, list[int]] = {}
    bank_names = [f"sram{i}" for i in range(cfg.sram_bytes // cfg.bank_bytes)] + \
                 [f"acc{i}" for i in range(cfg.acc_bytes // cfg.bank_bytes)]
    bank_index = {n: i for i, n in enumerate(bank_names)}
    bank_spans: dict[tuple[int, int], list[tuple[int, int]]] = {}
    segments = []
    offset = 0
    totals = {"busy": {e: 0 for e in LANES}, "instrs": {e: 0 for e in LANES}, "macs": 0, "dram_bytes": 0}
    for name, ex in segs:
        st = ex.last_stats
        segments.append({"name": name, "start": offset, "end": offset + st.cycles})
        for e in LANES:
            totals["busy"][e] += st.busy[e]
            totals["instrs"][e] += st.instrs[e]
        totals["macs"] += st.macs
        totals["dram_bytes"] += st.dram_bytes
        for t in ex.last_trace:
            op = t.ins.op or "(setup)"
            if (name, op) not in op_index:
                op_index[(name, op)] = len(opnames)
                opnames.append(op)
            oi = op_index[(name, op)]
            s, b, e = offset + t.start, offset + t.busy_end, offset + t.end
            span = op_spans.setdefault(oi, [s, e])
            span[0], span[1] = min(span[0], s), max(span[1], e)
            text = str(t.ins) + (f"   ; {t.ins.comment}" if t.ins.comment else "")
            ins.append([LANES.index(t.ins.engine), s, b, e, oi, len(texts)])
            texts.append(text)
            for bank in t.writes:
                bank_spans.setdefault((bank_index[bank], 0), []).append((s, e))
            for bank in t.reads:
                bank_spans.setdefault((bank_index[bank], 1), []).append((s, b))
        offset += st.cycles
    ops = sorted(({"name": opnames[i], "start": a, "end": z} for i, (a, z) in op_spans.items()),
                 key=lambda o: o["start"])
    banks = [[bi, kind, s, e] for (bi, kind), spans in bank_spans.items() for s, e in _merge(spans)]
    cycles = offset
    peak = cfg.mxu_dim ** 2
    return {
        "label": label,
        "config": {"clock_ghz": cfg.clock_ghz, "mxu": f"{cfg.mxu_dim}x{cfg.mxu_dim}",
                   "sram_kib": cfg.sram_bytes // 1024, "acc_kib": cfg.acc_bytes // 1024,
                   "banks": cfg.sram_bytes // cfg.bank_bytes, "dma_gbs": cfg.dma_bytes_per_cycle * cfg.clock_ghz,
                   "dma_latency": cfg.dma_latency, "double_buffer": segs[0][1].double_buffer},
        "stats": {"cycles": cycles, "us": cycles / (cfg.clock_ghz * 1e3), "busy": totals["busy"],
                  "instrs": totals["instrs"], "macs": totals["macs"], "peak": peak,
                  "dram_kib": totals["dram_bytes"] / 1024},
        "segments": segments, "ops": ops, "opnames": opnames, "texts": texts, "ins": ins,
        "bank_names": bank_names, "banks": banks,
    }


def render_html(runs: list[dict], title: str, subtitle: Optional[str] = None) -> str:
    data = json.dumps({"title": title, "subtitle": subtitle or "", "runs": runs}, separators=(",", ":"))
    data = data.replace("</", "<\\/")      # never close the <script> early
    return PAGE.replace("__TITLE__", html.escape(title)).replace("__DATA__", data)


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --bg: #f7f7f5; --panel: #ffffff; --ink: #1d1d1f; --muted: #6b6b70; --line: #e4e4e0; --grid: #efefec;
    --dma: #2f6fdf; --mxu: #e08a00; --vpu: #1a9b6b; --write: #7c4dbd; --read: #b9a4dc; --seg: #c9c9c4;
    color-scheme: light;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #141416; --panel: #1d1d20; --ink: #ececee; --muted: #9c9ca3; --line: #2e2e33; --grid: #26262a;
      --dma: #5b8ff0; --mxu: #f2a53a; --vpu: #3cc68f; --write: #a57de0; --read: #5d4a80; --seg: #4a4a50;
      color-scheme: dark;
    }
  }
  * { box-sizing: border-box; }
  [hidden] { display: none !important; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif; }
  main { max-width: 1400px; margin: 0 auto; padding: 28px 20px 48px; }
  h1 { font-size: 22px; margin: 0 0 2px; letter-spacing: -0.01em; }
  .sub { color: var(--muted); margin: 0 0 18px; }
  .bar { display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center; margin: 0 0 18px;
         color: var(--muted); font-size: 13px; }
  .key { display: inline-flex; align-items: center; gap: 6px; }
  .sw { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
  button, label.toggle { font: inherit; font-size: 13px; color: var(--ink); background: var(--panel);
         border: 1px solid var(--line); border-radius: 7px; padding: 5px 10px; cursor: pointer; }
  label.toggle { display: inline-flex; gap: 6px; align-items: center; }
  .verdict { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px 16px;
             margin: 0 0 18px; font-size: 15px; }
  .run { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 16px 16px 10px;
         margin: 0 0 20px; }
  .run h2 { font-size: 16px; margin: 0 0 12px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 0 0 14px; }
  .card { border: 1px solid var(--line); border-radius: 9px; padding: 9px 12px; }
  .card .k { color: var(--muted); font-size: 12px; }
  .card .v { font-size: 19px; font-variant-numeric: tabular-nums; font-weight: 600; }
  .card .s { color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
  .meter { height: 5px; border-radius: 3px; background: var(--grid); margin-top: 6px; overflow: hidden; }
  .meter > div { height: 100%; }
  .canvaswrap { position: relative; overflow: hidden; border-top: 1px solid var(--line); padding-top: 6px; }
  canvas { display: block; width: 100%; cursor: grab; touch-action: none; }
  canvas.dragging { cursor: grabbing; }
  #tip { position: fixed; pointer-events: none; z-index: 10; max-width: 560px; background: var(--panel);
         border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; font-size: 12px;
         box-shadow: 0 6px 24px rgba(0,0,0,.18); display: none; }
  #tip code { font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre-wrap; word-break: break-all; }
  #tip .m { color: var(--muted); }
  .hint { color: var(--muted); font-size: 12px; margin: 6px 0 0; }
</style>
</head>
<body>
<main>
  <h1 id="title"></h1>
  <p class="sub" id="subtitle"></p>
  <div class="bar">
    <span class="key"><span class="sw" style="background:var(--dma)"></span>DMA (memory transfers)</span>
    <span class="key"><span class="sw" style="background:var(--mxu)"></span>MXU (matrix unit)</span>
    <span class="key"><span class="sw" style="background:var(--vpu)"></span>VPU (vector unit)</span>
    <span class="key"><span class="sw" style="background:var(--dma);opacity:.3"></span>DMA latency (overlaps other work)</span>
    <span class="key bankkey" hidden><span class="sw" style="background:var(--write)"></span>bank being written</span>
    <span class="key bankkey" hidden><span class="sw" style="background:var(--read)"></span>bank being read</span>
    <label class="toggle"><input type="checkbox" id="banks"> Memory banks</label>
    <button id="reset">Reset zoom</button>
  </div>
  <div id="verdict" class="verdict" hidden></div>
  <div id="runs"></div>
  <p class="hint">Scroll to zoom · drag to pan · double-click to reset · hover a bar for the instruction.</p>
</main>
<div id="tip"></div>
<script>
const DATA = __DATA__;
const LANE_NAMES = ["DMA", "MXU", "VPU"];
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = n => n.toLocaleString("en-US");
const tip = document.getElementById("tip");
document.getElementById("title").textContent = DATA.title;
document.getElementById("subtitle").textContent = DATA.subtitle;

const maxCycles = Math.max(...DATA.runs.map(r => r.stats.cycles), 1);
let view = [0, maxCycles];
let showBanks = false;

if (DATA.runs.length === 2) {
  const [a, b] = DATA.runs;
  const ratio = a.stats.cycles / b.stats.cycles;
  const el = document.getElementById("verdict");
  el.hidden = false;
  el.innerHTML = ratio >= 1
    ? `<b>${esc(b.label)}</b> finishes <b>${ratio.toFixed(2)}×</b> faster than <b>${esc(a.label)}</b> (${fmt(b.stats.cycles)} vs ${fmt(a.stats.cycles)} cycles).`
    : `<b>${esc(a.label)}</b> finishes <b>${(1 / ratio).toFixed(2)}×</b> faster than <b>${esc(b.label)}</b> (${fmt(a.stats.cycles)} vs ${fmt(b.stats.cycles)} cycles).`;
}
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c])); }

function card(k, v, s, frac, color) {
  const m = frac === undefined ? "" :
    `<div class="meter"><div style="width:${(100 * Math.min(frac, 1)).toFixed(1)}%;background:${color}"></div></div>`;
  return `<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div>${m}</div>`;
}

const runsEl = document.getElementById("runs");
const views = DATA.runs.map((run, ri) => {
  // per-lane instruction lists, sorted by start (each engine runs in order)
  run.lanes = [[], [], []];
  run.ins.forEach((x, i) => run.lanes[x[0]].push(i));
  const st = run.stats, c = run.config;
  const util = e => st.busy[e] / Math.max(st.cycles, 1);
  const sec = document.createElement("section");
  sec.className = "run";
  sec.innerHTML = `<h2>${esc(run.label)}</h2><div class="cards">
    ${card("Time", `${st.us.toFixed(1)} µs`, `${fmt(st.cycles)} cycles at ${c.clock_ghz} GHz`)}
    ${card("Matrix unit busy", `${(100 * util("mxu")).toFixed(0)}%`, `${fmt(st.instrs.mxu)} matmul tiles`, util("mxu"), css("--mxu"))}
    ${card("DMA busy", `${(100 * util("dma")).toFixed(0)}%`, `${fmt(st.instrs.dma)} transfers`, util("dma"), css("--dma"))}
    ${card("Vector unit busy", `${(100 * util("vpu")).toFixed(0)}%`, `${fmt(st.instrs.vpu)} vector ops`, util("vpu"), css("--vpu"))}
    ${card("Math throughput", `${fmt(Math.round(st.macs / Math.max(st.cycles, 1)))} MAC/cycle`,
           `${(100 * st.macs / Math.max(st.cycles, 1) / st.peak).toFixed(1)}% of ${fmt(st.peak)} peak`)}
    ${card("DRAM traffic", `${st.dram_kib >= 1024 ? (st.dram_kib / 1024).toFixed(2) + " MiB" : st.dram_kib.toFixed(0) + " KiB"}`,
           `${c.mxu} MXU · ${c.sram_kib} KiB SRAM · ${c.dma_gbs} GB/s`)}
  </div><div class="canvaswrap"><canvas></canvas></div>`;
  runsEl.appendChild(sec);
  return {run, canvas: sec.querySelector("canvas")};
});

const AXIS = 22, LANE = 26, OPLANE = 30, BANK = 5, LABEL = 64, GAP = 6;
function layout(run) {
  const rows = [{kind: "ops", y: AXIS, h: OPLANE, name: "IR ops"}];
  let y = AXIS + OPLANE + GAP;
  for (let l = 0; l < 3; l++) { rows.push({kind: "lane", lane: l, y, h: LANE, name: LANE_NAMES[l]}); y += LANE + 4; }
  if (showBanks) {
    y += GAP;
    run.bank_names.forEach((n, i) => { rows.push({kind: "bank", bank: i, y, h: BANK, name: n}); y += BANK + 1; });
  }
  return {rows, height: y + 8};
}

function draw(v) {
  const {run, canvas} = v;
  const lay = layout(run);
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  canvas.style.height = lay.height + "px";
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(lay.height * dpr);
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, lay.height);
  const plotW = w - LABEL, [t0, t1] = view, sx = plotW / (t1 - t0);
  const X = t => LABEL + (t - t0) * sx;
  v.X = X; v.T = x => t0 + (x - LABEL) / sx; v.lay = lay;
  const ink = css("--ink"), muted = css("--muted"), grid = css("--grid"), seg = css("--seg");
  const col = [css("--dma"), css("--mxu"), css("--vpu")];
  // axis + grid
  const step = niceStep((t1 - t0) / Math.max(4, plotW / 110));
  g.font = "11px -apple-system, Segoe UI, sans-serif"; g.textBaseline = "middle";
  for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) {
    const x = X(t);
    g.fillStyle = grid; g.fillRect(x, AXIS - 4, 1, lay.height);
    g.fillStyle = muted; g.textAlign = x > w - 24 ? "right" : "center";     // keep the last label on screen
    g.fillText(t >= 1e6 ? (t / 1e6).toFixed(2) + "M" : t >= 1e3 ? (t / 1e3).toFixed(t % 1000 ? 1 : 0) + "k" : String(t), x, 9);
  }
  // segment boundaries (when the program leaves the NPU)
  if (run.segments.length > 1) run.segments.forEach(s => { g.fillStyle = seg; g.fillRect(X(s.start), AXIS - 4, 1.5, lay.height); });
  g.textAlign = "right";
  for (const r of lay.rows) {
    if (r.kind !== "bank" || r.bank % 8 === 0) { g.fillStyle = muted; g.fillText(r.name, LABEL - 8, r.y + r.h / 2); }
  }
  // IR op spans
  const opRow = lay.rows[0];
  run.ops.forEach((o, i) => {
    if (o.end < t0 || o.start > t1) return;
    const x0 = Math.max(X(o.start), LABEL), x1 = Math.min(X(o.end), w);
    g.fillStyle = opColor(o.name, i); g.fillRect(x0, opRow.y + 3, Math.max(1, x1 - x0 - 1), opRow.h - 6);
    if (x1 - x0 > 50) {
      g.save(); g.beginPath(); g.rect(x0, opRow.y, x1 - x0 - 2, opRow.h); g.clip();
      g.fillStyle = "#fff"; g.textAlign = "left"; g.fillText(o.name, x0 + 5, opRow.y + opRow.h / 2); g.restore();
    }
  });
  // instructions
  for (let l = 0; l < 3; l++) {
    const r = lay.rows[1 + l], idx = run.lanes[l];
    let i = lowerBound(idx, run.ins, t0 - maxDur(run));
    for (; i < idx.length; i++) {
      const [, s, b, e] = run.ins[idx[i]];
      if (s > t1) break;
      if (e < t0) continue;
      const x0 = X(s), xb = X(b), xe = X(e);
      if (l === 0 && xe - xb > 0.5) { g.globalAlpha = 0.22; g.fillStyle = col[l]; g.fillRect(xb, r.y + r.h * 0.38, xe - xb, r.h * 0.24); g.globalAlpha = 1; }
      g.fillStyle = col[l]; g.fillRect(x0, r.y + 2, Math.max(1, xb - x0 - (xb - x0 > 3 ? 1 : 0)), r.h - 4);
    }
  }
  // memory banks
  if (showBanks) {
    const wcol = css("--write"), rcol = css("--read");
    for (const [bi, kind, s, e] of run.banks) {
      if (e < t0 || s > t1) continue;
      const r = lay.rows[4 + bi];
      g.fillStyle = kind === 0 ? wcol : rcol;
      g.fillRect(X(s), r.y, Math.max(1, X(e) - X(s)), r.h);
    }
  }
}
let durCache = new Map();
function maxDur(run) {
  if (!durCache.has(run)) durCache.set(run, run.ins.reduce((m, x) => Math.max(m, x[3] - x[1]), 0));
  return durCache.get(run);
}
function lowerBound(idx, ins, t) { let lo = 0, hi = idx.length; while (lo < hi) { const m = (lo + hi) >> 1; if (ins[idx[m]][1] < t) lo = m + 1; else hi = m; } return lo; }
function niceStep(raw) { const p = Math.pow(10, Math.floor(Math.log10(Math.max(raw, 1)))); for (const m of [1, 2, 5, 10]) if (m * p >= raw) return m * p; return 10 * p; }
function opColor(name, i) {
  const kind = name.split(/[ +]/)[0];
  const hues = {matmul: 28, conv2d: 18, softmax: 280, layernorm: 250, transpose: 200, maxpool2d: 165, reduce_sum: 320, reduce_max: 320, reduce_mean: 320};
  let h = hues[kind]; if (h === undefined) { h = 0; for (const c of kind) h = (h * 31 + c.charCodeAt(0)) % 360; }
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  return `hsl(${h} ${dark ? 45 : 55}% ${i % 2 ? (dark ? 38 : 46) : (dark ? 32 : 40)}%)`;
}
function redraw() { views.forEach(draw); }

// interaction: zoom, pan, reset, hover
views.forEach(v => {
  const c = v.canvas;
  c.addEventListener("wheel", ev => {
    ev.preventDefault();
    const t = v.T(ev.offsetX), f = Math.exp(ev.deltaY * 0.0015);
    let a = t - (t - view[0]) * f, b = t + (view[1] - t) * f;
    if (b - a < 20) return;
    view = [Math.max(0, a), Math.min(maxCycles, b)]; redraw(); hover(v, ev);
  }, {passive: false});
  let drag = null;
  c.addEventListener("pointerdown", ev => { drag = {x: ev.clientX, view: view.slice()}; c.classList.add("dragging"); c.setPointerCapture(ev.pointerId); });
  c.addEventListener("pointermove", ev => {
    if (drag) {
      const dt = (ev.clientX - drag.x) * (drag.view[1] - drag.view[0]) / (c.clientWidth - LABEL);
      let a = drag.view[0] - dt, b = drag.view[1] - dt;
      if (a < 0) { b -= a; a = 0; } if (b > maxCycles) { a -= b - maxCycles; b = maxCycles; }
      view = [Math.max(0, a), b]; redraw();
    } else hover(v, ev);
  });
  c.addEventListener("pointerup", () => { drag = null; c.classList.remove("dragging"); });
  c.addEventListener("pointerleave", () => { tip.style.display = "none"; });
  c.addEventListener("dblclick", () => { view = [0, maxCycles]; redraw(); });
});
document.getElementById("reset").onclick = () => { view = [0, maxCycles]; redraw(); };
document.getElementById("banks").onchange = ev => {
  showBanks = ev.target.checked;
  document.querySelectorAll(".bankkey").forEach(k => { k.hidden = !showBanks; });
  redraw();
};

function hover(v, ev) {
  const {run, lay} = v; if (!lay) return;
  const row = lay.rows.find(r => ev.offsetY >= r.y && ev.offsetY < r.y + r.h + 3);
  const t = v.T(ev.offsetX), slack = 3 * (view[1] - view[0]) / v.canvas.clientWidth;
  let html = null;
  if (row && row.kind === "lane") {
    const idx = run.lanes[row.lane];
    let i = lowerBound(idx, run.ins, t + slack) - 1;
    for (; i >= 0 && i >= lowerBound(idx, run.ins, t - maxDur(run)) - 1; i--) {
      const [l, s, b, e, oi, ti] = run.ins[idx[i]];
      if (t >= s - slack && t <= e + slack) {
        const lat = e > b ? ` <span class="m">(busy ${fmt(b - s)}, data ready after ${fmt(e - s)})</span>` : "";
        html = `<code>${esc(run.texts[ti])}</code><div class="m">${LANE_NAMES[l]} · cycles ${fmt(s)} – ${fmt(e)} · ${fmt(e - s)} cycles${lat}</div><div class="m">part of: ${esc(run.opnames[oi])}</div>`;
        break;
      }
    }
  } else if (row && row.kind === "ops") {
    const o = run.ops.find(o => t >= o.start && t <= o.end);
    if (o) html = `<b>${esc(o.name)}</b><div class="m">cycles ${fmt(o.start)} – ${fmt(o.end)} · ${fmt(o.end - o.start)} cycles · ${(100 * (o.end - o.start) / run.stats.cycles).toFixed(1)}% of the run</div>`;
  } else if (row && row.kind === "bank") {
    const span = run.banks.find(([bi, , s, e]) => bi === row.bank && t >= s - slack && t <= e + slack);
    html = `<b>${row.name}</b>${span ? `<div class="m">${span[1] === 0 ? "being written" : "being read"} · cycles ${fmt(span[2])} – ${fmt(span[3])}</div>` : `<div class="m">idle here</div>`}`;
  }
  if (!html) { tip.style.display = "none"; return; }
  tip.innerHTML = html; tip.style.display = "block";
  const x = Math.min(ev.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  const y = ev.clientY + 16 + tip.offsetHeight > window.innerHeight ? ev.clientY - tip.offsetHeight - 10 : ev.clientY + 16;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
window.addEventListener("resize", redraw);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
redraw();
</script>
</body>
</html>
"""
