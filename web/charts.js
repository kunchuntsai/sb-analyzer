// Chart stack: uPlot, phase bands behind every chart, err_est bands around every series,
// invalid stretches greyed rather than hidden, degraded stretches dashed.
/* global uPlot */
import { state, set } from "./store.js";

export const PHASE_COLORS = {
  drop_in: "rgba(96,165,250,0.13)",
  transition: "rgba(251,191,36,0.14)",
  takeoff_run: "rgba(52,211,153,0.13)",
  air: "rgba(167,139,250,0.16)",
};
export const PHASE_LABEL = {
  drop_in: "Drop-in", transition: "Transition", takeoff_run: "Takeoff run", air: "Air",
};

export const METRIC_META = {
  air_height: { label: "Board height above lip", short: "Air height", color: "#c084fc" },
  speed_along: { label: "Speed along fall line", short: "Speed", color: "#38bdf8" },
  com_height: { label: "CoM height above board", short: "CoM height", color: "#f43f5e" },
  com_lateral: { label: "CoM lateral (+ image right)", short: "CoM lateral", color: "#fb923c" },
  com_toe_heel: { label: "CoM toe (+) / heel (−) side", short: "Toe/heel", color: "#10b981" },
  board_edge: { label: "Board edge angle (+ toe / − heel)", short: "Edge angle", color: "#2dd4bf" },
  com_foreaft: { label: "CoM nose/tail (along board)", short: "Nose/tail", color: "#e879f9" },
  knee_flex_l: { label: "Knee flexion L", short: "Knee L", color: "#34d399" },
  knee_flex_r: { label: "Knee flexion R", short: "Knee R", color: "#22d3ee" },
  trunk_lean: { label: "Trunk lean (+ image right)", short: "Trunk lean", color: "#facc15" },
  line_offset: { label: "Board from left mat edge", short: "Line", color: "#f59e0b" },
};

// board_bump has no chart of its own; its events are drawn on the line chart
export const GROUPS = [
  { title: "Speed", metrics: ["speed_along"] },
  { title: "Flight", metrics: ["air_height"] },
  { title: "Edge: CoM over the board", metrics: ["com_toe_heel"], extraEvents: ["edge_change"] },
  { title: "Edge: board tilt", metrics: ["board_edge"], extraEvents: ["edge_change"] },
  { title: "Centre of mass", metrics: ["com_height", "com_foreaft", "com_lateral"] },
  { title: "Knee flexion", metrics: ["knee_flex_l", "knee_flex_r"] },
  { title: "Trunk lean", metrics: ["trunk_lean"] },
  { title: "Line", metrics: ["line_offset"], extraEvents: ["board_bump"] },
];

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function alpha(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
}

// Split one metric into valid / degraded / invalid series that join at their boundaries.
function splitByValidity(values, validity) {
  const pick = (kind) => values.map((v, i) =>
    validity[i] === kind || validity[i - 1] === kind || validity[i + 1] === kind ? v : null);
  return { valid: pick("valid"), degraded: pick("degraded"), invalid: pick("invalid") };
}

export function phaseBandsHook(getSpans) {
  return (u) => {
    const { ctx } = u;
    const { top, height } = u.bbox;
    for (const s of getSpans()) {
      const x0 = u.valToPos(s.a, "x", true), x1 = u.valToPos(s.b, "x", true);
      ctx.fillStyle = PHASE_COLORS[s.phase];
      ctx.fillRect(x0, top, x1 - x0, height);
    }
  };
}

function eventBandsHook(getEvents) {
  return (u) => {
    const { ctx } = u;
    const { top, height } = u.bbox;
    for (const e of getEvents()) {
      const x0 = u.valToPos(e.a, "x", true), x1 = u.valToPos(e.b, "x", true);
      const w = Math.max(3 * devicePixelRatio, x1 - x0);
      ctx.fillStyle = e.severity === "strong" ? "rgba(239,68,68,0.28)" : "rgba(239,68,68,0.15)";
      ctx.fillRect(x0 - (w - (x1 - x0)) / 2, top, w, height);
    }
  };
}

function cursorHook(getX) {
  return (u) => {
    const x = getX();
    if (x == null) return;
    const px = u.valToPos(x, "x", true);
    const { ctx } = u;
    ctx.save();
    ctx.strokeStyle = css("--fg");
    ctx.lineWidth = 1.5 * devicePixelRatio;
    ctx.beginPath(); ctx.moveTo(px, u.bbox.top); ctx.lineTo(px, u.bbox.top + u.bbox.height);
    ctx.stroke();
    ctx.restore();
  };
}

export function buildSeries(metrics, series, units) {
  const data = [];
  const opts = [];
  const bands = [];
  for (const m of metrics) {
    const s = series[m];
    if (!s) continue;
    const meta = METRIC_META[m];
    const up = s.value.map((v, i) => (v == null || s.err[i] == null ? null : v + s.err[i]));
    const lo = s.value.map((v, i) => (v == null || s.err[i] == null ? null : v - s.err[i]));
    const parts = splitByValidity(s.value, s.validity);
    const base = data.length + 1; // +1 for x
    data.push(up, lo, parts.valid, parts.degraded, parts.invalid);
    opts.push(
      { label: `${m}+err`, stroke: "transparent", points: { show: false }, legend: false },
      { label: `${m}-err`, stroke: "transparent", points: { show: false }, legend: false },
      { label: `${meta.label} (${units[m]})`, stroke: meta.color, width: 1.8, points: { show: false } },
      { label: "degraded", stroke: meta.color, width: 1.5, dash: [5, 4], points: { show: false }, legend: false },
      { label: "invalid", stroke: css("--invalid"), width: 1.5, points: { show: false }, legend: false },
    );
    bands.push({ series: [base, base + 1], fill: alpha(meta.color, 0.16) });
  }
  return { data, opts, bands };
}

// Y-axis range from the samples that count: valid and degraded values with their error bands.
// Invalid stretches (e.g. pose errors mid-air) are still drawn in grey, but clipped, so they
// cannot flatten the part of the run that matters.
const CHART_H = 138;
const MIN_SPAN = { m: 0.1, "m/s": 1, deg: 10 };

function yRange(metricsInGroup, metrics) {
  let lo = Infinity, hi = -Infinity, unit = "m";
  for (const m of metricsInGroup) {
    const s = metrics.series[m];
    if (!s) continue;
    unit = metrics.units[m];
    s.value.forEach((v, i) => {
      if (v == null || s.validity[i] === "invalid") return;
      const e = s.err[i] ?? 0;
      lo = Math.min(lo, v - e); hi = Math.max(hi, v + e);
    });
  }
  if (!Number.isFinite(lo)) return null; // nothing usable: let uPlot auto-range
  const minSpan = MIN_SPAN[unit] ?? 1;
  const mid = (lo + hi) / 2;
  let span = Math.max(hi - lo, minSpan);
  span *= 1.15;
  return () => [mid - span / 2, mid + span / 2];
}

export class ChartStack {
  constructor(root) {
    this.root = root;
    this.plots = [];
  }

  destroy() {
    this.plots.forEach((p) => p.destroy());
    this.plots = [];
    this.root.innerHTML = "";
  }

  render(metrics, spans, events = []) {
    this.destroy();
    this.metrics = metrics;
    this.t0 = metrics.t[0];
    const x = metrics.t.map((t) => t - this.t0);
    this.x = x;
    const spanList = spans.map((s) => ({ phase: s.phase, a: s.t_start - this.t0, b: s.t_end - this.t0 }));
    const width = Math.max(320, this.root.clientWidth - 8);
    for (const g of GROUPS) {
      const { data, opts, bands } = buildSeries(g.metrics, metrics.series, metrics.units);
      if (!data.length) continue;
      const box = document.createElement("div");
      box.className = "chart";
      const h = document.createElement("h3");
      h.innerHTML = `<span>${g.title}</span>` + g.metrics.filter((m) => metrics.series[m]).map((m) =>
        `<span class="key"><i style="background:${METRIC_META[m].color}"></i>${METRIC_META[m].label}</span>`).join("");
      box.appendChild(h);
      this.root.appendChild(box);
      const unit = metrics.units[g.metrics[0]];
      const own = new Set([...g.metrics, ...(g.extraEvents || [])]);
      const evs = events.filter((e) => own.has(e.metric))
        .map((e) => ({ a: e.t_start - this.t0, b: e.t_end - this.t0, severity: e.severity }));
      const yr = yRange(g.metrics, metrics);
      const plot = new uPlot({
        width, height: CHART_H,
        cursor: { sync: { key: "run" }, drag: { x: false, y: false }, points: { show: false } },
        legend: { show: false },
        scales: { x: { time: false }, ...(yr ? { y: { range: yr } } : {}) },
        axes: [
          { stroke: css("--muted"), grid: { stroke: css("--grid") }, ticks: { stroke: css("--grid") },
            values: (u, v) => v.map((s) => s.toFixed(1) + "s"), size: 26, font: "11px system-ui" },
          { stroke: css("--muted"), grid: { stroke: css("--grid") }, ticks: { stroke: css("--grid") },
            size: 52, label: unit, labelSize: 14, labelFont: "11px system-ui", space: 18,
            font: "11px system-ui" },
        ],
        series: [{ label: "t" }, ...opts],
        bands,
        hooks: {
          drawClear: [phaseBandsHook(() => spanList), eventBandsHook(() => evs)],
          draw: [cursorHook(() => x[state.frame])],
        },
      }, [x, ...data], box);
      this._bindScrub(plot);
      this.plots.push(plot);
    }
  }

  _bindScrub(plot) {
    let down = false;
    const toFrame = () => {
      const idx = plot.posToIdx(plot.cursor.left);
      if (idx != null && idx >= 0) set({ frame: idx, playing: false });
    };
    plot.over.addEventListener("mousedown", () => { down = true; toFrame(); });
    plot.over.addEventListener("mousemove", () => down && toFrame());
    window.addEventListener("mouseup", () => { down = false; });
  }

  redraw() {
    for (const p of this.plots) p.redraw(false, false);
  }

  resize() {
    const width = Math.max(320, this.root.clientWidth - 8);
    for (const p of this.plots) p.setSize({ width, height: CHART_H });
  }
}
