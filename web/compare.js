// Compare two runs on a phase-aligned common timeline: each phase is stretched to one unit,
// so u = 0..1 is drop-in, 1..2 transition, 2..3 takeoff run, whatever each run's timing.
/* global uPlot */
import { METRIC_META, PHASE_COLORS } from "./charts.js";
import { Viewer } from "./viewer.js";

const A_COLOR = "#38bdf8";
const B_COLOR = "#f472b6";
const METRICS = ["speed_along", "air_height", "com_toe_heel", "board_edge", "com_height", "knee_flex_l", "knee_flex_r",
  "trunk_lean", "line_offset", "com_foreaft"];
const PHASES = ["drop_in", "transition", "takeoff_run", "air"];
const NAMES = ["Drop-in", "Transition", "Takeoff run", "Air"];
const U_MAX = PHASES.length;

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function nearestIdx(ts, t) {
  let best = 0;
  for (let i = 1; i < ts.length; i++) if (Math.abs(ts[i] - t) < Math.abs(ts[best] - t)) best = i;
  return best;
}

export class CompareView {
  constructor(root) {
    this.root = root;
    this.plots = [];
    this.u = 0;
  }

  async open(clips, a, b, loadRun) {
    this.root.innerHTML = `
      <div class="cmp-head">
        <label>A <select id="cmpA"></select></label>
        <label>B <select id="cmpB"></select></label>
        <input id="cmpU" type="range" min="0" max="${U_MAX}" step="0.005" value="0">
        <span id="cmpULabel" class="mono"></span>
      </div>
      <div class="cmp-body">
        <div class="cmp-views">
          <figure><figcaption style="color:${A_COLOR}">A</figcaption><canvas id="cmpCanvasA"></canvas></figure>
          <figure><figcaption style="color:${B_COLOR}">B</figcaption><canvas id="cmpCanvasB"></canvas></figure>
        </div>
        <div id="cmpCharts" class="cmp-charts"></div>
      </div>`;
    const selA = this.root.querySelector("#cmpA");
    const selB = this.root.querySelector("#cmpB");
    for (const sel of [selA, selB]) {
      sel.innerHTML = clips.map((c) => `<option value="${c.clip_id}">${c.clip_id}</option>`).join("");
    }
    selA.value = a; selB.value = b;
    const reload = () => this.load(selA.value, selB.value, loadRun);
    selA.onchange = reload; selB.onchange = reload;
    const slider = this.root.querySelector("#cmpU");
    slider.oninput = () => this.setU(parseFloat(slider.value));
    await this.load(a, b, loadRun);
  }

  async load(a, b, loadRun) {
    const [cmp, runA, runB] = await Promise.all([
      fetch(`/compare?a=${a}&b=${b}`).then((r) => r.json()), loadRun(a), loadRun(b),
    ]);
    this.cmp = cmp; this.runA = runA; this.runB = runB;
    this.viewA = new Viewer(this.root.querySelector("#cmpCanvasA"));
    this.viewB = new Viewer(this.root.querySelector("#cmpCanvasB"));
    this.viewA.setStance(runA.detail.stance);
    this.viewB.setStance(runB.detail.stance);
    await Promise.all([
      this.viewA.load(a, runA.metrics.frames, runA.overlay),
      this.viewB.load(b, runB.metrics.frames, runB.overlay),
    ]);
    this.renderCharts();
    this.setU(this.u);
  }

  setU(u) {
    this.u = u;
    const k = Math.round((u / U_MAX) * (this.cmp.a.u.length - 1));
    const ta = this.cmp.a.t[k], tb = this.cmp.b.t[k];
    if (ta != null) this.viewA.draw(nearestIdx(this.runA.metrics.t, ta));
    if (tb != null) this.viewB.draw(nearestIdx(this.runB.metrics.t, tb));
    const p = Math.min(U_MAX - 1, Math.floor(u));
    this.root.querySelector("#cmpULabel").textContent = `${NAMES[p]} ${(100 * (u - p)).toFixed(0)}%`;
    for (const p of this.plots) p.redraw(false, false);
  }

  renderCharts() {
    this.plots.forEach((p) => p.destroy());
    this.plots = [];
    const host = this.root.querySelector("#cmpCharts");
    host.innerHTML = "";
    const { a, b, units } = this.cmp;
    const width = Math.max(320, host.clientWidth - 8);
    for (const m of METRICS) {
      if (!a.series[m] || !b.series[m]) continue;
      const box = document.createElement("div");
      box.className = "chart";
      box.innerHTML = `<h3><span>${METRIC_META[m].label} (${units[m]})</span>
        <span class="key"><i style="background:${A_COLOR}"></i>A</span>
        <span class="key"><i style="background:${B_COLOR}"></i>B</span></h3>`;
      host.appendChild(box);
      const split = (s, ok) => s.value.map((v, i) => (ok === (s.valid[i] || s.valid[i - 1] || s.valid[i + 1]) ? v : null));
      const invalidOnly = (s) => s.value.map((v, i) => (!s.valid[i] || !s.valid[i - 1] || !s.valid[i + 1] ? v : null));
      const plot = new uPlot({
        width, height: 112,
        legend: { show: false },
        cursor: { drag: { x: false, y: false }, points: { show: false } },
        scales: { x: { time: false, range: [0, U_MAX] } },
        axes: [
          { stroke: css("--muted"), grid: { stroke: css("--grid") }, ticks: { stroke: css("--grid") },
            splits: () => [0, 1, 2, 3, 4],
            values: (u, v) => v.map((x) => ({ 0: "drop-in", 1: "transition", 2: "takeoff", 3: "lip", 4: "land" }[x] ?? "")) },
          { stroke: css("--muted"), grid: { stroke: css("--grid") }, ticks: { stroke: css("--grid") }, size: 52 },
        ],
        series: [
          {},
          { stroke: A_COLOR, width: 1.8, points: { show: false } },
          { stroke: B_COLOR, width: 1.8, points: { show: false } },
          { stroke: css("--invalid"), width: 1.5, points: { show: false } },
          { stroke: css("--invalid"), width: 1.5, points: { show: false } },
        ],
        hooks: {
          drawClear: [(u) => {
            PHASES.forEach((ph, k) => {
              const x0 = u.valToPos(k, "x", true), x1 = u.valToPos(k + 1, "x", true);
              u.ctx.fillStyle = PHASE_COLORS[ph];
              u.ctx.fillRect(x0, u.bbox.top, x1 - x0, u.bbox.height);
            });
          }],
          draw: [(u) => {
            const px = u.valToPos(this.u, "x", true);
            u.ctx.strokeStyle = css("--fg"); u.ctx.lineWidth = 1.5 * devicePixelRatio;
            u.ctx.beginPath(); u.ctx.moveTo(px, u.bbox.top); u.ctx.lineTo(px, u.bbox.top + u.bbox.height); u.ctx.stroke();
          }],
        },
      }, [a.u, split(a.series[m], true), split(b.series[m], true),
        invalidOnly(a.series[m]), invalidOnly(b.series[m])], box);
      plot.over.addEventListener("mousemove", (e) => {
        if (e.buttons !== 1) return;
        const u = plot.posToVal(plot.cursor.left, "x");
        this.root.querySelector("#cmpU").value = u;
        this.setU(Math.max(0, Math.min(U_MAX, u)));
      });
      this.plots.push(plot);
    }
  }
}
