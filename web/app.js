import { state, set, subscribe, toKmh } from "./store.js";
import { Viewer } from "./viewer.js";
import { ChartStack, METRIC_META, PHASE_COLORS, PHASE_LABEL } from "./charts.js";
import { CompareView } from "./compare.js";
import { Importer } from "./importer.js";
import { Library } from "./library.js";

const $ = (s) => document.querySelector(s);
const viewer = new Viewer($("#frame"), $("#inset"), { responsive: true });
const charts = new ChartStack($("#charts"));
const compare = new CompareView($("#compare"));
const RATES = [0.2, 0.25, 0.5, 0.75, 1];
let run = null; // { detail, metrics, overlay, events }
let clips = [];
let playToken = 0;

async function loadRun(clipId) {
  const [detail, metrics, overlay, events] = await Promise.all([
    fetch(`/clips/${clipId}`).then((r) => r.json()),
    fetch(`/clips/${clipId}/metrics`).then((r) => r.json()),
    fetch(`/clips/${clipId}/overlay`).then((r) => r.json()),
    fetch(`/clips/${clipId}/events`).then((r) => r.json()),
  ]);
  toKmh(metrics.series, metrics.units);
  return { detail, metrics, overlay, events };
}



const fmt = (v, digits = 2) => (v == null || !Number.isFinite(v) ? "–" : v.toFixed(digits));
const idxOfFrame = (f) => Math.max(0, run.metrics.frames.indexOf(f));

// ---- header: compact QA strip ------------------------------------------------------------------
function renderQA(d) {
  const qa = d.qa || {};
  const conf = Object.values(qa.mean_kpt_conf || {});
  const resid = d.session?.residual_px;
  const g = qa.jump?.g_fit;
  const chip = (label, value, cls = "", title = "") =>
    `<span class="chip ${cls}" title="${title}">${label ? `<b>${label}</b>` : ""}${value}</span>`;
  $("#qa").innerHTML = [
    chip("", d.recorded_at?.replace("T", " ").slice(0, 16) ?? "–"),
    stanceChip(qa.stance, d.stance, chip),
    chip("detected", `${fmt(100 * (qa.detection_rate ?? 0), 0)}%`, qa.detection_rate < 0.9 ? "warn" : ""),
    chip("kpt", fmt(Math.min(...conf), 2), Math.min(...conf) < 0.5 ? "warn" : "", "lowest mean keypoint confidence of any phase"),
    chip("edge", `${fmt(resid, 1)}px`, resid > 8 ? "warn" : "", "mat-edge straightness residual"),
    chip("range", qa.range_model ? `±${fmt(100 * qa.range_model.mad_log, 1)}%` : "body", qa.range_model ? "" : "warn",
      "frame-to-frame noise of the body-size range fit"),
    chip("pitch", `${fmt(qa.pitch?.deg, 1)}°`, qa.pitch?.source === "config" ? "warn" : "", `camera pitch (${qa.pitch?.source})`),
    chip("g", g == null ? "–" : fmt(g, 1), g == null || g < 7.5 || g > 12.5 ? "warn" : "",
      "gravity recovered from the flight's descent; 9.8 means the flight geometry is consistent"),
  ].join("");
  $("#qa").title = `pipeline ${d.pipeline_version} · config ${d.config_hash}`;
  $("#exportCsv").href = `/clips/${d.clip_id}/export.csv`;
  $("#exportParquet").href = `/clips/${d.clip_id}/export.parquet`;
}

function stanceChip(st, raw, chip) {
  const value = st?.value ?? (raw === "left_lead" ? "regular" : raw === "right_lead" ? "goofy" : "–");
  if (!st) return chip("stance", value);
  const pct = (x) => `${Math.round(100 * x)}%`;
  const title = st.source === "config"
    ? `set in config.toml (auto would say: lead foot ${st.lead_foot.value}, toes ${st.toe_direction.value})`
    : `lead foot → ${st.lead_foot.value} (${pct(st.lead_foot.agreement)} of ${st.lead_foot.frames} frames); ` +
      `toe direction → ${st.toe_direction.value} (${pct(st.toe_direction.agreement)} of ${st.toe_direction.frames} frames)` +
      (st.cues_agree ? "" : " · cues disagree: toe direction used");
  const warn = st.source !== "config" && (!st.cues_agree || st.confidence < 0.8);
  return chip("stance", `${value} ${st.source === "config" ? "(set)" : pct(st.confidence)}`, warn ? "warn" : "", title);
}

// Headline numbers, computed from the samples that count (valid or degraded, never invalid).
// Each tile jumps the video to the frame where its value occurs.
function keyMetricTiles() {
  const m = run.metrics;
  const usable = (s, i) => s.value[i] != null && s.validity[i] !== "invalid";
  const extreme = (key, pick = (v) => v) => {
    const s = m.series[key];
    if (!s) return null;
    let best = null;
    s.value.forEach((v, i) => {
      if (!usable(s, i)) return;
      if (best === null || pick(v) > pick(s.value[best])) best = i;
    });
    return best === null ? null : { i: best, v: s.value[best], err: s.err[best] };
  };
  const card = (label, x, value, sub) => x
    ? `<div class="tile key" data-frame="${x.i}" title="Click to go to this moment (${fmt(m.t[x.i] - m.t[0], 2)} s)">
         <span>${label}</span><b>${value}</b><small>${sub}</small></div>`
    : `<div class="tile key"><span>${label}</span><b>–</b><small>not measured</small></div>`;

  const air = extreme("air_height");
  const spd = extreme("speed_along");
  const comH = extreme("com_height");
  // stance-aware: + toe edge / - heel edge
  const comL = extreme("com_toe_heel", Math.abs);
  const toeMax = extreme("com_toe_heel");
  const heelMax = extreme("com_toe_heel", (v) => -v);
  // line width: how far the board's track spreads across the slope while on the mat
  let line = null;
  const ls = m.series.line_offset;
  if (ls) {
    let lo = null, hi = null;
    ls.value.forEach((v, i) => {
      if (!usable(ls, i)) return;
      if (lo === null || v < ls.value[lo]) lo = i;
      if (hi === null || v > ls.value[hi]) hi = i;
    });
    if (lo !== null) line = { i: hi, v: ls.value[hi] - ls.value[lo], lo, hi };
  }
  const bw = run.detail.qa?.board_width_m ?? 0.25;
  return [
    card("Air height (max)", air, air ? `${fmt(air.v, 2)} m` : "", air ? `lip tip → summit ±${fmt(air.err, 2)}` : ""),
    card("Speed (max)", spd, spd ? `${fmt(spd.v, 0)} km/h` : "", spd ? "down the slope" : ""),
    card("CoM height (max)", comH, comH ? `${fmt(comH.v, 2)} m` : "", "above the board"),
    card("CoM lateral (max)", comL,
      comL ? `${comL.v >= 0 ? "+" : "−"}${fmt(Math.abs(comL.v) * 100, 0)} cm` : "",
      comL ? `toe +${fmt(Math.max(0, toeMax.v) * 100, 0)} / heel −${fmt(Math.max(0, -heelMax.v) * 100, 0)} cm` : ""),
    card("Line (width)", line, line ? `${fmt(line.v, 2)} m` : "",
      line ? `${fmt(line.v / bw, 1)} board widths across` : ""),
  ];
}

// Edge transitions: an ideal edge-to-edge roll keeps the new line within one board width of
// the old one. One tile per transition: shift in board widths, green / amber / red.
function edgeTransitionTiles(qa) {
  const tr = qa.edge_transitions || [];
  const bw = qa.board_width_m ?? 0.25;
  return tr.map((x, k) => {
    const cls = { ok: "good", borderline: "warn", over: "bad" }[x.verdict];
    const mark = { ok: "✓", borderline: "≈", over: "✗" }[x.verdict];
    return `<div class="tile ${cls}" title="Line shift across the slope during the ${x.direction} edge change: ${(Math.abs(x.shift_m) * 100).toFixed(0)} ± ${(x.err_m * 100).toFixed(0)} cm; one board width = ${(bw * 100).toFixed(0)} cm">
      <span>Edge change ${k + 1} (${x.direction})</span><b>${mark} ${x.board_widths.toFixed(1)} board widths</b></div>`;
  });
}

// ---- analysis result: jump + sudden movements --------------------------------------------------
function renderResult() {
  const qa = run.detail.qa || {};
  const j = qa.jump;
  const tile = (label, value, sub = "", cls = "") =>
    `<div class="tile ${cls}"><span>${label}</span><b>${value}</b><small>${sub}</small></div>`;
  $("#keyMetrics").innerHTML = keyMetricTiles().join("");
  $("#keyMetrics").querySelectorAll(".tile[data-frame]").forEach((el) =>
    el.addEventListener("click", () => set({ frame: +el.dataset.frame, playing: false })));
  $("#jump").innerHTML = [
    tile("Air time", j ? `${fmt(j.air_time_s, 2)} s` : "–",
      j ? (j.landing_seen ? "takeoff → touchdown" : "touchdown not seen") : "no flight tracked",
      j && !j.landing_seen ? "warn" : ""),
    tile("Time to apex", j ? `${fmt(j.t_apex - j.t_takeoff, 2)} s` : "–", ""),
    tile("Takeoff speed", j ? `${fmt(j.takeoff_speed_mps * 3.6, 0)} km/h` : "–", ""),
    ...edgeTransitionTiles(qa),
  ].join("");

  const ev = run.events;
  const strong = ev.filter((e) => e.severity === "strong").length;
  const byPhase = {};
  ev.forEach((e) => (byPhase[e.phase] = (byPhase[e.phase] || 0) + 1));
  $("#eventSummary").innerHTML = ev.length
    ? `<b>${ev.length}</b> sudden movement${ev.length > 1 ? "s" : ""} · <b class="strong">${strong}</b> strong · ` +
      Object.entries(byPhase).map(([p, n]) => `${PHASE_LABEL[p]} ${n}`).join(" · ")
    : "No sudden movements detected: the motion is smooth throughout.";
  const t0 = run.metrics.t[0];
  $("#events").innerHTML = ev.map((e, k) => `
    <li data-k="${k}" class="${e.severity}">
      <span class="dot"></span>
      <span class="when mono">${fmt(e.t_peak - t0, 2)}s</span>
      <span class="ph" style="background:${PHASE_COLORS[e.phase].replace(/[\d.]+\)$/, "0.5)")}">${PHASE_LABEL[e.phase]}</span>
      <span class="txt">${e.text}</span>
      <button class="slow" title="Replay at 0.25×">▶ 0.25×</button>
    </li>`).join("");
  $("#events").querySelectorAll("li").forEach((li) => {
    const e = ev[+li.dataset.k];
    li.addEventListener("click", () => set({ frame: idxOfFrame(e.frame_peak), playing: false }));
    li.querySelector(".slow").addEventListener("click", (evt) => {
      evt.stopPropagation();
      const pad = Math.round(0.25 / (run.metrics.t[1] - run.metrics.t[0]));
      playRange(Math.max(0, idxOfFrame(e.frame_start) - pad),
        Math.min(run.metrics.t.length - 1, idxOfFrame(e.frame_end) + pad), 0.25);
      li.scrollIntoView({ block: "nearest" });
    });
  });
}

function renderScrubStrip() {
  const t = run.metrics.t, t0 = t[0], T = t[t.length - 1] - t0;
  $("#phaseStrip").innerHTML = run.detail.phases.map((p) => {
    const l = (100 * (p.t_start - t0)) / T, w = (100 * (p.t_end - p.t_start)) / T;
    return `<div style="left:${l}%;width:${w}%;background:${PHASE_COLORS[p.phase].replace(/[\d.]+\)$/, "0.6)")}">${PHASE_LABEL[p.phase]}</div>`;
  }).join("");
  $("#eventTicks").innerHTML = run.events.map((e) => {
    const l = (100 * (e.t_start - t0)) / T, w = Math.max(0.6, (100 * (e.t_end - e.t_start)) / T);
    return `<i class="${e.severity}" style="left:${l}%;width:${w}%" title="${e.text}"></i>`;
  }).join("");
  $("#scrub").max = t.length - 1;
}

function renderReadout() {
  const i = state.frame, m = run.metrics;
  $("#readout").innerHTML = Object.keys(METRIC_META).filter((k) => m.series[k]).map((k) => {
    const s = m.series[k];
    const unit = m.units[k], d = unit === "deg" ? 0 : 2;
    return `<div class="${s.validity[i]}"><i style="background:${METRIC_META[k].color}"></i>
      <span>${METRIC_META[k].short}</span><b class="mono">${fmt(s.value[i], d)}</b><small>±${fmt(s.err[i], d)} ${unit}</small></div>`;
  }).join("");
  const t = m.t[i] - m.t[0];
  $("#frameInfo").textContent =
    `frame ${m.frames[i]} · ${t.toFixed(3)} s · ${PHASE_LABEL[m.phase[i]] ?? ""}`;
}

const edgeGauge = (i = state.frame) => {
  const s = run.metrics.series.board_edge;
  return s ? { value: s.value[i], err: s.err[i], validity: s.validity[i] } : null;
};

const activeEvents = (i = state.frame) => {
  const f = run.metrics.frames[i];
  return run.events.filter((e) => f >= e.frame_start && f <= e.frame_end);
};

function renderEventHighlight() {
  const f = run.metrics.frames[state.frame];
  const active = activeEvents();
  $(".stage").classList.toggle("alert", active.some((e) => e.severity === "strong"));
  $(".stage").classList.toggle("notice", active.length > 0);
  $("#events").querySelectorAll("li").forEach((li) => {
    const e = run.events[+li.dataset.k];
    li.classList.toggle("active", f >= e.frame_start && f <= e.frame_end);
  });
}

async function selectClip(clipId) {
  set({ clipId, frame: 0, playing: false });
  renderCurrentLabel();
  $("#loading").hidden = false;
  run = await loadRun(clipId);
  viewer.setStance(run.detail.stance);
  renderQA(run.detail);
  renderResult();
  await viewer.load(clipId, run.metrics.frames, run.overlay, (n, total) => {
    $("#loading").textContent = `Loading frames ${n}/${total}`;
  });
  $("#loading").hidden = true;
  $("#mode").textContent = viewer.mode === "follow" ? "Follow cam" : "Full frame";
  fitVideoColumn();
  charts.render(run.metrics, run.detail.phases, run.events);
  renderScrubStrip();
  onFrame();
}

// ---- resizable columns: drag the splitter; the video scales to the left column's width ------
const LEFT_KEY = "sbanalyze.leftWidth";
const MIN_LEFT = 280, MIN_RIGHT = 460;

// Default: the video as tall as the viewport allows (at the canvas's aspect ratio).
function defaultLeftWidth() {
  // the follow cam at its natural aspect, as tall as the window allows
  return Math.round(videoBoxHeight() * FOLLOW_ASPECT);
}

function clampLeft(w) {
  const hi = Math.min(window.innerWidth - MIN_RIGHT - 40, maxLeftWidth());
  return Math.max(Math.min(MIN_LEFT, hi), Math.min(hi, w));
}

// The video fills the left column's width, but never gets taller than the window (the controls
// under it must stay on screen). Past that, it stays centred in the wider column.
// The video box fills the left column's width and the window's remaining height: the window
// minus the header and everything under the video (scrubber, buttons, layers, hint), so the
// whole left column is always on screen. Dragging the splitter changes the box's width; the
// follow cam zooms to cover it, so a wider column shows the rider bigger.
const FOLLOW_ASPECT = 0.75;

function videoBoxHeight() {
  const left = $(".left"), c = $("#frame");
  const controls = left.scrollHeight - c.getBoundingClientRect().height;
  return Math.max(260, window.innerHeight - 48 - 24 - controls);
}

function maxLeftWidth() {
  return window.innerWidth - MIN_RIGHT - 40;
}

function sizeCanvas() {
  const c = $("#frame");
  const col = $(".left").clientWidth;
  // never taller than 1:2 portrait: past that a narrow column would show only a sliver
  const h = Math.min(videoBoxHeight(), col * 2);
  c.style.width = `${col}px`;
  c.style.height = `${Math.round(h)}px`;
  viewer.resize();
  if (run) viewer.draw(state.frame, activeEvents(), edgeGauge());
}

function setLeftWidth(w, save = false) {
  const px = clampLeft(Math.round(w));
  document.documentElement.style.setProperty("--left-w", `${px}px`);
  sizeCanvas();
  if (save) { try { localStorage.setItem(LEFT_KEY, String(px)); } catch { /* storage blocked */ } }
  return px;
}

function fitVideoColumn() {
  let saved = null;
  try { saved = +localStorage.getItem(LEFT_KEY) || null; } catch { /* storage blocked */ }
  setLeftWidth(saved ?? defaultLeftWidth());
}

(function wireSplitter() {
  const sp = $("#splitter");
  let startX = 0, startW = 0, raf = 0;
  sp.addEventListener("pointerdown", (e) => {
    startX = e.clientX;
    startW = $(".left").getBoundingClientRect().width;
    sp.setPointerCapture(e.pointerId);
    sp.classList.add("drag");
    document.body.classList.add("resizing");
  });
  sp.addEventListener("pointermove", (e) => {
    if (!sp.classList.contains("drag")) return;
    setLeftWidth(startW + e.clientX - startX);
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => charts.resize());
  });
  const end = (e) => {
    if (!sp.classList.contains("drag")) return;
    sp.classList.remove("drag");
    document.body.classList.remove("resizing");
    setLeftWidth(startW + e.clientX - startX, true);
    charts.resize();
  };
  sp.addEventListener("pointerup", end);
  sp.addEventListener("pointercancel", end);
  sp.addEventListener("dblclick", () => {
    try { localStorage.removeItem(LEFT_KEY); } catch { /* storage blocked */ }
    setLeftWidth(defaultLeftWidth());
    charts.resize();
  });
  // keyboard: focus the splitter and use the arrow keys
  sp.tabIndex = 0;
  sp.addEventListener("keydown", (e) => {
    const d = { ArrowLeft: -40, ArrowRight: 40 }[e.key];
    if (!d) return;
    e.preventDefault(); e.stopPropagation();
    setLeftWidth($(".left").getBoundingClientRect().width + d, true);
    charts.resize();
  });
})();

function onFrame() {
  if (!run) return;
  viewer.draw(state.frame, activeEvents(), edgeGauge());
  charts.redraw();
  renderReadout();
  renderEventHighlight();
  $("#scrub").value = state.frame;
}

function step(n) {
  if (!run) return;
  const last = run.metrics.frames.length - 1;
  set({ frame: Math.max(0, Math.min(last, state.frame + n)), playing: false });
}

// Playback: advance a clip clock by scaled wall-clock time and show the frame at that time.
// With slow-mo on events, the clock runs at 0.25× while a sudden movement is on screen.
function playRange(from, to, rate = null, loopFrom = from) {
  const t = run.metrics.t;
  const token = ++playToken;
  set({ frame: from, playing: true });
  let clock = t[from], last = performance.now();
  const tick = (now) => {
    if (!state.playing || token !== playToken) return;
    const base = rate ?? state.rate;
    const slow = $("#slowmo").checked && activeEvents().length ? Math.min(base, 0.25) : base;
    clock += ((now - last) / 1000) * slow;
    last = now;
    const target = clock;
    let i = state.frame;
    if (i >= to && state.loop) { // wrap around and keep playing
      set({ frame: loopFrom });
      clock = t[loopFrom];
      requestAnimationFrame(tick);
      return;
    }
    while (i < to && t[i + 1] <= target) i++;
    if (i !== state.frame) set({ frame: i });
    if (i >= to && !state.loop) { set({ playing: false }); return; }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function play() {
  const last = run.metrics.t.length - 1;
  playRange(state.frame >= last ? 0 : state.frame, last, null, 0); // loops over the whole run
}

function setRate(r) {
  set({ rate: r }); // the running clock picks the new rate up on its next tick
  $("#rate").value = String(r);
}

function toggleMode() {
  viewer.setMode(viewer.mode === "follow" ? "full" : "follow");
  $("#mode").textContent = viewer.mode === "follow" ? "Follow cam" : "Full frame";
  fitVideoColumn();
  onFrame();
}

subscribe((s, patch) => {
  if ("frame" in patch) onFrame();
  if ("layers" in patch) viewer.draw(state.frame, activeEvents(), edgeGauge());
  if ("playing" in patch) $("#play").textContent = s.playing ? "❚❚" : "▶";
});

// ---- wiring ------------------------------------------------------------------------------------
$("#scrub").addEventListener("input", (e) => set({ frame: +e.target.value, playing: false }));
$("#play").addEventListener("click", () => (state.playing ? set({ playing: false }) : play()));
$("#prev").addEventListener("click", () => step(-1));
$("#next").addEventListener("click", () => step(1));
$("#rate").addEventListener("change", (e) => setRate(+e.target.value));
$("#loop").addEventListener("change", (e) => set({ loop: e.target.checked }));
$("#mode").addEventListener("click", toggleMode);
$("#inset").addEventListener("click", toggleMode);

for (const box of document.querySelectorAll("#layers input")) {
  box.checked = state.layers[box.name];
  box.addEventListener("change", () => set({ layers: { ...state.layers, [box.name]: box.checked } }));
}
for (const tab of document.querySelectorAll("nav button")) {
  tab.addEventListener("click", async () => {
    document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("on", b === tab));
    const v = tab.dataset.view;
    set({ view: v, playing: false });
    $("#timeline").hidden = v !== "timeline";
    $("#compare").hidden = v !== "compare";
    if (v === "compare" && !compare.cmp && clips.length >= 2) {
      await compare.open(clips, clips[0].clip_id, clips[1].clip_id, loadRun);
    }
  });
}
window.addEventListener("keydown", (e) => {
  if (state.view !== "timeline" || ["SELECT", "INPUT"].includes(e.target.tagName)
      || document.querySelector("dialog[open]")) return;
  const k = { ArrowLeft: -1, ArrowRight: 1, ",": -10, ".": 10 }[e.key];
  if (k) { e.preventDefault(); step(k); }
  if (e.key === " ") { e.preventDefault(); state.playing ? set({ playing: false }) : play(); }
  if (e.key === "f") toggleMode();
  if (e.key === "l") { set({ loop: !state.loop }); $("#loop").checked = state.loop; }
  if (e.key === "[" || e.key === "]") {
    const i = RATES.indexOf(state.rate) + (e.key === "]" ? 1 : -1);
    setRate(RATES[Math.max(0, Math.min(RATES.length - 1, i))]);
  }
});
window.addEventListener("resize", () => {
  setLeftWidth($(".left").getBoundingClientRect().width); // keep within the new window
  sizeCanvas();
  charts.resize();
});

let allClips = [];

function renderCurrentLabel() {
  const c = allClips.find((x) => x.clip_id === state.clipId);
  $("#currentLabel").textContent = c
    ? (c.title || `${c.recorded_at.replace("T", " ").slice(0, 16)} · ${c.file_name}`) : "Videos";
}

async function refreshClips() {
  allClips = await fetch("/clips").then((r) => r.json());
  clips = allClips.filter((c) => c.status === "ok");
  renderCurrentLabel();
  if (library?.isOpen) library.render();
}

function showEmpty() {
  run = null;
  set({ clipId: null });
  $("#loading").hidden = false;
  $("#loading").innerHTML = `No videos yet. <button class="primary" id="firstImport">＋ Add videos</button>`;
  $("#firstImport").addEventListener("click", () => importer.open());
  $("#charts").innerHTML = ""; $("#events").innerHTML = ""; $("#jump").innerHTML = ""; $("#keyMetrics").innerHTML = "";
  $("#qa").innerHTML = ""; $("#eventSummary").textContent = "";
  renderCurrentLabel();
}

const importer = new Importer({
  getClips: () => allClips,
  onClipReady: async (clipId) => {
    await refreshClips();
    if (clipId && (!run || clipId === state.clipId)) await selectClip(clipId); // new, or re-analysed
  },
  onJobs: () => library?.isOpen && library.refreshJobs(),
});

const library = new Library({
  getClips: () => allClips,
  getCurrent: () => state.clipId,
  importer,
  onOpen: (clipId) => selectClip(clipId),
  onChanged: async ({ removed }) => {
    await refreshClips();
    if (removed && removed === state.clipId) {
      if (clips.length) await selectClip(clips[0].clip_id);
      else showEmpty();
    }
  },
});

(async () => {
  await refreshClips();
  const hash = new URLSearchParams(location.hash.slice(1));
  if (hash.has("import")) importer.open(hash.get("import") || "upload", hash.get("path"));
  if (!clips.length) { showEmpty(); return; }
  const params = hash;
  const pick = clips.find((c) => c.clip_id === params.get("clip")) ?? clips[0];
  await selectClip(pick.clip_id);
  if (params.has("left")) { setLeftWidth(+params.get("left")); charts.resize(); }
  if (params.has("frame")) set({ frame: +params.get("frame") });
  if (params.has("compare")) document.querySelector('nav button[data-view="compare"]').click();
})();
