// Frame viewer. The main canvas is a follow cam: a full-resolution crop that keeps the rider's
// bounding box centred. A small inset shows the whole frame for context. There is no <video>
// element: the run is preloaded as Image[] and indexed, so scrubbing is frame-exact.
// During a detected sudden movement the rider's box turns red (strong) or amber (moderate), and
// the body part responsible is highlighted.
import { state } from "./store.js";

const C = {
  skeleton: "#7dd3fc", joint: "#e0f2fe", lowConf: "rgba(148,163,184,0.55)",
  board: "#f59e0b", com: "#f43f5e", box: "rgba(255,255,255,0.55)", mat: "rgba(34,197,94,0.8)",
  strong: "#ef4444", moderate: "#f59e0b",
};
// HALPE-26
const K = { L_HIP: 11, R_HIP: 12, L_KNEE: 13, R_KNEE: 14, L_ANKLE: 15, R_ANKLE: 16, HEAD: 17, NECK: 18, HIP: 19 };
const PART_EDGES = {
  knee_flex_l: [[K.L_HIP, K.L_KNEE], [K.L_KNEE, K.L_ANKLE]],
  knee_flex_r: [[K.R_HIP, K.R_KNEE], [K.R_KNEE, K.R_ANKLE]],
  trunk_lean: [[K.HIP, K.NECK], [K.NECK, K.HEAD]],
};
const PART_JOINT = { knee_flex_l: K.L_KNEE, knee_flex_r: K.R_KNEE, trunk_lean: K.NECK };
const COM_METRICS = new Set(["com_height", "com_lateral", "com_foreaft", "edge_change"]);
const BOARD_METRICS = new Set(["line_offset", "board_bump", "edge_change"]);

function loadImages(urls, onProgress) {
  let done = 0;
  return Promise.all(urls.map((src) => new Promise((resolve) => {
    const img = new Image();
    img.onload = img.onerror = () => { onProgress?.(++done, urls.length); resolve(img); };
    img.src = src;
  })));
}

export class Viewer {
  // responsive: the canvas takes its size from its CSS box (the main viewer, which fills the
  // left column); otherwise it takes the image's own size (compare view).
  constructor(canvas, inset = null, { responsive = false } = {}) {
    this.responsive = responsive;
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.inset = inset;
    this.ictx = inset?.getContext("2d");
    this.full = [];
    this.follow = [];
    this.overlay = null;
    this.mode = "follow"; // "follow" | "full"
    this.toeOnRight = true; // seen from behind: regular riders face image-right, goofy image-left
  }

  async load(clipId, frames, overlay, onProgress) {
    this.overlay = overlay;
    const hasFollow = !!overlay.follow;
    const v = overlay.version ?? Date.now(); // new version after every re-analysis
    const urls = frames.map((f) => `/clips/${clipId}/frames/${f}?v=${v}`);
    const furls = hasFollow ? frames.map((f) => `/clips/${clipId}/follow/${f}?v=${v}`) : [];
    const total = urls.length + furls.length;
    let base = 0;
    const prog = (n) => onProgress?.(base + n, total);
    this.full = await loadImages(urls, prog);
    base = urls.length;
    this.follow = hasFollow ? await loadImages(furls, prog) : [];
    if (!hasFollow) this.mode = "full";
    this._size();
  }

  setStance(stance) {
    this.toeOnRight = stance !== "right_lead";
  }

  setMode(mode) {
    this.mode = this.overlay?.follow ? mode : "full";
    this._size();
    this.draw();
  }

  // Native size of the current source image.
  _src() {
    const o = this.overlay;
    return this.mode === "follow" ? [o.follow.width, o.follow.height] : [o.width, o.height];
  }

  // Match the canvas's pixel size to its on-screen box (crisp at any size / devicePixelRatio).
  resize() {
    if (!this.responsive) return;
    const r = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  _size() {
    const o = this.overlay;
    if (!o) return;
    if (this.responsive) {
      this.resize();
    } else {
      [this.canvas.width, this.canvas.height] = this._src();
    }
    if (this.inset) {
      this.inset.width = Math.round(o.width / 2);
      this.inset.height = Math.round(o.height / 2);
    }
  }

  // Where the source image goes on the canvas. The follow cam is centred on the rider, so it
  // *covers* the canvas: a wider box zooms in on the rider. The full frame is *contained*
  // (letterboxed), so nothing of the scene is lost.
  _placement() {
    const [iw, ih] = this._src();
    const cw = this.canvas.width, ch = this.canvas.height;
    const k = !this.responsive ? cw / iw
      : this.mode === "follow" ? Math.max(cw / iw, ch / ih) : Math.min(cw / iw, ch / ih);
    return { k, ox: (cw - iw * k) / 2, oy: (ch - ih * k) / 2, iw, ih };
  }

  // Map display-frame coordinates (the overlay's space) into canvas pixels.
  _mapper(i) {
    const { k, ox, oy } = this._placement();
    if (this.mode !== "follow") return { s: k, map: (p) => [p[0] * k + ox, p[1] * k + oy] };
    const [x0, y0, w] = this.overlay.follow.rects[i];
    const s = (this.overlay.follow.width / w) * k;
    return { s, map: (p) => [(p[0] - x0) * s + ox, (p[1] - y0) * s + oy] };
  }

  draw(i = state.frame, active = [], gauge = null) {
    const o = this.overlay;
    if (!o) return;
    const img = this.mode === "follow" ? this.follow[i] : this.full[i];
    if (!img) return;
    const { ctx, canvas } = this;
    const { k, ox, oy, iw, ih } = this._placement();
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, ox, oy, iw * k, ih * k);
    const f = o.frames[i];
    const { s, map } = this._mapper(i);
    const L = state.layers;
    // line widths follow the size the picture is shown at (not the canvas), so they look the
    // same whether the column is narrow or wide
    const lw = Math.max(2, Math.min(iw * k, canvas.width) / 260);
    const sev = active.some((e) => e.severity === "strong") ? "strong" : active.length ? "moderate" : null;
    const hot = new Set(active.map((e) => e.metric));

    if (L.matEdge && this.mode === "full" && o.mat_edge.length) {
      ctx.strokeStyle = C.mat; ctx.lineWidth = lw; ctx.setLineDash([6, 6]);
      ctx.beginPath();
      o.mat_edge.forEach((p, k) => (k ? ctx.lineTo(...map(p)) : ctx.moveTo(...map(p))));
      ctx.stroke(); ctx.setLineDash([]);
    }
    if (!f || !f.kp) { this._inset(i, f, sev); return; }
    const kp = f.kp.map(map);

    if (L.skeleton) {
      ctx.lineWidth = lw; ctx.lineCap = "round";
      for (const [a, b] of o.edges) {
        const conf = Math.min(f.sc[a], f.sc[b]);
        ctx.strokeStyle = conf >= 0.35 ? C.skeleton : C.lowConf;
        ctx.beginPath(); ctx.moveTo(...kp[a]); ctx.lineTo(...kp[b]); ctx.stroke();
      }
      ctx.fillStyle = C.joint;
      for (let k = 0; k < kp.length; k++) {
        if (f.sc[k] < 0.35) continue;
        ctx.beginPath(); ctx.arc(kp[k][0], kp[k][1], lw * 1.1, 0, 2 * Math.PI); ctx.fill();
      }
    }
    // highlighted body parts for the active events
    for (const m of hot) {
      const color = C[active.find((e) => e.metric === m).severity];
      for (const [a, b] of PART_EDGES[m] || []) {
        ctx.strokeStyle = color; ctx.lineWidth = lw * 3; ctx.lineCap = "round";
        ctx.beginPath(); ctx.moveTo(...kp[a]); ctx.lineTo(...kp[b]); ctx.stroke();
      }
      if (PART_JOINT[m] != null) {
        const [x, y] = kp[PART_JOINT[m]];
        ctx.strokeStyle = color; ctx.lineWidth = lw * 1.5;
        ctx.beginPath(); ctx.arc(x, y, lw * 7, 0, 2 * Math.PI); ctx.stroke();
      }
    }
    const boardHot = [...hot].some((m) => BOARD_METRICS.has(m));
    if ((L.board || boardHot) && f.board[0] && f.board[1]) {
      const [t, n] = f.board.map(map);
      ctx.strokeStyle = boardHot ? C[sev] : C.board; ctx.lineWidth = lw * (boardHot ? 4 : 2.2);
      ctx.beginPath(); ctx.moveTo(...t); ctx.lineTo(...n); ctx.stroke();
      ctx.fillStyle = ctx.strokeStyle;
      ctx.beginPath(); ctx.arc(n[0], n[1], lw * 2, 0, 2 * Math.PI); ctx.fill();
    }
    const comEv = active.find((e) => COM_METRICS.has(e.metric));
    if ((L.com || comEv) && f.com) {
      const [x, y] = map(f.com);
      if (f.contact) {
        ctx.strokeStyle = C.com; ctx.lineWidth = lw; ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(...map(f.contact)); ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.fillStyle = comEv ? C[comEv.severity] : C.com;
      ctx.beginPath(); ctx.arc(x, y, lw * (comEv ? 4 : 2.6), 0, 2 * Math.PI); ctx.fill();
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke();
      if (comEv) this._arrow(x, y, comEv, lw);
    }
    if (f.rider) {
      const [x0, y0] = map(f.rider.slice(0, 2));
      const [x1, y1] = map(f.rider.slice(2));
      if (L.bbox) {
        ctx.strokeStyle = sev ? C[sev] : C.box;
        ctx.lineWidth = sev ? lw * 2 : lw * 0.7;
        ctx.setLineDash(sev ? [] : [8, 6]);
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
        ctx.setLineDash([]);
      }
      if (sev) this._tags(x0, y0, active, lw); // the event labels show with or without the box
    }
    // on top of the box, so its dashed border never crosses the tilt label
    if (L.board && gauge && f.board[0] && f.board[1]) {
      this._boardTilt(f, map, gauge, lw, hot.has("edge_change") ? sev : null);
    }
    this._inset(i, f, sev);
  }

  // Board tilt drawn on the board itself, at its tail (the end nearest the camera). The bar is
  // the board's width seen from behind: level when flat, the toe end dropping on a toe edge.
  // The toe end sits on the rider's toe side of the picture (image-left for goofy).
  // Green = toe edge, blue = heel edge, white = flat, grey = not measurable here.
  _boardTilt(f, map, g, lw, sev) {
    const { ctx } = this;
    const v = g.value;
    const usable = v != null && Number.isFinite(v) && g.validity !== "invalid";
    const color = !usable ? "#94a3b8" : Math.abs(v) < 3 ? "#f8fafc" : v > 0 ? "#10b981" : "#60a5fa";
    const side = this.toeOnRight ? 1 : -1;
    const fromBoard = f.edge_src === "board";
    let tail, half;
    if (f.edge_anchor && f.edge_half) {
      // the board tail's centre and width, smoothed over time: the bar glides with the board
      tail = map(f.edge_anchor);
      const [ax1] = map([f.edge_anchor[0] + f.edge_half, f.edge_anchor[1]]);
      half = Math.max(lw * 6, Math.abs(ax1 - tail[0]));
    } else {
      const [trail, lead] = f.board.map(map);
      // the tail extends past the rear binding, away from the nose
      tail = [trail[0] + (trail[0] - lead[0]) * 0.35, trail[1] + (trail[1] - lead[1]) * 0.35];
      const rider = f.rider ? map(f.rider.slice(2))[0] - map(f.rider.slice(0, 2))[0] : lw * 40;
      half = Math.max(lw * 9, rider * 0.28);
    }
    // the bar's tilt is the smoothed edge angle (not a single frame's raw detection)
    const a = usable ? (v * Math.PI) / 180 : 0;
    const dx = Math.cos(a) * half, dy = Math.sin(a) * half;
    const toeEnd = [tail[0] + side * dx, tail[1] + dy];
    const heelEnd = [tail[0] - side * dx, tail[1] - dy];
    ctx.save();
    // level reference
    ctx.strokeStyle = "rgba(255,255,255,0.55)"; ctx.lineWidth = Math.max(1, lw * 0.6); ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(tail[0] - half * 1.15, tail[1]); ctx.lineTo(tail[0] + half * 1.15, tail[1]); ctx.stroke();
    ctx.setLineDash([]);
    // the board's width, tilted; dark outline so it reads on the white mat
    ctx.lineCap = "round";
    ctx.strokeStyle = "rgba(0,0,0,0.65)"; ctx.lineWidth = lw * 3.6;
    ctx.beginPath(); ctx.moveTo(...heelEnd); ctx.lineTo(...toeEnd); ctx.stroke();
    ctx.strokeStyle = sev ? C[sev] : color; ctx.lineWidth = lw * 2.4;
    ctx.beginPath(); ctx.moveTo(...heelEnd); ctx.lineTo(...toeEnd); ctx.stroke();
    // T / H end markers
    const fs = Math.round(Math.max(11, lw * 5));
    ctx.font = `700 ${fs}px system-ui, sans-serif`;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    const tag = (pt, text, dir) => {
      const x = pt[0] + dir * fs * 0.9, y = pt[1];
      ctx.fillStyle = "rgba(11,15,20,0.75)";
      ctx.beginPath(); ctx.arc(x, y, fs * 0.72, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = "#fff"; ctx.fillText(text, x, y + 1);
    };
    tag(toeEnd, "T", side);
    tag(heelEnd, "H", -side);
    // angle label under the tail
    const label = (!usable ? "edge –" : Math.abs(v) < 3 ? `flat ${Math.abs(v).toFixed(0)}°`
      : `${v > 0 ? "toe" : "heel"} ${Math.abs(v).toFixed(0)}°${g.err != null ? ` ±${g.err.toFixed(0)}` : ""}`)
      + (usable && !fromBoard ? " · feet" : "");
    ctx.font = `700 ${fs}px system-ui, sans-serif`;
    const w = ctx.measureText(label).width + 12, h = fs + 8;
    const ly = tail[1] + half * 0.55 + h / 2 + 4;
    ctx.fillStyle = "rgba(11,15,20,0.8)";
    ctx.beginPath(); ctx.roundRect(tail[0] - w / 2, ly - h / 2, w, h, 5); ctx.fill();
    if (sev) { ctx.strokeStyle = C[sev]; ctx.lineWidth = 2; ctx.stroke(); }
    ctx.fillStyle = color; ctx.fillText(label, tail[0], ly + 1);
    ctx.restore();
  }

  _arrow(x, y, e, lw) {
    const { ctx } = this;
    const len = lw * 14;
    // com_height: + is up; com_lateral / fore-aft: + is image right
    const [dx, dy] = e.metric === "com_height" ? [0, -Math.sign(e.delta)] : [Math.sign(e.delta), 0];
    const tx = x + dx * len, ty = y + dy * len;
    ctx.strokeStyle = C[e.severity]; ctx.fillStyle = C[e.severity]; ctx.lineWidth = lw * 1.6;
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(tx, ty); ctx.stroke();
    const a = Math.atan2(ty - y, tx - x);
    ctx.beginPath();
    ctx.moveTo(tx + Math.cos(a) * lw * 3, ty + Math.sin(a) * lw * 3);
    ctx.lineTo(tx + Math.cos(a + 2.4) * lw * 3, ty + Math.sin(a + 2.4) * lw * 3);
    ctx.lineTo(tx + Math.cos(a - 2.4) * lw * 3, ty + Math.sin(a - 2.4) * lw * 3);
    ctx.fill();
  }

  _tags(x0, y0, active, lw) {
    const { ctx } = this;
    const fs = Math.round(lw * 6.5);
    ctx.font = `600 ${fs}px system-ui, sans-serif`;
    let y = Math.max(fs + 6, y0 - 6);
    const lines = active.map((e) => e.text);
    y -= (lines.length - 1) * (fs + 6);
    y = Math.max(fs + 6, y);
    for (const [k, text] of lines.entries()) {
      const w = ctx.measureText(text).width + 12;
      const x = Math.max(4, Math.min(this.canvas.width - w - 4, x0));
      ctx.fillStyle = C[active[k].severity];
      ctx.fillRect(x, y - fs - 2, w, fs + 8);
      ctx.fillStyle = "#fff";
      ctx.fillText(text, x + 6, y + 1);
      y += fs + 10;
    }
  }

  // Context inset: the whole frame, with the follow-cam window and the rider box.
  _inset(i, f, sev) {
    if (!this.ictx) return;
    const { ictx, inset } = this;
    const img = this.mode === "follow" ? this.full[i] : this.follow[i];
    if (!img) { inset.style.display = "none"; return; }
    inset.style.display = "";
    const o = this.overlay;
    if (this.mode === "follow") {
      inset.width = Math.round(o.width / 2); inset.height = Math.round(o.height / 2);
      ictx.drawImage(img, 0, 0, inset.width, inset.height);
      const k = inset.width / o.width;
      const [x0, y0, w, h] = o.follow.rects[i];
      ictx.strokeStyle = "rgba(255,255,255,0.8)"; ictx.lineWidth = 1.5;
      ictx.strokeRect(x0 * k, y0 * k, w * k, h * k);
      if (f?.rider) {
        const [a, b, c, d] = f.rider;
        ictx.strokeStyle = sev ? C[sev] : C.box; ictx.lineWidth = sev ? 3 : 1.5;
        ictx.strokeRect(a * k, b * k, (c - a) * k, (d - b) * k);
      }
    } else {
      inset.width = Math.round(o.follow.width / 3); inset.height = Math.round(o.follow.height / 3);
      ictx.drawImage(img, 0, 0, inset.width, inset.height);
      if (sev) { ictx.strokeStyle = C[sev]; ictx.lineWidth = 4; ictx.strokeRect(2, 2, inset.width - 4, inset.height - 4); }
    }
  }
}
