// Video library: a slide-out list of every run, with thumbnails and key numbers. Click a card to
// open it. Add videos with the button or by dropping files/folders onto the panel. Remove or
// re-analyse a run from its card. Videos being processed appear at the top with progress.
const $ = (s, root = document) => root.querySelector(s);
const fmt = (v, d = 2) => (v == null || !Number.isFinite(v) ? "–" : v.toFixed(d));

export class Library {
  constructor({ getClips, getCurrent, onOpen, onChanged, importer }) {
    this.getClips = getClips;
    this.getCurrent = getCurrent;
    this.onOpen = onOpen;
    this.onChanged = onChanged;
    this.importer = importer;
    this.panel = $("#library");
    this.jobs = [];
    this.confirming = null; // clip_id awaiting "Remove?" confirmation
    this.filter = "";
    this.poll = null;
    this._wire();
  }

  open() {
    this.panel.classList.add("open");
    this.panel.setAttribute("aria-hidden", "false");
    $("#libraryScrim").hidden = false;
    this.refreshJobs();
    this.render();
    $("#librarySearch").focus({ preventScroll: true });
  }

  close() {
    this.panel.classList.remove("open");
    this.panel.setAttribute("aria-hidden", "true");
    $("#libraryScrim").hidden = true;
    this.confirming = null;
  }

  get isOpen() { return this.panel.classList.contains("open"); }

  async refreshJobs() {
    this.jobs = await fetch("/jobs").then((r) => r.json()).catch(() => []);
    const active = this.jobs.some((j) => j.status === "queued" || j.status === "running");
    if (this.isOpen) this.render();
    clearTimeout(this.poll);
    if (active) this.poll = setTimeout(() => this.refreshJobs(), 1200);
  }

  label(c) {
    const when = (c.recorded_at || "").replace("T", " ").slice(0, 16);
    return { when, name: c.file_name || c.clip_id };
  }

  render() {
    const clips = this.getClips();
    const current = this.getCurrent();
    const q = this.filter.toLowerCase();
    const shown = clips.filter((c) => !q || `${c.recorded_at} ${c.file_name}`.toLowerCase().includes(q));
    const active = this.jobs.filter((j) => j.status === "queued" || j.status === "running");
    const failed = this.jobs.filter((j) => j.status === "failed" && !this.dismissed?.has(j.id));
    $("#libraryCount").textContent = `${clips.filter((c) => c.status === "ok").length} video${clips.length === 1 ? "" : "s"}`;

    const jobCards = [...active, ...failed].map((j) => `
      <div class="lib-card job ${j.status}">
        <div class="thumb placeholder">${j.status === "failed" ? "⚠" : "⏳"}</div>
        <div class="meta">
          <b title="${j.name}">${j.name}</b>
          <span class="muted">${j.status === "failed" ? `failed: ${j.error ?? ""}` : j.status === "queued" ? "waiting…" : `${j.stage} · ${Math.round(100 * j.progress)}%`}</span>
          ${j.status === "failed" ? `<button class="link dismiss" data-job="${j.id}">dismiss</button>`
            : `<i class="bar"><b style="width:${Math.round(100 * j.progress)}%"></b></i>`}
        </div>
      </div>`).join("");

    const cards = shown.map((c) => {
      const { when, name } = this.label(c);
      const s = c.summary || {};
      const ok = c.status === "ok";
      const confirming = this.confirming === c.clip_id;
      const thumb = ok && c.thumb_frame != null
        ? `<img loading="lazy" src="/clips/${c.clip_id}/follow/${c.thumb_frame}" alt="">`
        : `<div class="thumb placeholder">${ok ? "🎞" : "⚠"}</div>`;
      return `
      <div class="lib-card ${c.clip_id === current ? "current" : ""} ${ok ? "" : "bad"}" data-clip="${c.clip_id}" tabindex="0">
        <div class="thumb">${thumb}</div>
        <div class="meta">
          <b>${when}</b>
          <span class="muted fname" title="${c.path}">${name}${c.uploaded_copy ? "" : ' <span class="tag" title="Processed in place from your folder">linked</span>'}</span>
          ${ok ? `<span class="stats">
              <span title="Air time">✈ ${fmt(s.air_time_s)} s</span>
              <span title="Jump height (board above lip)">↥ ${fmt(s.jump_height_m)} m</span>
              <span title="Sudden movements (strong)" class="${s.strong ? "warn" : ""}">⚠ ${s.events ?? 0}${s.strong ? ` (${s.strong})` : ""}</span>
            </span>` : `<span class="err">${c.status.replaceAll("_", " ")}</span>`}
        </div>
        <div class="acts">
          ${confirming ? `
            <span class="ask">Remove?</span>
            <button class="yes" data-act="remove-yes" title="Remove this run">Yes</button>
            <button data-act="remove-no">No</button>` : `
            <button data-act="reprocess" title="Analyse again">↻</button>
            <button data-act="remove" title="Remove from the library">🗑</button>`}
        </div>
      </div>`;
    }).join("");

    $("#libraryList").innerHTML = jobCards + (cards || (jobCards ? "" : `
      <div class="lib-empty">No videos yet.<br>Click <b>＋ Add videos</b> or drop videos here.</div>`));

    this.panel.querySelectorAll(".lib-card[data-clip]").forEach((el) => {
      const id = el.dataset.clip;
      el.addEventListener("click", (e) => {
        const act = e.target.closest("button")?.dataset.act;
        if (act) { e.stopPropagation(); this.action(act, id); return; }
        const c = this.getClips().find((x) => x.clip_id === id);
        if (c?.status === "ok") { this.onOpen(id); this.close(); }
      });
      el.addEventListener("keydown", (e) => {
        if (e.key === "Enter") el.click();
        if (e.key === "Delete" || e.key === "Backspace") { this.confirming = id; this.render(); }
      });
    });
    this.panel.querySelectorAll(".dismiss").forEach((b) => b.addEventListener("click", () => {
      (this.dismissed ??= new Set()).add(b.dataset.job);
      this.render();
    }));
  }

  async action(act, id) {
    const c = this.getClips().find((x) => x.clip_id === id);
    if (act === "remove") { this.confirming = id; this.render(); return; }
    if (act === "remove-no") { this.confirming = null; this.render(); return; }
    if (act === "remove-yes") {
      this.confirming = null;
      const r = await fetch(`/clips/${id}`, { method: "DELETE" });
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        this.toast(d.detail || "Could not remove this run.");
      } else {
        const d = await r.json();
        this.toast(d.removed_source ? "Removed the run and its uploaded copy."
          : "Removed the run. Your original video file was not touched.");
      }
      await this.onChanged({ removed: id });
      this.render();
      return;
    }
    if (act === "reprocess" && c) {
      const r = await fetch("/jobs", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paths: [c.path] }),
      });
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        this.toast(d.detail || "Could not start the analysis.");
      } else {
        this.toast("Re-analysing… it will update when done.");
        this.refreshJobs();
        this.importer.watch(); // the importer's poller refreshes the list when jobs finish
      }
    }
  }

  toast(msg) {
    const t = $("#libraryToast");
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(this.toastTimer);
    this.toastTimer = setTimeout(() => (t.hidden = true), 3500);
  }

  _wire() {
    $("#libraryBtn").addEventListener("click", () => (this.isOpen ? this.close() : this.open()));
    $("#libraryClose").addEventListener("click", () => this.close());
    $("#libraryScrim").addEventListener("click", () => this.close());
    $("#libraryAdd").addEventListener("click", () => { this.close(); this.importer.open(); });
    $("#librarySearch").addEventListener("input", (e) => { this.filter = e.target.value; this.render(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && this.isOpen) this.close(); });
    // drop videos or folders straight onto the panel
    const p = this.panel;
    p.addEventListener("dragover", (e) => { e.preventDefault(); p.classList.add("over"); });
    p.addEventListener("dragleave", (e) => { if (!p.contains(e.relatedTarget)) p.classList.remove("over"); });
    p.addEventListener("drop", async (e) => {
      e.preventDefault();
      p.classList.remove("over");
      this.close();
      this.importer.open("upload");
      await this.importer._dropEntries(e.dataTransfer);
    });
  }
}
