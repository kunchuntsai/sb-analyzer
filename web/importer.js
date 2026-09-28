// Import dialog: pick a folder (or files, or drag & drop) in the browser and upload the videos,
// or browse a folder on this computer and process the videos in place. Processing runs in a
// background queue on the server; progress is polled and shown here and in the header.
const VIDEO = /\.(mov|mp4|m4v)$/i;
const $ = (s, root = document) => root.querySelector(s);

const safeName = (n) => n.split("/").pop().replace(/[^A-Za-z0-9._-]/g, "_");
const baseName = (p) => (p || "").split(/[\\/]/).pop();
const mb = (b) => `${(b / 1e6).toFixed(1)} MB`;

export class Importer {
  constructor({ onClipReady, getClips, onJobs = () => {} }) {
    this.onJobs = onJobs;
    this.dlg = $("#importDlg");
    this.onClipReady = onClipReady;
    this.getClips = getClips;
    this.items = []; // { key, name, size, where, file?, path?, checked, status, progress, jobId }
    this.jobs = new Map();
    this.seenDone = new Set();
    this.polling = null;
    this.tab = "upload";
    this._wire();
  }

  open(tab = null, path = null) {
    if (!this.dlg.open) this.dlg.showModal();
    if (path) this.browsed = true; // don't let the tab switch load the home folder first
    if (tab) this.dlg.querySelector(`.itabs button[data-tab="${tab}"]`)?.click();
    if (path) this.browse(path);
    else if (this.tab === "local" && !this.browsed) this.browse();
    this.render();
  }

  // ---- collecting candidates ---------------------------------------------------------------
  addFiles(files) {
    for (const f of files) {
      if (!VIDEO.test(f.name)) continue;
      const where = (f.webkitRelativePath || f.relPath || "").split("/").slice(0, -1).join("/");
      const key = `u:${where}/${f.name}:${f.size}`;
      if (this.items.some((it) => it.key === key)) continue;
      this.items.push({ key, name: f.name, size: f.size, where, file: f, checked: true, status: "new" });
    }
    this._markImported();
    this.render();
  }

  async _dropEntries(dt) {
    const out = [];
    const walk = async (entry, prefix) => {
      if (entry.isFile) {
        const f = await new Promise((res, rej) => entry.file(res, rej));
        f.relPath = `${prefix}${f.name}`;
        out.push(f);
      } else if (entry.isDirectory) {
        const reader = entry.createReader();
        let batch;
        do {
          batch = await new Promise((res, rej) => reader.readEntries(res, rej));
          for (const e of batch) await walk(e, `${prefix}${entry.name}/`);
        } while (batch.length);
      }
    };
    const entries = [...dt.items].map((i) => i.webkitGetAsEntry?.()).filter(Boolean);
    if (entries.length) for (const e of entries) await walk(e, "");
    else out.push(...dt.files);
    this.addFiles(out);
  }

  async browse(path = null) {
    this.browsed = true;
    const seq = (this.browseSeq = (this.browseSeq || 0) + 1);
    const box = $("#importBrowse");
    box.innerHTML = `<div class="muted">Loading…</div>`;
    const r = await fetch(`/browse${path ? `?path=${encodeURIComponent(path)}` : ""}`);
    const d = await r.json();
    if (seq !== this.browseSeq) return; // a newer folder was requested meanwhile
    if (!r.ok) { box.innerHTML = `<div class="err">${d.detail}</div>`; return; }
    $("#importPath").value = d.path;
    box.innerHTML = [
      d.parent ? `<div class="dir up" data-path="${d.parent}">⬑ ..</div>` : "",
      ...d.dirs.map((n) => `<div class="dir" data-path="${d.path}/${n}">📁 ${n}</div>`),
      d.videos.length ? `<div class="muted small">${d.videos.length} video${d.videos.length > 1 ? "s" : ""} in this folder</div>` : "",
      ...d.videos.map((v) => `<div class="vid">🎞 ${v.name} <span class="muted">${mb(v.size)}</span></div>`),
    ].join("");
    box.querySelectorAll(".dir").forEach((el) => el.addEventListener("click", () => this.browse(el.dataset.path)));
    this.folderVideos = d.videos;
    $("#importAddFolder").disabled = !d.videos.length;
    $("#importAddFolder").textContent = d.videos.length ? `Add ${d.videos.length} video${d.videos.length > 1 ? "s" : ""} from this folder` : "No videos in this folder";
  }

  addFolderVideos() {
    for (const v of this.folderVideos || []) {
      const key = `p:${v.path}`;
      if (this.items.some((it) => it.key === key)) continue;
      this.items.push({ key, name: v.name, size: v.size, where: v.path.replace(/\/[^/]*$/, ""), path: v.path, checked: true, status: "new" });
    }
    this._markImported();
    this.render();
  }

  _markImported() {
    const done = new Map(this.getClips().map((c) => [safeName(baseName(c.path)), c]));
    for (const it of this.items) {
      if (it.status === "new" && done.has(safeName(it.name))) {
        it.status = "imported";
        it.checked = false;
      }
    }
  }

  // ---- importing -----------------------------------------------------------------------------
  async start() {
    const todo = this.items.filter((it) => it.checked && ["new", "imported", "failed"].includes(it.status));
    if (!todo.length) return;
    const session = $("#importSession").value.trim() || null;
    $("#importGo").disabled = true;
    // paths on this computer: queue straight away
    const local = todo.filter((it) => it.path);
    if (local.length) {
      local.forEach((it) => (it.status = "queued"));
      const js = await this._post(local.map((it) => it.path), session);
      js.forEach((j, k) => (local[k].jobId = j.id));
    }
    this._poll();
    // browser files: upload one at a time, queue each as soon as it lands
    for (const it of todo.filter((x) => x.file)) {
      it.status = "uploading"; it.progress = 0; this.render();
      try {
        const res = await this._upload(it);
        it.status = "queued"; this.render();
        const [j] = await this._post([res.path], session);
        it.jobId = j.id;
      } catch (e) {
        it.status = "failed"; it.error = String(e);
      }
      this.render();
    }
    $("#importGo").disabled = false;
    this.render();
  }

  _upload(it) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("PUT", `/uploads/${encodeURIComponent(safeName(it.name))}`);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) { it.progress = e.loaded / e.total; this.renderRow(it); }
      };
      xhr.onload = () => (xhr.status < 300 ? resolve(JSON.parse(xhr.responseText))
        : reject(new Error(JSON.parse(xhr.responseText || "{}").detail || xhr.statusText)));
      xhr.onerror = () => reject(new Error("upload failed"));
      xhr.send(it.file);
    });
  }

  async _post(paths, session) {
    const r = await fetch("/jobs", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paths, session_id: session }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail);
    return d;
  }

  // start following background jobs (e.g. a re-analysis started from the library)
  watch() { this._poll(); }

  _poll() {
    if (this.polling) return;
    const tick = async () => {
      const list = await fetch("/jobs").then((r) => r.json()).catch(() => []);
      for (const j of list) this.jobs.set(j.id, j);
      for (const it of this.items) {
        const j = it.jobId && this.jobs.get(it.jobId);
        if (!j) continue;
        it.status = j.status === "running" ? "processing" : j.status;
        it.progress = j.progress; it.stage = j.stage; it.error = j.error;
      }
      // every job that finishes (from this dialog or from the library) updates the run list
      for (const j of list) {
        if ((j.status === "done" || j.status === "failed") && !this.seenDone.has(j.id)) {
          this.seenDone.add(j.id);
          await this.onClipReady(j.status === "done" ? j.clip_id : null);
        }
      }
      this.render();
      this._renderHeader(list);
      this.onJobs(list);
      const busy = list.some((j) => j.status === "queued" || j.status === "running")
        || this.items.some((it) => it.status === "uploading");
      this.polling = busy ? setTimeout(tick, 1000) : null;
    };
    this.polling = setTimeout(tick, 300);
  }

  // ---- rendering -----------------------------------------------------------------------------
  _statusHtml(it) {
    const pct = `${Math.round(100 * (it.progress || 0))}%`;
    switch (it.status) {
      case "imported": return `<span class="st done">already imported</span>`;
      case "uploading": return `<span class="st run">uploading ${pct}</span><i class="bar"><b style="width:${pct}"></b></i>`;
      case "queued": return `<span class="st">queued</span>`;
      case "processing": return `<span class="st run">${it.stage} · ${pct}</span><i class="bar"><b style="width:${pct}"></b></i>`;
      case "done": return `<span class="st done">✓ ready</span>`;
      case "failed": return `<span class="st err" title="${it.error || ""}">failed</span>`;
      default: return `<span class="st new">new</span>`;
    }
  }

  renderRow(it) {
    const row = this.dlg.querySelector(`[data-key="${CSS.escape(it.key)}"] .status`);
    if (row) row.innerHTML = this._statusHtml(it);
  }

  render() {
    const list = $("#importList");
    if (!this.items.length) {
      list.innerHTML = `<div class="muted empty">No videos selected yet.</div>`;
    } else {
      list.innerHTML = this.items.map((it) => `
        <label class="row" data-key="${it.key}">
          <input type="checkbox" ${it.checked ? "checked" : ""} ${["uploading", "queued", "processing", "done"].includes(it.status) ? "disabled" : ""}>
          <span class="nm" title="${it.where}/${it.name}">${it.name}<small>${it.where || ""}</small></span>
          <span class="sz">${mb(it.size)}</span>
          <span class="status">${this._statusHtml(it)}</span>
        </label>`).join("");
      list.querySelectorAll(".row").forEach((row) => {
        const it = this.items.find((x) => x.key === row.dataset.key);
        row.querySelector("input").addEventListener("change", (e) => { it.checked = e.target.checked; this._count(); });
      });
    }
    this._count();
  }

  _count() {
    const n = this.items.filter((it) => it.checked && ["new", "imported", "failed"].includes(it.status)).length;
    $("#importGo").textContent = n ? `Import ${n} video${n > 1 ? "s" : ""}` : "Import";
    $("#importGo").disabled = !n;
  }

  _renderHeader(list) {
    const active = list.filter((j) => j.status === "queued" || j.status === "running");
    const btn = $("#importBtn");
    if (!active.length) { btn.textContent = "＋ Import videos"; btn.classList.remove("busy"); return; }
    const run = active.find((j) => j.status === "running");
    btn.classList.add("busy");
    btn.textContent = `Processing ${active.length} · ${run ? Math.round(100 * run.progress) : 0}%`;
  }

  _wire() {
    $("#importBtn").addEventListener("click", () => this.open());
    $("#importClose").addEventListener("click", () => this.dlg.close());
    $("#importGo").addEventListener("click", () => this.start());
    $("#importPickDir").addEventListener("click", () => $("#importDirInput").click());
    $("#importPickFiles").addEventListener("click", () => $("#importFileInput").click());
    $("#importDirInput").addEventListener("change", (e) => { this.addFiles(e.target.files); e.target.value = ""; });
    $("#importFileInput").addEventListener("change", (e) => { this.addFiles(e.target.files); e.target.value = ""; });
    $("#importClear").addEventListener("click", () => {
      this.items = this.items.filter((it) => ["uploading", "queued", "processing"].includes(it.status));
      this.render();
    });
    const drop = $("#importDrop");
    drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("over"));
    drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); this._dropEntries(e.dataTransfer); });
    this.dlg.querySelectorAll(".itabs button").forEach((b) => b.addEventListener("click", () => {
      this.tab = b.dataset.tab;
      this.dlg.querySelectorAll(".itabs button").forEach((x) => x.classList.toggle("on", x === b));
      $("#importTabUpload").hidden = this.tab !== "upload";
      $("#importTabLocal").hidden = this.tab !== "local";
      if (this.tab === "local" && !this.browsed) this.browse();
    }));
    $("#importGoPath").addEventListener("click", () => this.browse($("#importPath").value));
    $("#importPath").addEventListener("keydown", (e) => { if (e.key === "Enter") this.browse(e.target.value); });
    $("#importAddFolder").addEventListener("click", () => this.addFolderVideos());
    // pick up jobs already running (e.g. after a page reload)
    fetch("/jobs").then((r) => r.json()).then((list) => {
      // jobs that finished before this page loaded are old news
      list.filter((j) => j.status === "done" || j.status === "failed").forEach((j) => this.seenDone.add(j.id));
      if (list.some((j) => j.status === "queued" || j.status === "running")) this._poll();
    }).catch(() => {});
  }
}
