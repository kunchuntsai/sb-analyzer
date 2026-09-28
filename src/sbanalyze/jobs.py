"""Background processing queue for clips imported from the UI.

One worker thread processes clips in order, so only one pose model is ever loaded. The model is
kept between jobs. Job state is kept in memory; the results themselves go to the store as usual.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import Config
from .contracts import PipelineError
from .store import Store

log = logging.getLogger("sbanalyze")


@dataclass
class Job:
    id: str
    path: str
    name: str
    session_id: str | None
    status: str = "queued"  # queued | running | done | failed
    stage: str = "Waiting"
    progress: float = 0.0
    clip_id: str | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


class JobQueue:
    def __init__(self, cfg: Config, store: Store):
        self.cfg, self.store = cfg, store
        self.jobs: dict[str, Job] = {}
        self._q: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._backend = None
        self._worker: threading.Thread | None = None

    def submit(self, path: Path, session_id: str | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:10], path=str(path), name=path.name, session_id=session_id)
        with self._lock:
            self.jobs[job.id] = job
        self._q.put(job.id)
        self._ensure_worker()
        return job

    def list(self) -> list[dict]:
        with self._lock:
            return [j.as_dict() for j in sorted(self.jobs.values(), key=lambda j: j.created)]

    def active(self, path: str) -> Job | None:
        with self._lock:
            return next((j for j in self.jobs.values()
                         if j.path == path and j.status in ("queued", "running")), None)

    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run, name="sbanalyze-jobs", daemon=True)
            self._worker.start()

    def _run(self) -> None:
        from .pipeline import _backend, process

        while True:
            job = self.jobs[self._q.get()]
            job.status, job.stage, job.progress = "running", "Starting", 0.0

            def report(stage: str, frac: float, job: Job = job) -> None:
                job.stage, job.progress = stage, round(float(frac), 3)

            try:
                if self._backend is None:
                    report("Loading the pose model", 0.01)
                    self._backend = _backend(self.cfg)
                job.clip_id = process(Path(job.path), self.cfg, self.store,
                                      session_id=job.session_id, backend=self._backend,
                                      progress=report)
                job.status, job.stage, job.progress = "done", "Done", 1.0
            except PipelineError as e:
                job.status, job.error = "failed", f"{e.stage}: {e} ({e.status})"
            except Exception as e:  # keep the worker alive for the next clip
                log.exception("job %s failed", job.id)
                job.status, job.error = "failed", f"{type(e).__name__}: {e}"
            finally:
                job.finished = time.time()
                self._q.task_done()
