"""Batch processor and local server."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Snowboard approach analyzer")


@app.command()
def process(paths: list[Path], session: str | None = None,
            start: int | None = None, end: int | None = None,
            reuse_poses: bool = typer.Option(False, help="Recompute metrics from cached poses")
            ) -> None:
    """Import and analyse clips. The session defaults to the recording date."""
    from .config import load_config
    from .contracts import PipelineError
    from .pipeline import process as run
    from .store import Store

    cfg = load_config()
    store = Store(cfg.data_dir)
    manual = (start, end) if start is not None and end is not None else None
    backend = None
    for p in paths:
        try:
            if backend is None and not reuse_poses:
                from .pipeline import _backend
                backend = _backend(cfg)
            cid = run(p, cfg, store, session_id=session, manual_window=manual, backend=backend,
                      reuse_poses=reuse_poses)
            typer.echo(f"ok   {p.name} -> {cid}")
        except PipelineError as e:
            typer.echo(f"FAIL {p.name}: {e} (status={e.status})", err=True)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve the timeline UI at http://HOST:PORT."""
    import uvicorn

    uvicorn.run("sbanalyze.api:app", host=host, port=port, log_level="warning")


if __name__ == "__main__":
    app()
