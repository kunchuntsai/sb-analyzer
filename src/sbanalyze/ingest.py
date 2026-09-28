"""Decode and normalise: apply rotation side data explicitly, key frames on presentation time."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import av
import numpy as np

from .contracts import ClipMeta, ViewRole

_FILENAME_TS = re.compile(r"(\d{8})-(\d{6})")


def clip_id_for(path: Path) -> str:
    """Stable id: file name stem plus a short content hash of the first 1 MB."""
    with open(path, "rb") as f:
        head = f.read(1 << 20)
    return f"{path.stem}-{hashlib.sha1(head).hexdigest()[:6]}"


def _recorded_at(path: Path, container: av.container.InputContainer) -> datetime:
    # Phone exports often rewrite creation_time to the export date; the capture app's
    # file name keeps the real capture time, so prefer it. Times are local wall-clock, naive.
    m = _FILENAME_TS.search(path.name)
    if m:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")  # noqa: DTZ007
    ct = container.metadata.get("creation_time")
    if ct:
        return datetime.fromisoformat(ct).replace(tzinfo=None)
    return datetime.fromtimestamp(path.stat().st_mtime)  # noqa: DTZ006


def _rotation(container: av.container.InputContainer) -> int:
    stream = container.streams.video[0]
    for frame in container.decode(stream):
        return int(getattr(frame, "rotation", 0) or 0)
    return 0


def probe(path: Path, run_id: str, view: ViewRole = ViewRole.FALL_LINE,
          t_offset_s: float = 0.0) -> ClipMeta:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        w, h = s.codec_context.width, s.codec_context.height
        fps = float(s.average_rate)
        n = s.frames or round(float(s.duration * s.time_base) * fps)
        recorded = _recorded_at(path, c)
        rot = _rotation(c)
    if rot % 180:
        w, h = h, w
    return ClipMeta(
        clip_id=clip_id_for(path), run_id=run_id, view=view, path=path,
        width=w, height=h, fps=fps, n_frames=n, rotation_deg=rot,
        recorded_at=recorded, t_offset_s=t_offset_s,
    )


def _apply_rotation(img: np.ndarray, rotation_deg: int) -> np.ndarray:
    # Display-matrix rotation is counter-clockwise degrees; np.rot90 with k>0 is also CCW.
    k = (rotation_deg // 90) % 4
    return np.ascontiguousarray(np.rot90(img, k=k)) if k else img


def iter_frames(meta: ClipMeta, start: int = 0, end: int | None = None,
                fmt: str = "bgr24", scale_width: int | None = None,
                ) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_idx, t_sec, image) in display orientation, t keyed on PTS.

    `scale_width` is the width *after* rotation; scaling happens in the decoder, which is far
    cheaper than decoding 4K and resizing in numpy.
    """
    with av.open(str(meta.path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        tb = float(s.time_base)
        first_pts = None
        for idx, frame in enumerate(c.decode(s)):
            if first_pts is None:
                first_pts = frame.pts or 0
            if idx < start:
                continue
            if end is not None and idx > end:
                break
            t = (frame.pts - first_pts) * tb if frame.pts is not None else idx / meta.fps
            if scale_width:
                # rotated clips are stored landscape; the display width is the coded height
                if meta.rotation_deg % 180:
                    tw, th = round(meta.height * scale_width / meta.width), scale_width
                else:
                    tw, th = scale_width, round(meta.height * scale_width / meta.width)
                img = frame.reformat(width=tw, height=th, format=fmt).to_ndarray()
            else:
                img = frame.to_ndarray(format=fmt)
            yield idx, float(t) + meta.t_offset_s, _apply_rotation(img, meta.rotation_deg)
