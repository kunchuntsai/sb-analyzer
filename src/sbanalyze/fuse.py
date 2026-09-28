"""Optional second-view merge: source selection, not triangulation (ADR-011).

The only module that knows more than one camera can exist. Same type in, same type out.
"""

from __future__ import annotations

from .contracts import MetricSample
from .metrics.validity import RANK


def select_best_source(primary: list[MetricSample], secondary: list[MetricSample]
                       ) -> list[MetricSample]:
    """Per (frame, metric): the better validity rating wins; ties go to the lower err_est.

    `secondary` must already be resampled onto the primary frame grid, in run time.
    """
    best: dict[tuple[int, str], MetricSample] = {(s.frame_idx, s.metric): s for s in primary}
    for s in secondary:
        key = (s.frame_idx, s.metric)
        cur = best.get(key)
        if cur is None:
            best[key] = s
            continue
        a, b = RANK[s.validity], RANK[cur.validity]
        if a > b or (a == b and s.err_est < cur.err_est):
            best[key] = s
    return sorted(best.values(), key=lambda s: (s.metric, s.frame_idx))


def fuse(primary: list[MetricSample], secondary: list[MetricSample] | None
         ) -> list[MetricSample]:
    if secondary is None:
        return primary  # the normal case
    return select_best_source(primary, secondary)
