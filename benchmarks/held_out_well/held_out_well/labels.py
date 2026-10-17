"""Label semantics shared by the split, the grid and the scorer.

3W 2.0.0 row labels: 0 normal, 1..9 steady event class, 101..109
transient of event 1..9, NA unlabelled (stored here as -1). The event
class of a row is its label modulo 100: a transient is merged with its
event.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .constants import MIN_EPISODE_ROWS, NA, TRANSIENT_OFFSET


def event_class(y: Any) -> Any:
    """Map raw labels to event classes (transients merged)."""
    y = np.asarray(y)
    return np.where(y >= TRANSIENT_OFFSET, y - TRANSIENT_OFFSET, y)


def runs_from_labels(values: Any) -> list[list[int]]:
    """Run-length encode a label vector as [label, start, end) rows.

    NA must already be coded as ``NA`` (-1).
    """
    v = np.asarray(values, dtype=np.int64)
    if v.size == 0:
        return []
    cut = np.flatnonzero(np.diff(v) != 0) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, v.size]
    return [[int(v[s]), int(s), int(e)] for s, e in zip(starts, ends)]


def episodes_from_runs(
    class_runs: Sequence[Sequence[int]],
) -> list[dict[str, Any]]:
    """Turn the label runs of one file into episodes.

    Adjacent runs of the same event class above 0 (a transient followed
    by its steady state) form one episode. NA runs and normal runs
    separate episodes. ``known_onset`` is true when the run just before
    the episode is labelled normal inside the same file, i.e. the
    normal-to-event transition is observed.
    """
    out: list[dict[str, Any]] = []
    prev: dict[str, int] | None = None
    for lab, s, e in class_runs:
        ev = int(event_class(lab)) if lab >= 0 else NA
        if ev > 0:
            if out and prev is not None and prev["ev"] == ev and out[-1]["end"] == s:
                out[-1]["end"] = e
            else:
                known = prev is not None and prev["ev"] == 0 and prev["end"] == s
                out.append(
                    {
                        "event": ev,
                        "start": s,
                        "end": e,
                        "known_onset": bool(known),
                    }
                )
        prev = {"ev": ev, "end": e}
    return out


def class_support(
    class_runs: Sequence[Sequence[int]],
    min_rows: int = MIN_EPISODE_ROWS,
) -> tuple[dict[int, int], list[int]]:
    """Rows per event class and the supported classes of one file.

    An episode shorter than ``min_rows`` does not count as support.
    """
    rows: dict[int, int] = {}
    for lab, s, e in class_runs:
        if lab < 0:
            continue
        ev = int(event_class(lab))
        rows[ev] = rows.get(ev, 0) + (e - s)
    eps = episodes_from_runs(class_runs)
    supported = sorted(
        {ep["event"] for ep in eps if ep["end"] - ep["start"] >= min_rows}
    )
    return rows, supported
