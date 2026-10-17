"""Label semantics shared by manifest, scorer and baselines (numpy only).

3W 2.0.0 row labels: 0 normal, 1..9 steady event class, 101..109 transient of event 1..9, NA unlabelled.
We store NA as -1. The *event class* of a row is label mod 100 (transient merged with its event).
"""
from __future__ import annotations

import numpy as np

TRANSIENT_OFFSET = 100
NA = -1
EVENT_NAMES = {
    0: "normal", 1: "abrupt_increase_bsw", 2: "spurious_closure_dhsv", 3: "severe_slugging", 4: "flow_instability",
    5: "rapid_productivity_loss", 6: "quick_restriction_pck", 7: "scaling_pck", 8: "hydrate_production_line",
    9: "hydrate_service_line",
}
MIN_SUPPORTED_EPISODE_ROWS = 300  # an episode shorter than 5 minutes does not count as class support


def event_class(y):
    """Map raw labels to event classes (transients merged); NA stays -1."""
    y = np.asarray(y)
    return np.where(y >= TRANSIENT_OFFSET, y - TRANSIENT_OFFSET, y)


def episodes_from_runs(class_runs):
    """class_runs: list of [label, start, end_exclusive] for one file -> list of episode dicts.

    Adjacent runs with the same event class >0 (e.g. transient 102 then steady 2) form one episode.
    NA runs and normal runs separate episodes. ``known_onset`` is true when the run immediately before
    the episode is labelled normal (a normal->event transition is observed inside the file).
    """
    out = []
    prev = None
    for lab, s, e in class_runs:
        ev = int(event_class(lab)) if lab >= 0 else -1
        if ev > 0:
            if out and prev is not None and prev["ev"] == ev and out[-1]["end"] == s:
                out[-1]["end"] = e
            else:
                out.append({"event": ev, "start": s, "end": e, "known_onset": bool(prev is not None and prev["ev"] == 0 and prev["end"] == s)})
        prev = {"ev": ev, "end": e}
    return out


def class_support(class_runs, min_rows=MIN_SUPPORTED_EPISODE_ROWS):
    """Rows per event class and the set of supported classes for one file."""
    rows = {}
    for lab, s, e in class_runs:
        if lab < 0:
            continue
        ev = int(event_class(lab))
        rows[ev] = rows.get(ev, 0) + (e - s)
    eps = episodes_from_runs(class_runs)
    supported = sorted({ep["event"] for ep in eps if ep["end"] - ep["start"] >= min_rows})
    return rows, supported
