"""The decision grid: where a model decides and what counts.

Decisions sit on a fixed grid of rows of each instance: one every
``STRIDE`` rows (30 s) from row ``FIRST_ROW`` (3,600), so that a full
causal lookback exists. A decision is eligible when at least 90 % of the
previous 900 rows have a finite value in at least one of the 18 process
channels. Ineligible decisions are excluded from every denominator.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .constants import ELIG_FRAC, ELIG_WINDOW, FIRST_ROW, STRIDE
from .labels import event_class


def n_decisions(rows: int) -> int:
    """Number of decisions of an instance with ``rows`` rows."""
    return max(0, -(-(rows - FIRST_ROW) // STRIDE))


def decision_rows(rows: int) -> np.ndarray:
    """Row index of every decision of an instance."""
    return FIRST_ROW + STRIDE * np.arange(n_decisions(rows), dtype=np.int64)


def eligibility_from_any_process(any_proc: np.ndarray) -> np.ndarray:
    """Eligibility at each decision row.

    ``any_proc[row]`` is true when at least one process channel is
    finite at that row.
    """
    r = decision_rows(len(any_proc))
    cs = np.r_[0, np.cumsum(any_proc.astype(np.int64))]
    window = cs[r + 1] - cs[r + 1 - ELIG_WINDOW]
    return window >= ELIG_FRAC * ELIG_WINDOW


def make_decision_table(
    index_files: Sequence[Mapping[str, Any]],
    elig_by_path: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, ...]:
    """Build the decision table of a list of instances.

    ``index_files`` are label-index records sorted by path. Returns the
    arrays (instance, row, event class, eligible, operational state).
    """
    inst, r, y, el, st = [], [], [], [], []
    for i, f in enumerate(index_files):
        rr = decision_rows(f["rows"])
        labels = {}
        for name, key in (("y", "class_runs"), ("st", "state_runs")):
            runs = np.array(f[key], dtype=np.int64)
            pos = np.searchsorted(runs[:, 1], rr, side="right") - 1
            labels[name] = runs[:, 0][pos]
        y.append(event_class(labels["y"]).astype(np.int8))
        st.append(labels["st"].astype(np.int8))
        e = np.asarray(elig_by_path[f["path"]], dtype=bool)
        if len(e) != len(rr):
            raise ValueError(f"{f['path']}: {len(e)} flags, {len(rr)} rows")
        inst.append(np.full(len(rr), i, dtype=np.int32))
        r.append(rr.astype(np.int32))
        el.append(e)
    return (
        np.concatenate(inst),
        np.concatenate(r),
        np.concatenate(y),
        np.concatenate(el),
        np.concatenate(st),
    )


def array_digest(a: np.ndarray) -> str:
    """SHA-256 of the C-contiguous bytes of an array."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
