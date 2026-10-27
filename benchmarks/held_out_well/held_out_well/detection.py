"""Event detection: alarms, episodes, onsets and per-well statistics.

The 3W Toolkit has no time-related metrics yet (``BACKLOG.md`` lists
them as wanted), so this module is specific to the benchmark. Everything
that depends on the evaluated wells is reduced to one vector of
sufficient statistics per well. Metrics of any set of wells, and of any
bootstrap resample of wells, are then computed from sums of these
vectors.

Alarm rule. An alarm switches on after ``ON_K`` consecutive eligible
decisions with a score at or above the threshold, and off after
``OFF_K`` consecutive eligible decisions below it. Ineligible decisions,
missing scores and the start of an instance reset the state.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .benchmark import Benchmark
from .constants import (
    FIRST_ROW,
    LOGIT_CLIP,
    N_CLASSES,
    NBINS,
    ONSET_TOL,
    STRIDE,
)

# ----------------------------------------------------------- layout
_ITEMS = [
    ("conf", N_CLASSES * N_CLASSES),
    ("hist_pos", NBINS),
    ("hist_neg", NBINS),
    ("ep_total", N_CLASSES),
    ("ep_obs", N_CLASSES),
    ("ep_cov", N_CLASSES),
    ("ep_fresh", N_CLASSES),
    ("on_obs", N_CLASSES),
    ("on_det", N_CLASSES),
    ("scalars", 8),
]
LAYOUT: dict[str, tuple[int, int]] = {}
_offset = 0
for _key, _size in _ITEMS:
    LAYOUT[_key] = (_offset, _offset + _size)
    _offset += _size
DIM = _offset
SCALARS = [
    "norm_dec",
    "norm_alarm",
    "false_alarms",
    "norm_total",
    "norm_elig",
    "event_total",
    "event_elig",
    "cls_dec",
]


def part(S: np.ndarray, key: str) -> np.ndarray:
    """The block ``key`` of a (batch of) statistics vector(s)."""
    a, b = LAYOUT[key]
    return S[..., a:b]


# ----------------------------------------------------------- alarms
def alarm_state(
    bench: Benchmark, score: np.ndarray, theta: float | None
) -> tuple[np.ndarray, np.ndarray]:
    """Alarm state and rising edges at every decision (vectorised).

    With two decisions needed to switch on and two to switch off, the
    state can be computed without a loop: a decision is on if it and
    the previous decision of the same instance are both above the
    threshold, and otherwise keeps the previous state unless it is
    forced off.
    """
    n = bench.n
    ok = bench.elig & np.isfinite(score)
    if theta is None:
        above = np.zeros(n, dtype=bool)
    else:
        above = ok & (score >= theta)
    same_prev = np.r_[False, bench.inst[1:] == bench.inst[:-1]]
    prev_above = np.r_[False, above[:-1]] & same_prev
    set_on = above & prev_above
    force_off = (~ok) | (~same_prev) | (ok & ~above & ~prev_above)
    val = np.where(set_on, 1, np.where(force_off, 0, -1))
    idx = np.maximum.accumulate(np.where(val >= 0, np.arange(n), 0))
    state = val[idx] == 1
    prev_state = np.r_[False, state[:-1]] & same_prev
    rise = state & ~prev_state
    return state, rise


# ------------------------------------------------- score quantisation
def logit_bin(score: np.ndarray) -> np.ndarray:
    """Index of the logit bin of each score in [0, 1]."""
    s = np.clip(score, 1e-12, 1 - 1e-12)
    z = np.log(s / (1 - s))
    z = np.clip(z, -LOGIT_CLIP, LOGIT_CLIP - 1e-9)
    return ((z + LOGIT_CLIP) / (2 * LOGIT_CLIP) * NBINS).astype(np.int64)


# ------------------------------------------------------- validation
def validate(
    bench: Benchmark,
    score: Any,
    pred: Any,
    wells: Sequence[str],
    need_pred: bool = False,
) -> np.ndarray:
    """Check a prediction against the decision table.

    Returns the mask of decisions that belong to the evaluated wells.
    """
    score = np.asarray(score, dtype=np.float64)
    if score.shape != (bench.n,):
        raise ValueError(f"score must have shape ({bench.n},)")
    sel = bench.well_mask(wells)[bench.well_of_decision]
    need = sel & bench.elig
    if not np.all(np.isfinite(score[need])):
        raise ValueError("non-finite score on eligible decisions")
    if np.any((score[need] < 0) | (score[need] > 1)):
        raise ValueError("scores must lie in [0, 1]")
    if need_pred:
        p = np.asarray(pred)
        if p.shape != (bench.n,) or np.any((p[need] < 0) | (p[need] >= N_CLASSES)):
            raise ValueError("pred must give a class 0..9 on every decision")
    return sel


# ---------------------------------------------- per-well statistics
def well_statistics(
    bench: Benchmark,
    score: Any,
    pred: Any,
    wells: Sequence[str],
    theta: float | None,
) -> tuple[np.ndarray, list[str], list[list[int]]]:
    """Sufficient statistics of every evaluated well.

    Returns the matrix (wells x DIM), the well names and, per well, the
    detection delays in seconds of the detected onsets.
    """
    sel = validate(bench, score, pred, wells, need_pred=pred is not None)
    score = np.where(sel, np.asarray(score, dtype=np.float64), np.nan)
    nw = len(bench.wells)
    w = bench.well_of_decision
    y = bench.y.astype(np.int64)
    lab = (y >= 0) & sel
    labe = lab & bench.elig
    state, rise = alarm_state(bench, score, theta)
    S = np.zeros((nw, DIM))
    sc = S[:, slice(*LAYOUT["scalars"])]
    # window classification: confusion matrix per well
    if pred is not None:
        p = np.asarray(pred).astype(np.int64)
        m = labe & (p >= 0) & (p < N_CLASSES)
        cell = w[m] * N_CLASSES**2 + y[m] * N_CLASSES + p[m]
        conf = np.bincount(cell, minlength=nw * N_CLASSES**2)
        S[:, slice(*LAYOUT["conf"])] = conf.reshape(nw, N_CLASSES**2)
        sc[:, SCALARS.index("cls_dec")] = np.bincount(w[m], minlength=nw)
    # event score histograms: event class above 0 against normal
    pos = labe & (y > 0)
    neg = labe & (y == 0)
    b = logit_bin(np.where(np.isfinite(score), score, 0.5))
    for key, mask in (("hist_pos", pos), ("hist_neg", neg)):
        h = np.bincount(w[mask] * NBINS + b[mask], minlength=nw * NBINS)
        S[:, slice(*LAYOUT[key])] = h.reshape(nw, NBINS)
    counts = {
        "norm_dec": neg,
        "norm_alarm": neg & state,
        "false_alarms": neg & rise,
        "norm_total": lab & (y == 0),
        "norm_elig": neg,
        "event_total": lab & (y > 0),
        "event_elig": pos,
    }
    for name, mask in counts.items():
        sc[:, SCALARS.index(name)] = np.bincount(w[mask], minlength=nw)
    # episodes, via prefix sums over the decision table
    cs_e = np.r_[0, np.cumsum(bench.elig & sel)]
    cs_a = np.r_[0, np.cumsum(state)]
    cs_r = np.r_[0, np.cumsum(rise)]
    off = bench.inst_offset[bench.ep_inst]
    ndec = bench.inst_offset[bench.ep_inst + 1] - off
    lo = np.maximum(0, -(-(bench.ep_start - FIRST_ROW) // STRIDE))
    hi = np.minimum(ndec - 1, -(-(bench.ep_end - FIRST_ROW) // STRIDE) - 1)
    hi_on = np.minimum(ndec - 1, (bench.ep_start + ONSET_TOL - FIRST_ROW) // STRIDE)
    wanted = bench.well_mask(wells)
    ok_ep = wanted[bench.ep_well] & (hi >= lo)
    gl, gh, gho = off + lo, off + hi, off + hi_on
    n_el = cs_e[gh + 1] - cs_e[gl]
    n_al = cs_a[gh + 1] - cs_a[gl]
    n_ri = cs_r[gh + 1] - cs_r[gl]
    obs = ok_ep & (n_el >= 1)
    cov = obs & (n_al >= 1)
    fresh = cov & (n_ri >= 1)
    n_el_on = cs_e[gho + 1] - cs_e[gl]
    n_ri_on = cs_r[gho + 1] - cs_r[gl]
    onset_obs = ok_ep & (bench.ep_known == 1) & (hi_on >= lo) & (n_el_on >= 2)
    onset_det = onset_obs & (n_ri_on >= 1)
    ev, ew = bench.ep_event, bench.ep_well
    for key, mask in (
        ("ep_total", ok_ep),
        ("ep_obs", obs),
        ("ep_cov", cov),
        ("ep_fresh", fresh),
        ("on_obs", onset_obs),
        ("on_det", onset_det),
    ):
        h = np.bincount(ew[mask] * N_CLASSES + ev[mask], minlength=nw * N_CLASSES)
        S[:, slice(*LAYOUT[key])] = h.reshape(nw, N_CLASSES)
    # delay of every detected onset: first rising edge after its start
    delays: list[list[int]] = [[] for _ in range(nw)]
    rise_idx = np.flatnonzero(rise)
    for k in np.flatnonzero(onset_det):
        first = np.searchsorted(rise_idx, gl[k])
        d = bench.r[rise_idx[first]] - bench.ep_start[k]
        delays[ew[k]].append(int(d))
    keep = [i for i, name in enumerate(bench.wells) if wanted[i]]
    return S[keep], [bench.wells[i] for i in keep], [delays[i] for i in keep]


# --------------------------------------------- metrics from the sums
def detection_metrics(S: Any) -> dict[str, np.ndarray]:
    """Alarm, episode and eligibility metrics from summed statistics.

    ``S`` has shape (..., DIM); every output has the leading shape.
    Class counts that the toolkit metrics need are not touched here.
    """
    S = np.asarray(S, dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    ev = slice(1, N_CLASSES)
    ep_total = part(S, "ep_total")[..., ev].sum(-1)
    ep_obs = part(S, "ep_obs")[..., ev].sum(-1)
    ep_cov = part(S, "ep_cov")[..., ev].sum(-1)
    ep_fresh = part(S, "ep_fresh")[..., ev].sum(-1)
    on_obs = part(S, "on_obs")[..., ev].sum(-1)
    on_det = part(S, "on_det")[..., ev].sum(-1)
    sc = part(S, "scalars")

    def g(name: str) -> np.ndarray:
        return sc[..., SCALARS.index(name)]

    with np.errstate(all="ignore"):
        out["episode_coverage"] = np.where(ep_obs > 0, ep_cov / ep_obs, np.nan)
        out["carried_in_share_of_covered"] = np.where(
            ep_cov > 0, (ep_cov - ep_fresh) / ep_cov, np.nan
        )
        out["onset_detection_300s"] = np.where(on_obs > 0, on_det / on_obs, np.nan)
        days = g("norm_dec") * STRIDE / 86400.0
        out["false_alarms_per_normal_day"] = np.where(
            days > 0, g("false_alarms") / days, np.nan
        )
        out["normal_time_occupation"] = np.where(
            g("norm_dec") > 0, g("norm_alarm") / g("norm_dec"), np.nan
        )
        out["normal_eligibility"] = np.where(
            g("norm_total") > 0, g("norm_elig") / g("norm_total"), np.nan
        )
        out["event_eligibility"] = np.where(
            g("event_total") > 0, g("event_elig") / g("event_total"), np.nan
        )
        out["episodes_total"] = ep_total
        out["episodes_unobservable"] = ep_total - ep_obs
        out["normal_days"] = days
        for c in range(1, N_CLASSES):
            o = part(S, "ep_obs")[..., c]
            out[f"coverage_c{c}"] = np.where(
                o > 0, part(S, "ep_cov")[..., c] / o, np.nan
            )
        out["event_decisions"] = part(S, "hist_pos").sum(-1)
        out["normal_decisions"] = part(S, "hist_neg").sum(-1)
    return out
