"""Causal window features and the training-free reference detector.

For a decision at row ``r`` only rows ``r - 3599 .. r`` of the same
instance are used, which is the lookback cap of the protocol.
Everything is computed from prefix sums (linear in the number of rows)
except the window minimum and maximum, which use strided windows at
decision rows only.
"""

from __future__ import annotations

import numpy as np

from ..constants import FIRST_ROW, PROCESS_CHANNELS, STATE_CHANNELS, STRIDE
from ..grid import decision_rows

CONT = PROCESS_CHANNELS
STATE = STATE_CHANNELS
W2, W3, W4 = 300, 900, 3600
EPS_REL = 1e-9

PER_CHANNEL = [
    "last",
    "mean_w2",
    "mean_w3",
    "mean_w4",
    "std_w2",
    "std_w3",
    "std_w4",
    "min_w3",
    "max_w3",
    "range_rel_w3",
    "drift_w3",
    "z_last_w4",
    "z_w2_w4",
    "cv_w3",
    "rough_w3",
    "extrema_rate_w3",
    "finite_w3",
]
PER_STATE = ["last", "changes_w3", "changes_w4"]


def feature_names() -> list[str]:
    """Names of the 333 features (17 per channel, 3 per valve)."""
    names = [f"{c}|{f}" for c in CONT for f in PER_CHANNEL]
    return names + [f"{c}|{f}" for c in STATE for f in PER_STATE]


def _prefix(a: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(a, dtype=np.float64)]


def _window_sums(cs: np.ndarray, r: np.ndarray, width: int) -> np.ndarray:
    return cs[r + 1] - cs[r - width + 1]


def channel_features(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Features of one process channel at the decision rows ``r``.

    ``x`` is the channel (NaN where missing). Returns an array of shape
    (decisions, len(PER_CHANNEL)) in float32.
    """
    n = len(r)
    out = np.full((n, len(PER_CHANNEL)), np.nan, dtype=np.float64)
    fin = np.isfinite(x)
    ix = {k: i for i, k in enumerate(PER_CHANNEL)}
    if not fin.any():
        out[:, ix["finite_w3"]] = 0.0
        return out.astype(np.float32)
    # numerical conditioning only; added back wherever a level is output
    shift = float(np.median(x[fin][:7200]))
    xs = np.where(fin, x - shift, 0.0)
    c0, c1, c2 = _prefix(fin.astype(np.float64)), _prefix(xs), _prefix(xs * xs)

    def stats(width):
        cnt = _window_sums(c0, r, width)
        s1 = _window_sums(c1, r, width)
        s2 = _window_sums(c2, r, width)
        with np.errstate(all="ignore"):
            m = np.where(cnt > 0, s1 / cnt, np.nan)
            v = np.where(cnt > 1, np.maximum(s2 / cnt - m * m, 0.0), np.nan)
        return cnt, m, np.sqrt(v)

    _, m2, s2 = stats(W2)
    cnt3, m3, s3 = stats(W3)
    _, m4, s4 = stats(W4)
    last = np.where(fin[r], x[r], np.nan)
    out[:, ix["last"]] = last
    out[:, ix["mean_w2"]] = m2 + shift
    out[:, ix["mean_w3"]] = m3 + shift
    out[:, ix["mean_w4"]] = m4 + shift
    out[:, ix["std_w2"]] = s2
    out[:, ix["std_w3"]] = s3
    out[:, ix["std_w4"]] = s4
    # min and max over W3, with strided windows at decision rows only
    xw = np.where(fin, x, np.nan)
    start = FIRST_ROW - W3 + 1
    win = np.lib.stride_tricks.sliding_window_view(xw, W3)[start::STRIDE][:n]
    with np.errstate(all="ignore"):
        mn = np.fmin.reduce(win, axis=1)
        mx = np.fmax.reduce(win, axis=1)
    out[:, ix["min_w3"]], out[:, ix["max_w3"]] = mn, mx
    with np.errstate(all="ignore"):
        out[:, ix["range_rel_w3"]] = (mx - mn) / (np.abs(m3 + shift) + EPS_REL + s4)
        # drift: last quarter of W3 minus its first quarter, in units of
        # the standard deviation over W4
        q = W3 // 4
        n_first = _window_sums(c0, r - W3 + q, q)
        n_last = _window_sums(c0, r, q)
        first = _window_sums(c1, r - W3 + q, q) / np.maximum(n_first, 1)
        lastq = _window_sums(c1, r, q) / np.maximum(n_last, 1)
        ok_q = (n_first > 0) & (n_last > 0)
        scale = s4 + EPS_REL * (1 + np.abs(m4 + shift))
        out[:, ix["drift_w3"]] = np.where(ok_q, (lastq - first) / scale, np.nan)
        out[:, ix["z_last_w4"]] = (last - (m4 + shift)) / scale
        out[:, ix["z_w2_w4"]] = (m2 - m4) / scale
        out[:, ix["cv_w3"]] = s3 / (np.abs(m3 + shift) + EPS_REL + s4)
    # roughness and rate of local extrema over W3
    dx = np.diff(xw)
    dfin = np.isfinite(dx)
    ad = np.where(dfin, np.abs(dx), 0.0)
    cd0, cd1 = _prefix(dfin.astype(np.float64)), _prefix(ad)
    rr = r - 1  # dx[i] = x[i+1] - x[i]; the window rows lo..r use dx[lo..r-1]
    cntd = cd0[rr + 1] - cd0[r - W3 + 1]
    with np.errstate(all="ignore"):
        mean_abs = np.where(cntd > 0, (cd1[rr + 1] - cd1[r - W3 + 1]) / cntd, np.nan)
        out[:, ix["rough_w3"]] = mean_abs / scale
    ext = np.zeros(len(dx), dtype=np.float64)
    prod = dx[1:] * dx[:-1]
    ext[1:] = np.where(np.isfinite(prod) & (prod < 0), 1.0, 0.0)
    ce = _prefix(ext)
    with np.errstate(all="ignore"):
        rate = (ce[rr + 1] - ce[r - W3 + 2]) / cntd
        out[:, ix["extrema_rate_w3"]] = np.where(cntd > 2, rate, np.nan)
    out[:, ix["finite_w3"]] = cnt3 / W3
    return out.astype(np.float32)


def state_features(e: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Features of one valve-state channel at the decision rows."""
    n = len(r)
    out = np.full((n, len(PER_STATE)), np.nan, dtype=np.float64)
    fin = np.isfinite(e)
    if not fin.any():
        return out.astype(np.float32)
    out[:, 0] = np.where(fin[r], e[r], np.nan)
    d = np.diff(e)
    changed = np.where(np.isfinite(d) & (d != 0), 1.0, 0.0)
    cc = _prefix(changed)
    rr = r - 1
    out[:, 1] = cc[rr + 1] - cc[r - W3 + 1]
    out[:, 2] = cc[rr + 1] - cc[r - W4 + 1]
    return out.astype(np.float32)


def instance_features(cont: np.ndarray, state: np.ndarray) -> np.ndarray:
    """All features of an instance at its decision rows.

    ``cont`` (rows x 18) holds the process channels and ``state``
    (rows x 9) the valve states, NaN where absent. Returns an array of
    shape (decisions, 333) in float32.
    """
    r = decision_rows(cont.shape[0])
    blocks = [
        channel_features(cont[:, j].astype(np.float64), r) for j in range(len(CONT))
    ]
    blocks += [
        state_features(state[:, j].astype(np.float64), r) for j in range(len(STATE))
    ]
    return np.concatenate(blocks, axis=1)


# --------------------------------------------- reference detector
SIMPLE_CHANNELS = CONT[:16]  # pressures, flows, temperatures
SHORT, REF = 300, 3300  # windows [r-299, r] and [r-3599, r-300]


def _past_stats(prefix, r, lo_off, hi_off):
    """Count, mean and standard deviation over a trailing window.

    The window holds rows r - lo_off + 1 .. r - hi_off.
    """
    c0, c1, c2 = prefix
    cnt = c0[r - hi_off + 1] - c0[r - lo_off + 1]
    s1 = c1[r - hi_off + 1] - c1[r - lo_off + 1]
    s2 = c2[r - hi_off + 1] - c2[r - lo_off + 1]
    with np.errstate(all="ignore"):
        m = np.where(cnt > 0, s1 / cnt, np.nan)
        var = np.where(cnt > 1, np.maximum(s2 / cnt - m * m, 0.0), np.nan)
    return cnt, m, np.sqrt(var)


def simple_detector_score(cont: np.ndarray) -> np.ndarray:
    """Windowed z-test against the instance's own recent past.

    Maximum over channels of ``max(level_z, vol_z)`` where
    ``level_z = |mean_short - mean_ref| / (std_ref + 1e-3 |mean_ref|)``
    and ``vol_z = 2 max(0, std_short / (std_ref + eps) - 1)``. A channel
    is usable at a decision when 80 % of both windows are finite. No
    training and no labels. Returns the raw score (NaN where no channel
    is usable).
    """
    r = decision_rows(cont.shape[0])
    best = np.full(len(r), -np.inf)
    usable = np.zeros(len(r), dtype=bool)
    for j in range(len(SIMPLE_CHANNELS)):
        x = cont[:, j].astype(np.float64)
        fin = np.isfinite(x)
        if not fin.any():
            continue
        shift = float(np.median(x[fin][:7200]))
        xs = np.where(fin, x - shift, 0.0)
        c0, c1, c2 = _prefix(fin.astype(np.float64)), _prefix(xs), _prefix(xs * xs)

        prefix = (c0, c1, c2)
        cnt_s, m_s, sd_s = _past_stats(prefix, r, SHORT, 0)
        cnt_r, m_r, sd_r = _past_stats(prefix, r, SHORT + REF, SHORT)
        ok = (cnt_s >= 0.8 * SHORT) & (cnt_r >= 0.8 * REF)
        with np.errstate(all="ignore"):
            mean_ref = m_r + shift
            d = sd_r + 1e-3 * np.abs(mean_ref) + 1e-12
            level = np.abs(m_s - m_r) / d
            vol = 2.0 * np.maximum(0.0, sd_s / d - 1.0)
            z = np.where(ok, np.maximum(level, vol), -np.inf)
        best = np.maximum(best, z)
        usable |= ok
    return np.where(usable & np.isfinite(best), best, np.nan)


def simple_score_to_unit(z: np.ndarray) -> np.ndarray:
    """Monotone map of the raw score to [0, 1)."""
    z = np.asarray(z, dtype=np.float64)
    return np.where(np.isfinite(z), z / (z + 3.0), np.nan)
