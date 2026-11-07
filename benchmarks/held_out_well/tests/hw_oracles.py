"""Slow or independent implementations that serve as test oracles."""

import numpy as np
from held_out_well.constants import N_CLASSES, OFF_K, ON_K
from held_out_well.detection import part


def alarm_state_reference(inst, elig, score, theta):
    """Plain loop over decisions: the alarm state of the protocol."""
    out = np.zeros(len(score), dtype=bool)
    state = False
    on = off = 0
    for i in range(len(score)):
        if i == 0 or inst[i] != inst[i - 1]:
            state = False
            on = off = 0
        if not elig[i] or not np.isfinite(score[i]):
            state = False
            on = off = 0
            out[i] = False
            continue
        above = theta is not None and score[i] >= theta
        on = on + 1 if above else 0
        off = 0 if above else off + 1
        if on >= ON_K:
            state = True
        if off >= OFF_K:
            state = False
        out[i] = state
    return out


def metrics_numpy(S, classes=None):
    """Closed-form classification, AUROC and AP from summed statistics.

    The formulas of the first implementation of the scorer, written
    with numpy only. The toolkit-based scorer must agree with them.
    """
    S = np.asarray(S, dtype=np.float64)
    out = {}
    conf = part(S, "conf").reshape(S.shape[:-1] + (N_CLASSES, N_CLASSES))
    tp = np.einsum("...ii->...i", conf)
    rowsum, colsum = conf.sum(-1), conf.sum(-2)
    den = 2 * tp + (colsum - tp) + (rowsum - tp)
    with np.errstate(all="ignore"):
        f1 = np.where(den > 0, 2 * tp / den, 0.0)
        rec = np.where(rowsum > 0, tp / rowsum, 0.0)
    present = rowsum > 0
    if classes is not None:
        present = present & np.isin(np.arange(N_CLASSES), classes)
    npres = present.sum(-1)
    safe = np.maximum(npres, 1)
    out["macro_f1"] = np.where(npres > 0, (f1 * present).sum(-1) / safe, np.nan)
    events = present.copy()
    events[..., 0] = False
    nev = events.sum(-1)
    out["macro_f1_event_classes"] = np.where(
        nev > 0, (f1 * events).sum(-1) / np.maximum(nev, 1), np.nan
    )
    out["balanced_accuracy"] = np.where(
        npres > 0, (rec * present).sum(-1) / safe, np.nan
    )
    total = conf.sum((-1, -2))
    out["accuracy"] = np.where(total > 0, tp.sum(-1) / np.maximum(total, 1), np.nan)
    for c in range(N_CLASSES):
        out[f"f1_c{c}"] = np.where(rowsum[..., c] > 0, f1[..., c], np.nan)
    hp, hn = part(S, "hist_pos"), part(S, "hist_neg")
    P, N = hp.sum(-1), hn.sum(-1)
    with np.errstate(all="ignore"):
        below = np.cumsum(hn, -1) - hn
        auc_num = (hp * (below + 0.5 * hn)).sum(-1)
        out["auroc"] = np.where((P > 0) & (N > 0), auc_num / (P * N), np.nan)
        rp = np.cumsum(hp[..., ::-1], -1)
        rn = np.cumsum(hn[..., ::-1], -1)
        prec = np.where(rp + rn > 0, rp / (rp + rn), 1.0)
        dr = hp[..., ::-1] / np.maximum(P[..., None], 1)
        out["average_precision"] = np.where(P > 0, (prec * dr).sum(-1), np.nan)
    return out


def window(x, rr, width):
    """Rows rr - width + 1 .. rr: finite values and the raw window."""
    w = x[rr - width + 1 : rr + 1]
    return w[np.isfinite(w)], w


def reference_channel_features(x, r):
    """Straightforward slice implementation of the channel features."""
    from held_out_well.baselines.features import (
        EPS_REL,
        PER_CHANNEL,
        W2,
        W3,
        W4,
    )

    out = np.full((len(r), len(PER_CHANNEL)), np.nan)
    ix = {k: i for i, k in enumerate(PER_CHANNEL)}
    for k, rr in enumerate(r):
        a2, _ = window(x, rr, W2)
        a3, w3 = window(x, rr, W3)
        a4, _ = window(x, rr, W4)
        out[k, ix["last"]] = x[rr] if np.isfinite(x[rr]) else np.nan
        for name, a in (("w2", a2), ("w3", a3), ("w4", a4)):
            out[k, ix["mean_" + name]] = a.mean() if len(a) else np.nan
            out[k, ix["std_" + name]] = a.std() if len(a) > 1 else np.nan
        out[k, ix["min_w3"]] = a3.min() if len(a3) else np.nan
        out[k, ix["max_w3"]] = a3.max() if len(a3) else np.nan
        out[k, ix["finite_w3"]] = len(a3) / W3
        s4 = a4.std() if len(a4) > 1 else np.nan
        m4 = a4.mean() if len(a4) else np.nan
        m3 = a3.mean() if len(a3) else np.nan
        m2 = a2.mean() if len(a2) else np.nan
        span = a3.max() - a3.min() if len(a3) else np.nan
        out[k, ix["range_rel_w3"]] = (
            span / (abs(m3) + EPS_REL + s4) if len(a3) else np.nan
        )
        denom = s4 + EPS_REL * (1 + abs(m4))
        out[k, ix["z_last_w4"]] = (out[k, ix["last"]] - m4) / denom
        out[k, ix["z_w2_w4"]] = (m2 - m4) / denom
        if len(a3) > 1:
            out[k, ix["cv_w3"]] = a3.std() / (abs(m3) + EPS_REL + s4)
        q = W3 // 4
        f1 = w3[:q][np.isfinite(w3[:q])]
        f2 = w3[-q:][np.isfinite(w3[-q:])]
        if len(f1) and len(f2):
            out[k, ix["drift_w3"]] = (f2.mean() - f1.mean()) / denom
        d = np.diff(w3)
        dd = np.abs(d)[np.isfinite(d)]
        out[k, ix["rough_w3"]] = (dd.mean() / denom) if len(dd) else np.nan
        prod = d[1:] * d[:-1]
        cnt = np.isfinite(d).sum()
        extrema = (np.isfinite(prod) & (prod < 0)).sum()
        out[k, ix["extrema_rate_w3"]] = extrema / cnt if cnt > 2 else np.nan
    return out
