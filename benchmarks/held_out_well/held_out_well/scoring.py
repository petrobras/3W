"""Scoring of a prediction on the decision grid.

Classification metrics (macro-F1, balanced accuracy, accuracy, per-class
F1) and the threshold-free event metrics (AUROC, average precision) are
computed by ``ThreeWToolkit.metrics``. The toolkit functions accept
``sample_weight``, so one call on the (truth, prediction) cells of a
summed confusion matrix, or on the score bins of summed histograms,
gives the metric of the pooled decisions of any set of wells without
holding the decisions. That is what makes a bootstrap over wells with
2,000 resamples cheap. Alarm, episode and onset metrics live in
``detection``: the toolkit has none.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from ThreeWToolkit.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from .benchmark import Benchmark
from .constants import BOOT_B, BOOT_SEED, N_CLASSES
from .detection import detection_metrics, part, well_statistics

NAN = float("nan")


# ---------------------------------------------- toolkit metrics
def classification_metrics(
    conf: Any, classes: Sequence[int] | None = None
) -> dict[str, Any]:
    """Classification metrics of a confusion matrix.

    ``conf[t, p]`` counts decisions of class ``t`` predicted as ``p``.
    Macro averages run over the classes that occur in the truth (and, if
    ``classes`` is given, belong to it). Returns macro-F1 over those
    classes, macro-F1 over the event classes among them (normal
    excluded), balanced accuracy, accuracy over all decisions, and the
    per-class F1 (NaN for a class absent from the truth).
    """
    conf = np.asarray(conf, dtype=np.float64).reshape(N_CLASSES, N_CLASSES)
    truth = conf.sum(1)
    present = [
        c
        for c in range(N_CLASSES)
        if truth[c] > 0 and (classes is None or c in classes)
    ]
    out: dict[str, Any] = {
        "macro_f1": NAN,
        "macro_f1_event_classes": NAN,
        "balanced_accuracy": NAN,
        "accuracy": NAN,
        "f1": np.full(N_CLASSES, NAN),
    }
    t, p = np.nonzero(conf)
    if len(t) == 0:
        return out
    w = conf[t, p]
    out["accuracy"] = float(accuracy_score(t, p, sample_weight=w))
    per_class = f1_score(
        t,
        p,
        labels=list(range(N_CLASSES)),
        average=None,
        sample_weight=w,
        zero_division=0,
    )
    out["f1"] = np.where(truth > 0, np.asarray(per_class), NAN)
    if not present:
        return out
    out["macro_f1"] = float(
        f1_score(
            t,
            p,
            labels=present,
            average="macro",
            sample_weight=w,
            zero_division=0,
        )
    )
    events = [c for c in present if c > 0]
    if events:
        out["macro_f1_event_classes"] = float(
            f1_score(
                t,
                p,
                labels=events,
                average="macro",
                sample_weight=w,
                zero_division=0,
            )
        )
    keep = np.isin(t, present)
    with warnings.catch_warnings():
        # predictions of a class that never occurs are dropped by design
        warnings.simplefilter("ignore", UserWarning)
        out["balanced_accuracy"] = float(
            balanced_accuracy_score(t[keep], p[keep], sample_weight=w[keep])
        )
    return out


def threshold_free_metrics(hist_pos: Any, hist_neg: Any) -> tuple[float, float]:
    """AUROC and average precision of the event score.

    ``hist_pos`` and ``hist_neg`` count event and normal decisions per
    score bin. The toolkit functions run on the non-empty bins, with
    the counts as weights; ties inside a bin count as ties.
    """
    hp = np.asarray(hist_pos, dtype=np.float64)
    hn = np.asarray(hist_neg, dtype=np.float64)
    if hp.sum() <= 0:
        return NAN, NAN
    ip, ineg = np.flatnonzero(hp), np.flatnonzero(hn)
    y_true = np.r_[np.ones(len(ip), dtype=int), np.zeros(len(ineg), dtype=int)]
    y_score = np.r_[ip, ineg].astype(np.float64)
    w = np.r_[hp[ip], hn[ineg]]
    ap = float(average_precision_score(y_true, y_score, sample_weight=w))
    if hn.sum() <= 0:
        return NAN, ap
    auc = float(roc_auc_score(y_true, y_score, sample_weight=w))
    return auc, ap


def metrics_from_sums(
    S: Any, classes: Sequence[int] | None = None
) -> dict[str, np.ndarray]:
    """All metrics from summed per-well statistics.

    ``S`` has shape (..., DIM): one vector, a batch of vectors (for
    example bootstrap resamples) or one vector per well. Every returned
    array has the leading shape of ``S``.
    """
    S = np.asarray(S, dtype=np.float64)
    lead = S.shape[:-1]
    flat = S.reshape(-1, S.shape[-1])
    out = detection_metrics(flat)
    cls = [classification_metrics(part(r, "conf"), classes) for r in flat]
    tf = [
        threshold_free_metrics(part(r, "hist_pos"), part(r, "hist_neg")) for r in flat
    ]
    for key in (
        "macro_f1",
        "macro_f1_event_classes",
        "balanced_accuracy",
        "accuracy",
    ):
        out[key] = np.array([c[key] for c in cls], dtype=np.float64)
    f1 = np.array([c["f1"] for c in cls], dtype=np.float64)
    for c in range(N_CLASSES):
        out[f"f1_c{c}"] = f1[:, c]
    out["auroc"] = np.array([a for a, _ in tf], dtype=np.float64)
    out["average_precision"] = np.array([b for _, b in tf], dtype=np.float64)
    return {k: np.asarray(v).reshape(lead) for k, v in out.items()}


# ----------------------------------------------- bootstrap
def resample_counts(n_wells: int, B: int = BOOT_B, seed: int = BOOT_SEED) -> np.ndarray:
    """How often each well is drawn in each of ``B`` resamples."""
    rng = np.random.default_rng(seed)
    draws = [
        np.bincount(rng.integers(0, n_wells, n_wells), minlength=n_wells)
        for _ in range(B)
    ]
    return np.stack(draws).astype(np.float64)


def bootstrap(
    S: np.ndarray,
    classes: Sequence[int] | None = None,
    B: int = BOOT_B,
    seed: int = BOOT_SEED,
    S2: np.ndarray | None = None,
    chunk: int = 250,
) -> dict[str, list[float]]:
    """Percentile intervals (95 %) from resampling whole wells.

    ``S`` has one row of statistics per well. With ``S2`` (the same
    wells, another prediction) the interval is that of the paired
    difference ``S`` minus ``S2``, on the same resamples. Each entry is
    [lower, upper, number of resamples where the metric was defined].
    """
    counts = resample_counts(S.shape[0], B, seed)
    acc: dict[str, list[np.ndarray]] = {}
    for a in range(0, B, chunk):
        c = counts[a : a + chunk]
        m1 = metrics_from_sums(c @ S, classes)
        if S2 is not None:
            m2 = metrics_from_sums(c @ S2, classes)
            m1 = {k: m1[k] - m2[k] for k in m1}
        for k, v in m1.items():
            acc.setdefault(k, []).append(v)
    out: dict[str, list[float]] = {}
    for k, chunks in acc.items():
        v = np.concatenate(chunks)
        if np.isfinite(v).any():
            out[k] = [
                float(np.nanpercentile(v, 2.5)),
                float(np.nanpercentile(v, 97.5)),
                int(np.isfinite(v).sum()),
            ]
    return out


# ------------------------------------------------------ report
def report(
    bench: Benchmark,
    score: Any,
    pred: Any,
    wells: Sequence[str],
    theta: float | None,
    classes: Sequence[int] | None = None,
    boot: bool = True,
    B: int = BOOT_B,
    seed: int = BOOT_SEED,
) -> dict[str, Any]:
    """Point estimates and bootstrap intervals for a set of wells.

    ``score`` is the event score in [0, 1] of every decision and
    ``pred`` the class (0..9) or ``None`` for a pure detector. Metrics
    pool the eligible decisions of the wells. ``theta`` is the alarm
    threshold chosen on development predictions (``select_threshold``).
    """
    S, names, delays = well_statistics(bench, score, pred, wells, theta)
    point = {k: float(v) for k, v in metrics_from_sums(S.sum(0), classes).items()}
    flat = [d for ds in delays for d in ds]
    point["onset_delay_median_s"] = float(np.median(flat)) if flat else NAN
    if pred is not None:
        per_well = metrics_from_sums(S, classes)["macro_f1"]
        ok = np.isfinite(per_well)
        point["per_well_macro_f1_mean"] = (
            float(np.mean(per_well[ok])) if ok.any() else NAN
        )
        point["per_well_macro_f1_sd"] = (
            float(np.std(per_well[ok], ddof=1)) if ok.sum() > 1 else NAN
        )
    point.update(exact_event_metrics(bench, score, wells))
    res: dict[str, Any] = {
        "n_wells": len(names),
        "wells": names,
        "theta": theta,
        "point": point,
        "confusion": part(S.sum(0), "conf")
        .reshape(N_CLASSES, N_CLASSES)
        .astype(int)
        .tolist(),
    }
    if pred is not None:
        res["per_well_macro_f1"] = dict(zip(names, per_well.tolist()))
    if boot:
        res["ci95"] = bootstrap(S, classes, B, seed)
    return res


def exact_event_metrics(
    bench: Benchmark, score: Any, wells: Sequence[str]
) -> dict[str, float]:
    """AUROC and average precision on the unquantised scores.

    The same toolkit functions as ``threshold_free_metrics``, applied
    to the pooled decisions themselves. They differ from the binned
    values only through ties inside a score bin.
    """
    score = np.asarray(score, dtype=np.float64)
    sel = bench.well_mask(wells)[bench.well_of_decision]
    sel = sel & bench.elig & (bench.y >= 0) & np.isfinite(score)
    y = (bench.y[sel] > 0).astype(int)
    if y.sum() == 0:
        return {"auroc_exact": NAN, "average_precision_exact": NAN}
    ap = float(average_precision_score(y, score[sel]))
    if y.sum() == len(y):
        return {"auroc_exact": NAN, "average_precision_exact": ap}
    auc = float(roc_auc_score(y, score[sel]))
    return {"auroc_exact": auc, "average_precision_exact": ap}


# ----------------------------------------------- threshold
def select_threshold(
    bench: Benchmark,
    score: Any,
    wells: Sequence[str],
    fa_per_day_max: float = 1.0,
    occupation_max: float = 0.05,
    min_normal_days: float = 0.5,
) -> tuple[float | None, list[dict[str, Any]]]:
    """The alarm threshold, frozen on development predictions.

    Among the candidate thresholds that meet both budgets (false alarms
    per normal day and share of normal time in alarm) on the given
    out-of-fold predictions, the one with the highest episode coverage;
    a tie goes to the lower false alarm rate. Returns the threshold
    (None if no candidate is feasible) and the table of candidates.
    """
    score = np.asarray(score, dtype=np.float64)
    sel = bench.well_mask(wells)[bench.well_of_decision]
    cand = score[sel & bench.elig & (bench.y == 0) & np.isfinite(score)]
    if len(cand) == 0:
        return None, []
    qs = np.unique(
        np.r_[
            np.quantile(cand, np.linspace(0.5, 1.0, 101)), np.linspace(0.05, 0.99, 48)
        ]
    )
    table: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for th in qs:
        S, _, _ = well_statistics(bench, score, None, wells, float(th))
        m = detection_metrics(S.sum(0))
        fa = float(m["false_alarms_per_normal_day"])
        occ = float(m["normal_time_occupation"])
        cov = float(m["episode_coverage"])
        days = float(m["normal_days"])
        feasible = (
            days >= min_normal_days and fa <= fa_per_day_max and occ <= occupation_max
        )
        table.append(
            {
                "theta": float(th),
                "fa_per_day": fa,
                "occupation": occ,
                "coverage": cov,
                "feasible": bool(feasible),
            }
        )
        if feasible and (
            best is None or (cov, -fa) > (best["coverage"], -best["fa_per_day"])
        ):
            best = table[-1]
    return (best["theta"] if best else None), table


# --------------------------------------------------------- output
def jsonable(o: Any) -> Any:
    """Convert numpy values to JSON types; NaN becomes null."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o
