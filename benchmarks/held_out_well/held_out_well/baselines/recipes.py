"""Training recipes of the reference baselines.

Four baselines come with the benchmark. Two need no class labels (a
windowed z-test and a volatility detector fitted on normal operation)
and two are supervised (a random forest and LightGBM on the 333 causal
features). Hyper-parameters were fixed before any scoring and nothing
was tuned on a fold. The grouped fold plan trains the final model, the
one that predicts the sealed test wells, on all development wells.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..benchmark import Benchmark
from ..constants import N_CLASSES

CAP_PER_CLASS = 20000  # training decisions per class and fold
TRAIN_EVERY = 2  # use every second decision
FoldPlan = list[tuple[str, list[str], list[str]]]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fold_plan(bench: Benchmark) -> FoldPlan:
    """Grouped folds: 5 development folds, then development -> test."""
    dev = bench.wells_with_role("dev")
    plan: FoldPlan = []
    for k in range(5):
        held = bench.wells_with_role("dev", k)
        plan.append((f"dev{k}", [w for w in dev if w not in held], held))
    plan.append(("test", dev, bench.wells_with_role("test")))
    return plan


def train_sample(
    bench: Benchmark,
    have: np.ndarray,
    wells: Sequence[str],
    cap_per_class: int,
    rng: np.random.Generator,
    every: int = TRAIN_EVERY,
) -> np.ndarray:
    """Training decisions: eligible, labelled, of the given wells.

    Takes every ``every``-th such decision and caps the number per
    class at random.
    """
    m = bench.well_mask(wells)[bench.well_of_decision]
    m = m & bench.elig & (bench.y >= 0) & have
    idx = np.flatnonzero(m)[::every]
    chosen = []
    for c in range(N_CLASSES):
        ic = idx[bench.y[idx] == c]
        if len(ic) > cap_per_class:
            ic = rng.choice(ic, cap_per_class, replace=False)
        chosen.append(ic)
    return np.sort(np.concatenate(chosen))


def predict_full(model: Any, X: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Class probabilities of the given rows, columns 0..9."""
    p = np.zeros((len(rows), N_CLASSES), dtype=np.float32)
    chunk = 100000
    cols = np.asarray(model.classes_, dtype=int)
    for a in range(0, len(rows), chunk):
        part = model.predict_proba(X[rows[a : a + chunk]])
        p[a : a + chunk][:, cols] = part
    return p


def run_folds(
    bench: Benchmark,
    X: np.ndarray,
    have: np.ndarray,
    make_model: Callable[[], Any],
    plan: FoldPlan,
    cap_per_class: int = CAP_PER_CLASS,
    seed: int = 0,
) -> np.ndarray:
    """Fit on the training wells of each fold, predict the others.

    Returns class probabilities (float16) for every eligible decision of
    the predicted wells and NaN elsewhere.
    """
    proba = np.full((bench.n, N_CLASSES), np.nan, dtype=np.float16)
    for name, train_wells, pred_wells in plan:
        t0 = time.time()
        salt = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 1000
        rng = np.random.default_rng(seed + salt)
        idx = train_sample(bench, have, train_wells, cap_per_class, rng)
        model = make_model()
        model.fit(X[idx], bench.y[idx].astype(int))
        rows = np.flatnonzero(
            bench.well_mask(pred_wells)[bench.well_of_decision] & bench.elig & have
        )
        proba[rows] = predict_full(model, X, rows).astype(np.float16)
        log(
            f"  fold {name}: train {len(idx)}, predict {len(rows)}, {time.time() - t0:.0f}s"
        )
    return proba


def to_score_pred(proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Event score P(class > 0) and the most probable class."""
    p = proba.astype(np.float32)
    known = np.isfinite(p[:, 0])
    score = np.where(known, 1.0 - p[:, 0], np.nan).astype(np.float64)
    best = np.argmax(np.nan_to_num(p, nan=-1.0), axis=1)
    return score, np.where(known, best, -1).astype(np.int8)


class VolatilityDetector:
    """One-class baseline: scale-free variability against normal.

    For each process channel the 15-minute variability index
    (``cv_w3``) is divided by its 99th percentile over the normal
    training decisions; the score is the maximum over channels,
    ``v``, mapped to ``v / (v + 1)``. Only the label normal versus not
    normal of the training wells is used.
    """

    classes_ = np.array([0, 1])

    def __init__(self, feature_names: Sequence[str]):
        self.cols = [i for i, n in enumerate(feature_names) if n.endswith("|cv_w3")]
        self.q: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> VolatilityDetector:
        z = X[:, self.cols][np.asarray(y) == 0]
        q = np.nanpercentile(z, 99, axis=0)
        self.q = np.where(np.isfinite(q) & (q > 0), q, np.nan)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X[:, self.cols] / self.q
        with np.errstate(all="ignore"):
            v = np.nanmax(np.where(np.isfinite(z), z, -np.inf), axis=1)
        v = np.where(np.isfinite(v), np.maximum(v, 0.0), 0.0)
        s = v / (v + 1.0)
        return np.stack([1.0 - s, s], axis=1).astype(np.float32)


def make_random_forest(smoke: bool = False) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=10 if smoke else 100,
        min_samples_leaf=10,
        max_features="sqrt",
        max_samples=0.3,
        class_weight="balanced_subsample",
        n_jobs=4,
        random_state=0,
    )


def make_lightgbm(smoke: bool = False) -> Any:
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=20 if smoke else 200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=1.0,
        class_weight="balanced",
        n_jobs=4,
        random_state=0,
        verbose=-1,
    )
