"""Run the reference baselines on a local copy of the 3W dataset.

    python -m held_out_well baselines --dataset-root dataset --out out

Instances are read with the 3W Toolkit loader, features are built per
instance, every model is fitted fold by fold (grouped by well, see
``recipes.fold_plan``) and scored with the benchmark scorer. The alarm
threshold of each model is chosen on its development out-of-fold scores.
Writes ``preds_<model>.npz`` (``score`` and, for classifiers, ``pred``)
and ``baselines.json``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from ..benchmark import Benchmark
from ..constants import DATASET_DIR
from ..loader import open_dataset, signal_arrays
from ..scoring import jsonable, report, select_threshold
from . import features as F
from . import recipes as R

MODELS = ("simple", "volatility", "rf", "lgbm")


def build_features(
    bench: Benchmark,
    root: str | Path,
    active: Sequence[int],
    workers: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Feature matrix and simple-detector score for active instances.

    Rows of instances that are not active stay NaN and ``have`` is
    False there.
    """
    ds = open_dataset(root)
    where = {f"{DATASET_DIR}/{p.as_posix()}": j for j, p in enumerate(ds.files_events)}
    missing = [p for p in bench.inst_path if p not in where]
    if missing:
        raise FileNotFoundError(f"{len(missing)} instances missing, e.g. {missing[0]}")
    X = np.full((bench.n, len(F.feature_names())), np.nan, dtype=np.float32)
    simple = np.full(bench.n, np.nan, dtype=np.float32)
    have = np.zeros(bench.n, dtype=bool)
    t0 = time.time()

    def one(i: int):
        cont, state = signal_arrays(ds[where[bench.inst_path[i]]])
        return i, F.instance_features(cont, state), F.simple_detector_score(cont)

    with ThreadPoolExecutor(workers) as ex:
        for done, (i, fx, sc) in enumerate(ex.map(one, active), 1):
            a, b = bench.inst_offset[i], bench.inst_offset[i + 1]
            if fx.shape[0] != b - a:
                raise ValueError(f"{bench.inst_path[i]}: {fx.shape[0]} != {b - a}")
            X[a:b], simple[a:b], have[a:b] = fx, sc, True
            if done % 100 == 0:
                R.log(
                    f"features {done}/{len(active)} instances ({time.time() - t0:.0f}s)"
                )
    return X, simple, have


def run(
    root: str | Path,
    out_dir: str | Path,
    models: Sequence[str] = MODELS,
    smoke: bool = False,
    workers: int = 4,
    boot: int = 2000,
) -> dict[str, Any]:
    """Fit and score the baselines; returns the results dictionary."""
    bench = Benchmark.load()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan = R.fold_plan(bench)
    dev, test = bench.wells_with_role("dev"), bench.wells_with_role("test")
    cap = R.CAP_PER_CLASS
    if smoke:  # code-path check on development wells only
        d0, d1, d2 = (bench.wells_with_role("dev", k) for k in (0, 1, 2))
        plan = [("dev0", d1 + d2, d0), ("smoke_final", d0 + d1, d2)]
        dev, test, cap = d0 + d2, [], 500
    active_wells = set(dev) | set(test) | {w for _, tr, _ in plan for w in tr}
    active = [
        i
        for i in range(len(bench.inst_rows))
        if bench.wells[bench.inst_well[i]] in active_wells
    ]
    X, simple_raw, have = build_features(bench, root, active, workers)
    names = F.feature_names()
    results: dict[str, Any] = {"smoke": smoke, "n_features": len(names)}

    def finish(name: str, score: np.ndarray, pred: np.ndarray | None) -> None:
        theta, _ = select_threshold(bench, score, dev)
        reports = {
            "dev_oof": report(
                bench, score, pred, dev, theta, bench.dev_classes(), B=boot
            )
        }
        if test:
            reports["test"] = report(bench, score, pred, test, theta, B=boot)
        results[name] = {"theta": theta, "reports": reports}
        arrays = {"score": score.astype(np.float32)}
        if pred is not None:
            arrays["pred"] = pred
        np.savez_compressed(out / f"preds_{name}.npz", **arrays)
        (out / "baselines.json").write_text(json.dumps(jsonable(results), indent=1))
        R.log(f"{name}: theta={theta}")

    if "simple" in models:
        unit = F.simple_score_to_unit(simple_raw)
        sel = bench.elig & have
        finish("simple", np.where(sel, np.nan_to_num(unit, nan=0.0), np.nan), None)
    if "volatility" in models:
        proba = R.run_folds(
            bench, X, have, lambda: R.VolatilityDetector(names), plan, cap
        )
        finish("volatility", R.to_score_pred(proba)[0], None)
    for name, maker in (("rf", R.make_random_forest), ("lgbm", R.make_lightgbm)):
        if name in models:
            R.log(f"== {name}")
            proba = R.run_folds(bench, X, have, partial(maker, smoke), plan, cap)
            score, pred = R.to_score_pred(proba)
            finish(name, score, pred)
    return results
