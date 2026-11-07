"""Small synthetic benchmarks for the tests."""

import numpy as np
from held_out_well.benchmark import Benchmark
from held_out_well.grid import make_decision_table, n_decisions


def fake_file(path, well, rows, class_runs, state_runs=None):
    """A label-index record with the given label runs."""
    if state_runs is None:
        state_runs = [[-1, 0, 3600], [0, 3600, rows]]
    return {
        "path": path,
        "well": well,
        "rows": rows,
        "class_runs": class_runs,
        "state_runs": state_runs,
    }


def make_bench(files, elig=None, manifest=None):
    """A Benchmark over fake files (all wells are test wells)."""
    files = sorted(files, key=lambda f: f["path"])
    flags = {}
    for f in files:
        if elig and f["path"] in elig:
            flags[f["path"]] = elig[f["path"]]
        else:
            flags[f["path"]] = np.ones(n_decisions(f["rows"]), bool)
    inst, r, y, el, st = make_decision_table(files, flags)
    wells = sorted({f["well"] for f in files})
    roles = manifest or {w: ("test", -1) for w in wells}
    return Benchmark(
        inst,
        r,
        y,
        el,
        st,
        [f["rows"] for f in files],
        [wells.index(f["well"]) for f in files],
        [f["path"] for f in files],
        wells,
        [f["class_runs"] for f in files],
        roles,
    )


def scenario():
    """One well, two instances.

    A: normal, then event 2 with a known onset (transient then steady);
    B: all normal.
    """
    a = fake_file(
        "dataset/2/WELL-00001_a.parquet",
        "WELL-00001",
        9000,
        [[-1, 0, 3600], [0, 3600, 5100], [102, 5100, 6600], [2, 6600, 9000]],
    )
    b = fake_file(
        "dataset/0/WELL-00001_b.parquet",
        "WELL-00001",
        8000,
        [[-1, 0, 3600], [0, 3600, 8000]],
    )
    return make_bench([a, b])
