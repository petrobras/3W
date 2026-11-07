"""The committed protocol files agree with each other."""

import csv
import json

import numpy as np
from held_out_well.benchmark import PROTOCOL_DIR, Benchmark
from held_out_well.grid import array_digest, make_decision_table, n_decisions
from held_out_well.loader import TABLE_ARRAYS, compare_with_committed

TEST_WELLS = [
    "WELL-00001",
    "WELL-00004",
    "WELL-00011",
    "WELL-00013",
    "WELL-00015",
    "WELL-00023",
    "WELL-00026",
    "WELL-00035",
    "WELL-00037",
    "WELL-00041",
]


def load_table():
    z = np.load(PROTOCOL_DIR / "decisions.npz")
    return {k: z[k] for k in TABLE_ARRAYS}


def test_decision_arrays_match_their_digests():
    digests = json.loads((PROTOCOL_DIR / "decisions_arrays.sha256.json").read_text())
    table = load_table()
    for name in TABLE_ARRAYS:
        assert array_digest(table[name]) == digests[name]["sha256"], name
        assert str(table[name].dtype) == digests[name]["dtype"]


def test_benchmark_shapes_and_wells():
    b = Benchmark.load()
    assert b.n == 962002 and len(b.inst_rows) == 1119 and len(b.wells) == 40
    assert b.wells_with_role("test") == TEST_WELLS
    assert len(b.wells_with_role("dev")) == 30
    for k in range(5):
        assert len(b.wells_with_role("dev", k)) == 6
    assert (np.bincount(b.inst) == [n_decisions(r) for r in b.inst_rows]).all()
    assert b.y.min() >= 0 and b.y.max() <= 9
    assert b.dev_classes() == [0, 1, 2, 4, 5, 7, 8, 9]
    assert len(b.ep_start) == 481
    in_test = np.isin(b.ep_well, [b.wells.index(w) for w in TEST_WELLS])
    assert in_test.sum() == 110


def test_instances_agree_with_the_label_index():
    index = json.loads((PROTOCOL_DIR / "index_labels.json").read_text())
    by_path = {f["path"]: f for f in index["files"]}
    with open(PROTOCOL_DIR / "instances.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["path"] for r in rows] == sorted(by_path)
    for r in rows:
        rec = by_path[r["path"]]
        assert int(r["rows"]) == rec["rows"] and r["sha"] == rec["sha"]
        assert r["well"] == rec["well"]


def test_label_index_regenerates_the_decision_labels():
    """Labels and states of the decision table follow from the runs."""
    index = json.loads((PROTOCOL_DIR / "index_labels.json").read_text())
    files = sorted(index["files"], key=lambda f: f["path"])
    table = load_table()
    flags = {}
    for i, f in enumerate(files):
        flags[f["path"]] = table["elig"][table["inst"] == i]
    inst, r, y, elig, state = make_decision_table(files, flags)
    assert np.array_equal(inst, table["inst"]) and np.array_equal(r, table["r"])
    assert np.array_equal(y, table["y"]) and np.array_equal(state, table["state"])
    assert np.array_equal(elig, table["elig"])


def test_comparison_with_the_committed_files():
    index = json.loads((PROTOCOL_DIR / "index_labels.json").read_text())
    table = load_table()
    assert compare_with_committed(index, table)["all_equal"]
    # a changed label run is noticed
    index["files"][3]["class_runs"][-1][2] += 1
    result = compare_with_committed(index, table)
    assert not result["all_equal"]
    assert result["files_differing_by_field"]["class_runs"] == 1
    table["y"] = table["y"].copy()
    table["y"][0] += 1
    assert not compare_with_committed(index, table)["decision_arrays"]["y"]["equal"]
