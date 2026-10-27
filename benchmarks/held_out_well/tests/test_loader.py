"""Reading instances through the 3W Toolkit loader."""

import numpy as np
import pandas as pd
import pytest
from held_out_well.constants import PROCESS_CHANNELS, STATE_CHANNELS
from held_out_well.loader import (
    build_index,
    decision_table,
    git_blob_sha1,
    open_dataset,
    signal_arrays,
    well_of,
)
from ThreeWToolkit.dataset import parquet_dataset

N_FILES = 4


def write_instance(root, folder, name, labels, states, nan_from=None, drop=()):
    """Write one synthetic instance as a parquet file."""
    rows = len(labels)
    index = pd.date_range("2020-01-01 00:00:00", periods=rows, freq="1s")
    data = {c: np.linspace(0.0, 1.0, rows) for c in PROCESS_CHANNELS}
    data.update({c: np.zeros(rows) for c in STATE_CHANNELS})
    df = pd.DataFrame(data, index=index)
    if nan_from is not None:
        df.loc[df.index[nan_from:], PROCESS_CHANNELS] = np.nan
    df = df.drop(columns=list(drop))
    df["class"] = pd.array([None if v < 0 else v for v in labels], dtype="Int64")
    df["state"] = pd.array([None if v < 0 else v for v in states], dtype="Int64")
    path = root / str(folder) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow")


@pytest.fixture
def dataset_root(tmp_path, monkeypatch):
    rules = parquet_dataset.DATASET_VALIDATION_RULES["2.0.0"]
    monkeypatch.setitem(rules, "total_parquet_files", N_FILES)
    na = [-1] * 3600
    write_instance(
        tmp_path,
        0,
        "WELL-00001_20200101000000.parquet",
        na + [0] * 1400,
        na + [0] * 1400,
    )
    labels = na + [0] * 1200 + [102] * 600 + [2] * 1600
    write_instance(
        tmp_path, 2, "WELL-00001_20200102000000.parquet", labels, na + [1] * 3400
    )
    write_instance(
        tmp_path,
        3,
        "WELL-00002_20200103000000.parquet",
        na + [3] * 900,
        na + [0] * 900,
        nan_from=3700,
        drop=("P-JUS-BS", "QBS"),
    )
    write_instance(
        tmp_path, 0, "SIMULATED_00001.parquet", na + [0] * 100, na + [0] * 100
    )
    return tmp_path


def test_only_real_instances_are_listed(dataset_root):
    ds = open_dataset(dataset_root)
    assert len(ds) == 3
    names = [str(p) for p in ds.files_events]
    assert not any("SIMULATED" in n for n in names)
    subset = open_dataset(dataset_root, ["2/WELL-00001_20200102000000.parquet"])
    assert len(subset) == 1


def test_index_records(dataset_root):
    index, _ = build_index(dataset_root)
    f1, f2, f3 = index["files"]
    assert f1["path"] == "dataset/0/WELL-00001_20200101000000.parquet"
    assert f1["class_runs"] == [[-1, 0, 3600], [0, 3600, 5000]]
    assert f1["state_runs"] == [[-1, 0, 3600], [0, 3600, 5000]]
    assert f1["well"] == "WELL-00001" and f1["folder_class"] == 0
    assert f1["rows"] == 5000 and f1["n_na_class"] == 3600
    assert f1["t0"] == "2020-01-01 00:00:00"
    assert f1["t_last"] == "2020-01-01 01:23:19"
    assert f1["n_dt_ne_1"] == 0 and f1["min_dt"] == f1["max_dt"] == 1.0
    assert f1["any_process_finite_frac"] == 1.0
    assert f2["class_runs"] == [
        [-1, 0, 3600],
        [0, 3600, 4800],
        [102, 4800, 5400],
        [2, 5400, 7000],
    ]
    assert f2["state_runs"] == [[-1, 0, 3600], [1, 3600, 7000]]
    assert f2["folder_class"] == 2
    assert f3["well"] == "WELL-00002"
    assert f3["any_process_finite_frac"] == pytest.approx(3700 / 4500)
    for f in index["files"]:
        assert f["size"] > 0 and len(f["sha"]) == 40


def test_eligibility_follows_the_process_channels(dataset_root):
    _, elig = build_index(dataset_root)
    first = elig["dataset/0/WELL-00001_20200101000000.parquet"]
    assert first.all() and len(first) == 47
    flags = elig["dataset/3/WELL-00002_20200103000000.parquet"]
    # channels end at row 3700; 810 finite rows are needed in the window
    assert len(flags) == 30
    assert flags[:7].all() and not flags[7:].any()


def test_decision_table_from_rebuilt_index(dataset_root):
    index, elig = build_index(dataset_root)
    table = decision_table(index, elig)
    assert set(table) == {"inst", "r", "y", "elig", "state"}
    second = table["inst"] == 1
    r = table["r"][second]
    y = table["y"][second]
    assert y[r == 3600][0] == 0 and y[r == 4800][0] == 2
    assert y[r == 5400][0] == 2 and y[r == 6990][0] == 2
    assert (table["state"][second] == 1).all()


def test_signal_arrays_fill_missing_columns(dataset_root):
    ds = open_dataset(dataset_root, ["3/WELL-00002_20200103000000.parquet"])
    cont, state = signal_arrays(ds[0])
    assert cont.shape == (4500, 18) and state.shape == (4500, 9)
    assert cont.dtype == np.float32 and state.dtype == np.float32
    missing = PROCESS_CHANNELS.index("QBS")
    assert np.isnan(cont[:, missing]).all()
    present = PROCESS_CHANNELS.index("P-ANULAR")
    assert np.isfinite(cont[:3700, present]).all()
    assert np.isnan(cont[3700:, present]).all()


def test_names_and_hashes():
    assert well_of("0/WELL-00012_20200101000000.parquet") == "WELL-00012"
    assert git_blob_sha1(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
