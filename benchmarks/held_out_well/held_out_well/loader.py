"""Reading 3W instances with the 3W Toolkit; rebuilding the label index.

Instances are read by ``ThreeWToolkit.dataset.ParquetDataset``. From
each instance this module derives what the benchmark needs:

* the label runs and the eligibility of every decision, which make up
  the committed label index and decision table, and
* the process and valve-state arrays that the baselines use.

Rebuilding the index from the dataset and comparing it with the
committed files shows that the files still describe the dataset. For a
new dataset version, the same code rebuilds the tables; the well
manifest would then need a new committed draw.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ThreeWToolkit.dataset import (
    DatasetOutputs,
    ParquetDataset,
    ParquetDatasetConfig,
)

from .benchmark import PROTOCOL_DIR
from .constants import DATASET_DIR, NA, PROCESS_CHANNELS, STATE_CHANNELS
from .grid import array_digest, eligibility_from_any_process, make_decision_table
from .labels import runs_from_labels

TABLE_ARRAYS = ("inst", "r", "y", "elig", "state")
RECORD_FIELDS = (
    "path",
    "well",
    "folder_class",
    "sha",
    "size",
    "rows",
    "t0",
    "t_last",
    "n_dt_ne_1",
    "max_dt",
    "min_dt",
    "class_runs",
    "state_runs",
    "n_na_class",
    "any_process_finite_frac",
)


def open_dataset(
    root: str | Path, file_list: Sequence[str] | None = None
) -> ParquetDataset:
    """Open the real instances of a 3W dataset folder.

    ``root`` is the folder that holds the sub-folders ``0`` to ``9``.
    ``file_list`` selects instances by their path relative to ``root``.
    """
    if file_list is None:
        cfg = ParquetDatasetConfig(
            path=root, event_type=["real"], target_column="class"
        )
    else:
        cfg = ParquetDatasetConfig(
            path=root,
            split="list",
            file_list=list(file_list),
            event_type=["real"],
            target_column="class",
        )
    return ParquetDataset(cfg)


def well_of(file_name: str | Path) -> str:
    """Well identifier of an instance file name (``WELL-00001_...``)."""
    return Path(file_name).name.split("_")[0]


def git_blob_sha1(data: bytes) -> str:
    """The git object id of a file, as recorded in the label index."""
    header = b"blob " + str(len(data)).encode() + b"\0"
    return hashlib.sha1(header + data).hexdigest()


def _int_labels(series: Any) -> np.ndarray:
    """Integer label vector with NA coded as -1."""
    v = series.to_numpy(dtype="float64", na_value=np.nan)
    return np.where(np.isnan(v), NA, v).astype(np.int64)


def process_present(out: DatasetOutputs) -> np.ndarray:
    """True at rows where a process channel is finite."""
    sig = out.signal.reindex(columns=PROCESS_CHANNELS)
    values = sig.to_numpy(dtype="float64", na_value=np.nan)
    return np.isfinite(values).any(axis=1)


def signal_arrays(out: DatasetOutputs) -> tuple[np.ndarray, np.ndarray]:
    """Process channels and valve states as float32 matrices.

    Columns missing from an instance are NaN. Shapes are
    (rows, 18) and (rows, 9).
    """
    sig = out.signal.reindex(columns=PROCESS_CHANNELS + STATE_CHANNELS)
    values = sig.to_numpy(dtype="float64", na_value=np.nan)
    n = len(PROCESS_CHANNELS)
    return values[:, :n].astype(np.float32), values[:, n:].astype(np.float32)


def index_record(
    out: DatasetOutputs,
) -> tuple[dict[str, Any], np.ndarray]:
    """Label-index record of one instance and its eligibility flags.

    The record carries the label runs (raw labels with NA as -1), the
    operational-state runs and sanity facts about the time axis. ``sha``
    and ``size`` need the file bytes and are added by ``build_index``.
    """
    rel = Path(out.metadata["file_name"])
    sig = out.signal
    if out.label is None:
        raise ValueError(f"{rel}: no label column")
    labels = _int_labels(out.label)
    states = _int_labels(sig["state"])
    stamps = pd.DatetimeIndex(sig.index)
    t = stamps.values.astype("datetime64[ms]")
    dt = np.diff(t).astype("int64") / 1000.0
    any_proc = process_present(out)
    record = {
        "path": f"{DATASET_DIR}/{rel.as_posix()}",
        "well": well_of(rel),
        "folder_class": int(out.metadata["event_class"]),
        "rows": len(sig),
        "t0": str(stamps[0].strftime("%Y-%m-%d %H:%M:%S")),
        "t_last": str(stamps[-1].strftime("%Y-%m-%d %H:%M:%S")),
        "n_dt_ne_1": int((dt != 1.0).sum()),
        "max_dt": float(dt.max()) if len(dt) else 1.0,
        "min_dt": float(dt.min()) if len(dt) else 1.0,
        "class_runs": runs_from_labels(labels),
        "state_runs": runs_from_labels(states),
        "n_na_class": int((labels == NA).sum()),
        "any_process_finite_frac": float(any_proc.mean()),
    }
    return record, eligibility_from_any_process(any_proc)


def build_index(
    root: str | Path,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Rebuild the label index of all real instances under ``root``.

    Returns the index (same layout as ``protocol/index_labels.json``)
    and the eligibility flags of every decision, by instance path.
    """
    root = Path(root)
    t0 = time.time()
    ds = open_dataset(root)
    files: list[dict[str, Any]] = []
    elig: dict[str, np.ndarray] = {}
    for i in range(len(ds)):
        out = ds[i]
        record, flags = index_record(out)
        data = (root / Path(out.metadata["file_name"])).read_bytes()
        record["sha"] = git_blob_sha1(data)
        record["size"] = len(data)
        files.append({k: record[k] for k in RECORD_FIELDS})
        elig[record["path"]] = flags
        if progress is not None:
            progress(i + 1, len(ds))
    files.sort(key=lambda f: f["path"])
    meta = {"n_files": len(files), "seconds": round(time.time() - t0, 1)}
    return {"meta": meta, "files": files}, elig


def decision_table(
    index: dict[str, Any], elig: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Decision-table arrays of a rebuilt index."""
    arrays = make_decision_table(index["files"], elig)
    return dict(zip(TABLE_ARRAYS, arrays))


def compare_with_committed(
    index: dict[str, Any],
    table: dict[str, np.ndarray],
    protocol_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compare a rebuilt index and table with the committed ones.

    Reports, per record field, how many instances differ, and whether
    the SHA-256 of every decision array equals the committed value.
    """
    p = Path(protocol_dir) if protocol_dir else PROTOCOL_DIR
    committed = json.loads((p / "index_labels.json").read_text())["files"]
    digests = json.loads((p / "decisions_arrays.sha256.json").read_text())
    with open(p / "instances.csv") as f:
        table_order = [row["path"] for row in csv.DictReader(f)]
    mine = {f["path"]: f for f in index["files"]}
    theirs = {f["path"]: f for f in committed}
    result: dict[str, Any] = {
        "n_files_rebuilt": len(mine),
        "n_files_committed": len(theirs),
        "same_paths": sorted(mine) == sorted(theirs),
        "same_order": [f["path"] for f in index["files"]] == table_order,
    }
    differing: dict[str, int] = {}
    for field in RECORD_FIELDS:
        bad = [
            path
            for path in theirs
            if path not in mine or mine[path].get(field) != theirs[path][field]
        ]
        differing[field] = len(bad)
    result["files_differing_by_field"] = differing
    arrays = {}
    for name in TABLE_ARRAYS:
        want = digests[name]["sha256"]
        got = array_digest(table[name])
        arrays[name] = {"equal": got == want, "sha256": got}
    result["decision_arrays"] = arrays
    result["all_equal"] = (
        result["same_paths"]
        and result["same_order"]
        and not any(differing.values())
        and all(a["equal"] for a in arrays.values())
    )
    return result
