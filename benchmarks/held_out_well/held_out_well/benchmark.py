"""The committed benchmark tables: decisions, episodes and wells.

``Benchmark`` holds the decision table (one row per decision of every
real instance), the episodes derived from the label runs, and the well
manifest. It needs no sensor data: scoring a prediction only needs this
object and one score (and class) per decision.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .constants import MIN_EPISODE_ROWS
from .labels import episodes_from_runs

PROTOCOL_DIR = Path(__file__).resolve().parents[1] / "protocol"


def load_manifest(path: str | Path) -> dict[str, tuple[str, int]]:
    """Read ``manifest.csv`` as well -> (role, development fold)."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["well_id"]] = (row["role"], int(row["dev_fold"]))
    return out


class Benchmark:
    """Decision table, episodes and wells of the committed protocol."""

    def __init__(
        self,
        inst: Any,
        r: Any,
        y: Any,
        elig: Any,
        state: Any,
        inst_rows: Sequence[int],
        inst_well: Sequence[int],
        inst_path: Sequence[str],
        wells: Sequence[str],
        class_runs: Sequence[Sequence[Sequence[int]]],
        manifest: dict[str, tuple[str, int]],
    ):
        self.inst = np.asarray(inst, dtype=np.int32)
        self.r = np.asarray(r, dtype=np.int32)
        self.y = np.asarray(y, dtype=np.int8)
        self.elig = np.asarray(elig, dtype=bool)
        self.state = np.asarray(state, dtype=np.int8)
        self.n = len(self.inst)
        self.inst_rows = np.asarray(inst_rows, dtype=np.int64)
        self.inst_well = np.asarray(inst_well, dtype=np.int32)
        self.inst_path = list(inst_path)
        self.wells = list(wells)
        self.manifest = manifest
        counts = np.bincount(self.inst, minlength=len(inst_rows))
        self.inst_offset = np.r_[0, np.cumsum(counts)]
        self.well_of_decision = self.inst_well[self.inst]
        if not np.all(np.diff(self.inst) >= 0):
            raise ValueError("decisions must be sorted by instance")
        eps = []
        for i, runs in enumerate(class_runs):
            for ep in episodes_from_runs(runs):
                if ep["end"] - ep["start"] >= MIN_EPISODE_ROWS:
                    eps.append(
                        (i, ep["start"], ep["end"], ep["event"], ep["known_onset"])
                    )
        e = np.array(eps, dtype=np.int64).reshape(-1, 5)
        self.ep_inst = e[:, 0]
        self.ep_start = e[:, 1]
        self.ep_end = e[:, 2]
        self.ep_event = e[:, 3]
        self.ep_known = e[:, 4]
        self.ep_well = self.inst_well[self.ep_inst]

    @classmethod
    def load(cls, protocol_dir: str | Path | None = None) -> Benchmark:
        """Load the committed protocol files."""
        p = Path(protocol_dir) if protocol_dir else PROTOCOL_DIR
        d = np.load(p / "decisions.npz")
        with open(p / "instances.csv") as f:
            inst = list(csv.DictReader(f))
        wells = sorted({x["well"] for x in inst})
        widx = {w: i for i, w in enumerate(wells)}
        index = json.loads((p / "index_labels.json").read_text())
        runs_by_path = {x["path"]: x["class_runs"] for x in index["files"]}
        return cls(
            d["inst"],
            d["r"],
            d["y"],
            d["elig"],
            d["state"],
            [int(x["rows"]) for x in inst],
            [widx[x["well"]] for x in inst],
            [x["path"] for x in inst],
            wells,
            [runs_by_path[x["path"]] for x in inst],
            load_manifest(p / "manifest.csv"),
        )

    def wells_with_role(self, role: str, fold: int | None = None) -> list[str]:
        """Wells of a role ('dev' or 'test'), optionally of one fold."""
        return [
            w
            for w in self.wells
            if self.manifest[w][0] == role
            and (fold is None or self.manifest[w][1] == fold)
        ]

    def well_mask(self, wells: Sequence[str]) -> np.ndarray:
        """Boolean vector over ``self.wells`` of the given wells."""
        m = np.zeros(len(self.wells), dtype=bool)
        m[[self.wells.index(w) for w in wells]] = True
        return m

    def dev_classes(self) -> list[int]:
        """Classes that can be trained and scored on different wells.

        Class 0 and every event class with episodes in two or more
        development wells.
        """
        dev = set(self.wells_with_role("dev"))
        cnt: dict[int, set[int]] = {}
        for w, ev in zip(self.ep_well, self.ep_event):
            if self.wells[w] in dev:
                cnt.setdefault(int(ev), set()).add(int(w))
        return [0] + sorted(c for c, ws in cnt.items() if len(ws) >= 2)
