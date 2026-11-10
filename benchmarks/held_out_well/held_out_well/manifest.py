"""Deterministic, hash-ordered well split with machine-checked rules.

Only label and time metadata (the label index) is read: no sensor value
and no model output can influence the split. Randomness is the SHA-256
of a public drand beacon signature whose round was committed before it
existed (``protocol/COMMIT-A.md``). Pure hashing, no numpy RNG, so the
output is identical across library versions.

This module is a typeset copy of ``protocol/frozen/manifest.py.txt``,
the file whose SHA-256 was committed before the draw. The tests check
that both produce the same ``manifest.csv`` byte for byte.

Constraints of protocol v1:

K1  there are 10 test wells.
K2  every event class 1..9 has at least one supported test well and at
    least one supported development well; when a class has four or more
    wells overall, development has at least two of them.
K3  labelled normal time of the test wells is at least 24 h, with at
    least two test wells holding 6 h of normal each.
K4  test labelled rows are 15 to 35 % of all labelled rows.
K5  development wells are partitioned into 5 folds of 6; every fold has
    a well with 2 h of normal; every class with two or more development
    wells is supported in two or more different folds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .constants import N_DEV_FOLDS, N_TEST_WELLS, PROTOCOL_ID
from .labels import class_support

MAX_ATTEMPTS = 200000
COLUMNS = [
    "well_id",
    "role",
    "dev_fold",
    "n_files",
    "labelled_hours",
    "normal_hours",
    "classes_supported",
]


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hash_order(seed_hex: str, tag: str, attempt: int, items: Iterable[str]):
    """Sort items by SHA-256 of seed, protocol, tag, attempt and item.

    This is a stable, uniform permutation that needs no RNG library.
    """

    def key(x: str) -> str:
        text = f"{seed_hex}|{PROTOCOL_ID}|{tag}|{attempt}|{x}"
        return hashlib.sha256(text.encode()).hexdigest()

    return sorted(items, key=key)


def well_stats(index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-well statistics from the parsed label index."""
    wells: dict[str, dict[str, Any]] = {}
    for f in index["files"]:
        w = wells.setdefault(
            f["well"],
            {
                "n_files": 0,
                "labelled_rows": 0,
                "normal_rows": 0,
                "class_rows": {},
                "supported": set(),
            },
        )
        w["n_files"] += 1
        rows, supported = class_support(f["class_runs"])
        for ev, n in rows.items():
            w["class_rows"][ev] = w["class_rows"].get(ev, 0) + n
        w["labelled_rows"] += sum(rows.values())
        w["normal_rows"] += rows.get(0, 0)
        w["supported"].update(supported)
    return wells


def check_test(test: list[str], stats: dict[str, dict[str, Any]]):
    """Return the first violated test-set rule, or None."""
    wells = sorted(stats)
    dev = [w for w in wells if w not in test]
    for c in range(1, 10):
        wc = {w for w in wells if c in stats[w]["supported"]}
        if not wc:
            continue
        t, d = len(wc & set(test)), len(wc & set(dev))
        if t < 1 or d < 1:
            return f"class {c}: test {t} dev {d}"
        if len(wc) >= 4 and d < 2:
            return f"class {c}: dev has {d} wells but class has {len(wc)}"
    hours = sum(stats[w]["normal_rows"] for w in test) / 3600
    if hours < 24:
        return f"test normal {hours:.1f} h < 24"
    rich = sum(1 for w in test if stats[w]["normal_rows"] >= 6 * 3600)
    if rich < 2:
        return "fewer than 2 test wells with >=6 h normal"
    share = sum(stats[w]["labelled_rows"] for w in test) / sum(
        stats[w]["labelled_rows"] for w in wells
    )
    if not 0.15 <= share <= 0.35:
        return f"test row share {share:.3f}"
    return None


def check_folds(folds: list[list[str]], stats: dict[str, dict[str, Any]]):
    """Return the first violated development-fold rule, or None."""
    for k, fold in enumerate(folds):
        if not any(stats[w]["normal_rows"] >= 2 * 3600 for w in fold):
            return f"fold {k}: no well with >=2 h normal"
    dev = [w for fold in folds for w in fold]
    for c in range(1, 10):
        wc = [w for w in dev if c in stats[w]["supported"]]
        if len(wc) >= 2:
            nf = sum(
                1 for fold in folds if any(c in stats[w]["supported"] for w in fold)
            )
            if nf < 2:
                return f"class {c} supported in {nf} dev folds"
    return None


def generate(index: dict[str, Any], seed_hex: str):
    """Draw the test wells and the development folds.

    Returns the manifest rows and a dictionary describing the search.
    """
    stats = well_stats(index)
    wells = sorted(stats)
    unlabeled = [w for w in wells if stats[w]["labelled_rows"] == 0]
    if unlabeled:
        raise SystemExit(f"wells without labels: {unlabeled}")
    rejected_test: dict[str, int] = {}
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        order = hash_order(seed_hex, "test", attempt, wells)
        test = sorted(order[:N_TEST_WELLS])
        why = check_test(test, stats)
        if why is None:
            break
        reason = why.split(":")[0].split(" <")[0]
        rejected_test[reason] = rejected_test.get(reason, 0) + 1
        attempt += 1
    else:
        raise SystemExit("no test split satisfies constraints")
    dev = [w for w in wells if w not in test]
    fold_attempt = 0
    while fold_attempt < MAX_ATTEMPTS:
        order = hash_order(seed_hex, "folds", fold_attempt, dev)
        folds = [sorted(order[i::N_DEV_FOLDS]) for i in range(N_DEV_FOLDS)]
        if check_folds(folds, stats) is None:
            break
        fold_attempt += 1
    else:
        raise SystemExit("no dev folds satisfy constraints")
    fold_of = {w: k for k, fold in enumerate(folds) for w in fold}
    rows = []
    for w in wells:
        s = stats[w]
        rows.append(
            {
                "well_id": w,
                "role": "test" if w in test else "dev",
                "dev_fold": fold_of.get(w, -1),
                "n_files": s["n_files"],
                "labelled_hours": round(s["labelled_rows"] / 3600, 3),
                "normal_hours": round(s["normal_rows"] / 3600, 3),
                "classes_supported": ";".join(str(c) for c in sorted(s["supported"])),
            }
        )
    info = {
        "protocol": PROTOCOL_ID,
        "test_attempt_index": attempt,
        "fold_attempt_index": fold_attempt,
        "test_rejections_by_reason": rejected_test,
        "n_wells": len(wells),
    }
    return rows, info


def write_manifest(rows: list[dict[str, Any]], path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in COLUMNS})


def regenerate(
    index_path: str | Path, signature_hex: str, out_path: str | Path
) -> dict[str, Any]:
    """Run the committed rule on the label index and a beacon signature.

    The seed is the SHA-256 of the signature bytes, which is the drand
    randomness of that round. Writes ``out_path`` and returns the
    description of the draw, including the manifest hash.
    """
    raw = Path(index_path).read_bytes()
    index = json.loads(raw)
    seed_hex = sha256_hex(bytes.fromhex(signature_hex))
    rows, info = generate(index, seed_hex)
    write_manifest(rows, out_path)
    info.update(
        {
            "index_sha256": sha256_hex(raw),
            "seed_sha256_of_signature": seed_hex,
            "manifest_sha256": sha256_hex(Path(out_path).read_bytes()),
        }
    )
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate the well manifest")
    ap.add_argument("--index", required=True)
    ap.add_argument(
        "--beacon-signature-hex",
        required=True,
        help="drand quicknet signature of the committed round",
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    info = regenerate(a.index, a.beacon_signature_hex, a.out)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
