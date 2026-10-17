"""Deterministic, hash-ordered well split for 3W-HW with machine-checked constraints.

Only label/time metadata (index_labels.json) is read: no sensor values and no model output can
influence the split. Randomness is the SHA-256 of a public drand beacon signature, committed (as a
future round number) in git before the manifest exists. Pure hashing, no numpy RNG, so the output is
identical across library versions.

Constraints (v1):
  K1  |test| = 10 wells.
  K2  every event class 1..9 has >=1 supported test well and >=1 supported dev well; when a class has
      >=4 wells overall, dev has >=2 of them.
  K3  test labelled-normal time >= 24 h, with >=2 test wells holding >=6 h of normal each.
  K4  test labelled rows within 15-35 % of all labelled rows.
  K5  dev wells are partitioned into 5 folds of 6; every fold contains >=1 well with >=2 h normal; every
      class with >=2 dev wells is supported in >=2 different dev folds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

try:
    from .labels import class_support
except ImportError:  # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from labels import class_support  # type: ignore

PROTOCOL_ID = "3W-HW-v1"
N_TEST = 10
N_FOLDS = 5
MAX_ATTEMPTS = 200000


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hash_order(seed_hex: str, tag: str, attempt: int, items):
    """Sort items by SHA256(seed|protocol|tag|attempt|item): a stable uniform permutation."""
    def key(x):
        return hashlib.sha256(f"{seed_hex}|{PROTOCOL_ID}|{tag}|{attempt}|{x}".encode()).hexdigest()
    return sorted(items, key=key)


def well_stats(index):
    """index: parsed index_labels.json -> dict well -> stats."""
    wells = {}
    for f in index["files"]:
        w = wells.setdefault(f["well"], {"n_files": 0, "labelled_rows": 0, "normal_rows": 0, "class_rows": {}, "supported": set()})
        w["n_files"] += 1
        rows, supported = class_support(f["class_runs"])
        for ev, n in rows.items():
            w["class_rows"][ev] = w["class_rows"].get(ev, 0) + n
        w["labelled_rows"] += sum(rows.values())
        w["normal_rows"] += rows.get(0, 0)
        w["supported"].update(supported)
    return wells


def check_test(test, stats):
    wells = sorted(stats)
    dev = [w for w in wells if w not in test]
    classes = range(1, 10)
    for c in classes:
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
    if sum(1 for w in test if stats[w]["normal_rows"] >= 6 * 3600) < 2:
        return "fewer than 2 test wells with >=6 h normal"
    share = sum(stats[w]["labelled_rows"] for w in test) / sum(stats[w]["labelled_rows"] for w in wells)
    if not 0.15 <= share <= 0.35:
        return f"test row share {share:.3f}"
    return None


def check_folds(folds, stats):
    for k, fold in enumerate(folds):
        if not any(stats[w]["normal_rows"] >= 2 * 3600 for w in fold):
            return f"fold {k}: no well with >=2 h normal"
    dev = [w for fold in folds for w in fold]
    for c in range(1, 10):
        wc = [w for w in dev if c in stats[w]["supported"]]
        if len(wc) >= 2:
            nf = sum(1 for fold in folds if any(c in stats[w]["supported"] for w in fold))
            if nf < 2:
                return f"class {c} supported in {nf} dev folds"
    return None


def generate(index, seed_hex):
    stats = well_stats(index)
    wells = sorted(stats)
    unlabeled = [w for w in wells if stats[w]["labelled_rows"] == 0]
    if unlabeled:
        raise SystemExit(f"wells without labels: {unlabeled}")
    rejected_test = {}
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        order = hash_order(seed_hex, "test", attempt, wells)
        test = sorted(order[:N_TEST])
        why = check_test(test, stats)
        if why is None:
            break
        rejected_test[why.split(":")[0].split(" <")[0]] = rejected_test.get(why.split(":")[0].split(" <")[0], 0) + 1
        attempt += 1
    else:
        raise SystemExit("no test split satisfies constraints")
    dev = [w for w in wells if w not in test]
    fold_attempt = 0
    while fold_attempt < MAX_ATTEMPTS:
        order = hash_order(seed_hex, "folds", fold_attempt, dev)
        folds = [sorted(order[i::N_FOLDS]) for i in range(N_FOLDS)]
        if check_folds(folds, stats) is None:
            break
        fold_attempt += 1
    else:
        raise SystemExit("no dev folds satisfy constraints")
    rows = []
    fold_of = {w: k for k, fold in enumerate(folds) for w in fold}
    for w in wells:
        s = stats[w]
        rows.append({
            "well_id": w, "role": "test" if w in test else "dev", "dev_fold": fold_of.get(w, -1), "n_files": s["n_files"],
            "labelled_hours": round(s["labelled_rows"] / 3600, 3), "normal_hours": round(s["normal_rows"] / 3600, 3),
            "classes_supported": ";".join(str(c) for c in sorted(s["supported"])),
        })
    info = {"protocol": PROTOCOL_ID, "test_attempt_index": attempt, "fold_attempt_index": fold_attempt,
            "test_rejections_by_reason": rejected_test, "n_wells": len(wells)}
    return rows, info


def write_manifest(rows, path):
    cols = ["well_id", "role", "dev_fold", "n_files", "labelled_hours", "normal_hours", "classes_supported"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--beacon-signature-hex", required=True, help="drand quicknet signature of the committed round")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = Path(a.index).read_bytes()
    index = json.loads(raw)
    seed_hex = sha256_hex(bytes.fromhex(a.beacon_signature_hex))  # drand randomness = sha256(signature)
    rows, info = generate(index, seed_hex)
    write_manifest(rows, a.out)
    info.update({"index_sha256": sha256_hex(raw), "seed_sha256_of_signature": seed_hex,
                 "manifest_sha256": sha256_hex(Path(a.out).read_bytes())})
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
