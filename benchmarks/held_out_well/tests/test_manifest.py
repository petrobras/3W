"""The well split: rules, commitments and byte-for-byte regeneration."""

import hashlib
import json
import re
import subprocess
import sys

import pytest
from held_out_well.benchmark import PROTOCOL_DIR
from held_out_well.manifest import (
    check_folds,
    check_test,
    regenerate,
    well_stats,
)

FROZEN = PROTOCOL_DIR / "frozen"
SIGNATURE = json.loads((PROTOCOL_DIR / "drand_round.json").read_text())["signature"]
INDEX = PROTOCOL_DIR / "index_labels.json"
COMMITTED = PROTOCOL_DIR / "manifest.csv"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_regeneration_is_byte_identical(tmp_path):
    out = tmp_path / "manifest.csv"
    info = regenerate(INDEX, SIGNATURE, out)
    assert out.read_bytes() == COMMITTED.read_bytes()
    recorded = json.loads((PROTOCOL_DIR / "manifest_info.json").read_text())
    for key in ("manifest_sha256", "index_sha256", "test_attempt_index"):
        assert info[key] == recorded[key]
    assert info["manifest_sha256"] in (PROTOCOL_DIR / "COMMIT-B.md").read_text()


def test_frozen_code_regenerates_the_same_bytes(tmp_path):
    """The files committed before the draw give the same manifest."""
    (tmp_path / "manifest.py").write_bytes((FROZEN / "manifest.py.txt").read_bytes())
    (tmp_path / "labels.py").write_bytes((FROZEN / "labels.py.txt").read_bytes())
    out = tmp_path / "frozen_manifest.csv"
    subprocess.run(
        [
            sys.executable,
            str(tmp_path / "manifest.py"),
            "--index",
            str(INDEX),
            "--beacon-signature-hex",
            SIGNATURE,
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    assert out.read_bytes() == COMMITTED.read_bytes()


def test_commit_a_hashes_match_the_files_in_this_folder():
    text = (PROTOCOL_DIR / "COMMIT-A.md").read_text()
    listed = dict(re.findall(r"^- (\S+)\s+([0-9a-f]{64})$", text, re.MULTILINE))
    where = {
        "protocol/index_labels.json": INDEX,
        "src/w3/manifest.py": FROZEN / "manifest.py.txt",
        "src/w3/labels.py": FROZEN / "labels.py.txt",
        "protocol/PROTOCOL.md": FROZEN / "PROTOCOL.md",
    }
    assert set(listed) == set(where)
    for name, path in where.items():
        assert sha256(path) == listed[name], name


def test_committed_manifest_satisfies_the_split_rules():
    index = json.loads(INDEX.read_text())
    stats = well_stats(index)
    rows = [line.split(",") for line in COMMITTED.read_text().splitlines()[1:]]
    test = sorted(r[0] for r in rows if r[1] == "test")
    folds = [sorted(r[0] for r in rows if r[2] == str(k)) for k in range(5)]
    assert len(rows) == 40 and len(test) == 10
    assert [len(f) for f in folds] == [6] * 5
    assert check_test(test, stats) is None
    assert check_folds(folds, stats) is None
    for name in test:
        assert name not in {w for fold in folds for w in fold}


def test_rules_reject_bad_splits():
    index = json.loads(INDEX.read_text())
    stats = well_stats(index)
    wells = sorted(stats)
    assert check_test(wells[:10], stats) is not None
    fake = {
        "A": {"normal_rows": 100, "supported": set()},
        "B": {"normal_rows": 99999, "supported": set()},
    }
    assert check_folds([["A"], ["B"]], fake) == "fold 0: no well with >=2 h normal"
    assert check_folds([["B"], ["B"]], fake) is None


@pytest.mark.parametrize("which", ["test", "dev"])
def test_every_event_class_is_in_test_and_dev(which):
    """Rule K2: classes 1..9 occur on both sides of the split."""
    index = json.loads(INDEX.read_text())
    stats = well_stats(index)
    role = {
        line.split(",")[0]: line.split(",")[1]
        for line in COMMITTED.read_text().splitlines()[1:]
    }
    for c in range(1, 10):
        wells = [w for w in stats if c in stats[w]["supported"]]
        assert any(role[w] == which for w in wells), (c, which)
