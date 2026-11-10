"""The command line on predictions built from the truth."""

import json

import numpy as np
import pytest
from held_out_well import cli
from held_out_well.benchmark import Benchmark


@pytest.fixture(scope="module")
def oracle_file(tmp_path_factory):
    b = Benchmark.load()
    path = tmp_path_factory.mktemp("preds") / "oracle.npz"
    score = np.where(b.y > 0, 0.98, 0.02).astype(np.float32)
    np.savez(path, score=score, pred=b.y.astype(np.int8))
    return path


@pytest.mark.parametrize("split", ["test", "dev", "all"])
def test_oracle_scores_perfectly(oracle_file, tmp_path, capsys, split):
    out = tmp_path / "report.json"
    args = ["score", str(oracle_file), "--split", split]
    args += ["--theta", "0.5", "--boot", "10", "--json", str(out)]
    assert cli.main(args) == 0
    text = capsys.readouterr().out
    assert "Primary metric" in text and "Confusion matrix" in text
    rep = json.loads(out.read_text())
    assert rep["point"]["macro_f1"] == 1.0
    assert rep["point"]["balanced_accuracy"] == 1.0
    assert rep["point"]["auroc"] == pytest.approx(1.0)
    assert rep["point"]["auroc_exact"] == pytest.approx(1.0)
    assert rep["ci95"]["macro_f1"][:2] == [1.0, 1.0]
    assert rep["n_wells"] == {"test": 10, "dev": 30, "all": 40}[split]
    assert np.trace(np.array(rep["confusion"])) == np.sum(rep["confusion"])


def test_detector_file_without_classes(oracle_file, tmp_path, capsys):
    z = np.load(oracle_file)
    det = tmp_path / "detector.npz"
    np.savez(det, score=z["score"])
    assert cli.main(["score", str(det), "--theta", "0.5", "--boot", "0"]) == 0
    text = capsys.readouterr().out
    assert (
        "Primary metric, macro-F1 over the classes present in the pooled truth: -"
        in text
    )


def test_compare_reports_paired_differences(oracle_file, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "select_threshold", lambda *a, **k: (0.5, []))
    z = np.load(oracle_file)
    rng = np.random.default_rng(0)
    wrong = rng.random(z["pred"].shape) < 0.3
    noisy = np.where(wrong, rng.integers(0, 10, z["pred"].shape), z["pred"])
    other = tmp_path / "noisy.npz"
    np.savez(other, score=z["score"], pred=noisy.astype(np.int8))
    assert cli.main(["compare", str(oracle_file), str(other), "--boot", "10"]) == 0
    text = capsys.readouterr().out
    line = next(x for x in text.splitlines() if x.startswith("| macro_f1"))
    assert "+" in line


def test_manifest_command_checks_the_committed_file(capsys):
    assert cli.main(["manifest", "--check"]) == 0
    assert "identical to protocol/manifest.csv: True" in capsys.readouterr().out


def test_missing_development_scores_ask_for_a_theta(oracle_file, tmp_path):
    z = np.load(oracle_file)
    b = Benchmark.load()
    dev = b.well_mask(b.wells_with_role("dev"))[b.well_of_decision]
    score = np.where(dev, np.nan, z["score"]).astype(np.float32)
    path = tmp_path / "test_only.npz"
    np.savez(path, score=score, pred=z["pred"])
    with pytest.raises(SystemExit, match="--theta"):
        cli.main(["score", str(path), "--boot", "0"])
    # with a threshold given, the sealed test wells can be scored
    assert cli.main(["score", str(path), "--theta", "0.5", "--boot", "0"]) == 0
