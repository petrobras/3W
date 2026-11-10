"""Command line: score a prediction, compare two, check the protocol.

    python -m held_out_well score preds.npz --split test
    python -m held_out_well compare a.npz b.npz --split test
    python -m held_out_well manifest --check
    python -m held_out_well rebuild-index --dataset-root DIR --out DIR

A prediction file is an ``.npz`` with ``score`` (float, one P(event) in
[0, 1] per decision of ``protocol/decisions.npz``) and, for a
classifier, ``pred`` (integer class 0..9 per decision).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark import PROTOCOL_DIR, Benchmark
from .constants import BOOT_B, CLASS_NAMES, N_CLASSES
from .detection import well_statistics
from .scoring import (
    bootstrap,
    jsonable,
    metrics_from_sums,
    report,
    select_threshold,
)

PRIMARY = "macro_f1"
PAIRED_KEYS = [
    "macro_f1",
    "balanced_accuracy",
    "auroc",
    "average_precision",
    "episode_coverage",
    "onset_detection_300s",
    "false_alarms_per_normal_day",
    "normal_time_occupation",
]


def load_predictions(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read ``score`` and, if present, ``pred`` from an .npz file."""
    z = np.load(path)
    score = z["score"].astype(np.float64)
    pred = z["pred"].astype(np.int64) if "pred" in z.files else None
    return score, pred


def split_wells(bench: Benchmark, split: str) -> tuple[list[str], list[int] | None]:
    """Wells of a split and the class set its macro averages run over.

    The sealed test wells and the pooled set use every class present in
    the truth. The development wells use the classes that occur in two
    or more of them, so that a class can be trained and scored on
    different wells.
    """
    if split == "test":
        return bench.wells_with_role("test"), None
    if split == "dev":
        return bench.wells_with_role("dev"), bench.dev_classes()
    return list(bench.wells), None


def _fmt(rep: dict[str, Any], key: str, pct: bool = False, digits: int = 3) -> str:
    v = rep["point"].get(key)
    if v is None or not np.isfinite(v):
        return "-"
    scale = 100.0 if pct else 1.0
    unit = "%" if pct else ""
    d = 1 if pct else digits
    text = f"{v * scale:.{d}f}{unit}"
    ci = rep.get("ci95", {}).get(key)
    if ci:
        text += f" [{ci[0] * scale:.{d}f}, {ci[1] * scale:.{d}f}]"
    return text


def format_report(name: str, split: str, rep: dict[str, Any]) -> str:
    """Markdown summary: primary and complementary metrics, tables."""
    pt = rep["point"]
    primary = (
        "Primary metric, macro-F1 over the classes present in the pooled "
        f"truth: {_fmt(rep, PRIMARY)}"
    )
    lines = [
        f"### {name} on the {split} wells ({rep['n_wells']} wells)",
        "",
        primary,
        "",
        "| complementary metric | value [95 % well-bootstrap interval] |",
        "|---|---|",
    ]
    rows = [
        ("balanced accuracy", _fmt(rep, "balanced_accuracy")),
        ("macro-F1, event classes only", _fmt(rep, "macro_f1_event_classes")),
        ("accuracy", _fmt(rep, "accuracy")),
        (
            "macro-F1 per well, mean (sd)",
            "-"
            if "per_well_macro_f1_mean" not in pt
            else f"{pt['per_well_macro_f1_mean']:.3f} "
            f"({pt['per_well_macro_f1_sd']:.3f})",
        ),
        ("event AUROC", _fmt(rep, "auroc")),
        ("event average precision", _fmt(rep, "average_precision")),
        ("episode coverage", _fmt(rep, "episode_coverage", pct=True)),
        ("onset detected within 300 s", _fmt(rep, "onset_detection_300s", True)),
        (
            "false alarms per normal day",
            _fmt(rep, "false_alarms_per_normal_day", digits=2),
        ),
        ("normal time in alarm", _fmt(rep, "normal_time_occupation", pct=True)),
    ]
    lines += [f"| {k} | {v} |" for k, v in rows]
    facts = (
        f"Alarm threshold {rep['theta']}. Event AUROC / AP on unbinned "
        f"scores: {pt['auroc_exact']:.4f} / {pt['average_precision_exact']:.4f}. "
        f"Decisions: {pt['event_decisions']:.0f} event, "
        f"{pt['normal_decisions']:.0f} normal; "
        f"{pt['normal_days']:.1f} normal days; "
        f"{pt['episodes_total']:.0f} episodes "
        f"({pt['episodes_unobservable']:.0f} unobservable)."
    )
    lines += ["", facts]
    conf = np.array(rep["confusion"])
    if conf.sum() > 0:
        classes = " | ".join(str(c) for c in range(N_CLASSES))
        lines += [
            "",
            "Confusion matrix (rows: truth, columns: prediction, eligible decisions):",
            "",
            f"| truth \\ pred | {classes} | support | F1 |",
            "|" + "---|" * (N_CLASSES + 3),
        ]
        f1 = rep["point"]
        for t in range(N_CLASSES):
            if conf[t].sum() == 0:
                continue
            cells = " | ".join(str(int(v)) for v in conf[t])
            lines.append(
                f"| {t} {CLASS_NAMES[t]} | {cells} | {int(conf[t].sum())} | "
                f"{_fmt({'point': f1}, f'f1_c{t}')} |"
            )
    return "\n".join(lines)


def _theta(bench: Benchmark, score: np.ndarray, given: float | None) -> float | None:
    """The alarm threshold: given, or chosen on the development part."""
    if given is not None:
        return given
    dev = bench.wells_with_role("dev")
    on_dev = bench.well_mask(dev)[bench.well_of_decision] & bench.elig
    if not np.isfinite(score[on_dev]).all():
        raise SystemExit(
            "the file has no development scores to choose the alarm "
            "threshold from: add them or pass --theta"
        )
    theta, _ = select_threshold(bench, score, dev)
    if theta is None:
        print(
            "warning: no threshold meets the false-alarm budgets on the "
            "development wells; the alarm never fires",
            file=sys.stderr,
        )
    return theta


def cmd_score(a: argparse.Namespace) -> int:
    bench = Benchmark.load()
    score, pred = load_predictions(a.predictions)
    wells, classes = split_wells(bench, a.split)
    theta = _theta(bench, score, a.theta)
    rep = report(
        bench, score, pred, wells, theta, classes, boot=a.boot > 0, B=max(a.boot, 1)
    )
    print(format_report(Path(a.predictions).name, a.split, rep))
    if a.json:
        Path(a.json).write_text(json.dumps(jsonable(rep), indent=1))
    return 0


def cmd_compare(a: argparse.Namespace) -> int:
    bench = Benchmark.load()
    wells, classes = split_wells(bench, a.split)
    stats = []
    for path in (a.a, a.b):
        score, pred = load_predictions(path)
        theta = _theta(bench, score, None)
        S, _, _ = well_statistics(bench, score, pred, wells, theta)
        stats.append(S)
    ci = bootstrap(stats[0], classes, B=a.boot, S2=stats[1])
    print(f"### {Path(a.a).name} minus {Path(a.b).name}, {a.split} wells")
    print()
    print("| metric | difference [95 % well-bootstrap interval] |")
    print("|---|---|")
    ma = metrics_from_sums(stats[0].sum(0), classes)
    mb = metrics_from_sums(stats[1].sum(0), classes)
    for k in PAIRED_KEYS:
        d = float(ma[k]) - float(mb[k])
        if not np.isfinite(d) or k not in ci:
            continue
        print(f"| {k} | {d:+.4f} [{ci[k][0]:+.4f}, {ci[k][1]:+.4f}] |")
    return 0


def cmd_manifest(a: argparse.Namespace) -> int:
    from .manifest import regenerate

    sig = json.loads((PROTOCOL_DIR / "drand_round.json").read_text())["signature"]
    out = Path(a.out) if a.out else Path(tempfile.mkdtemp()) / "manifest.csv"
    info = regenerate(PROTOCOL_DIR / "index_labels.json", sig, out)
    print(json.dumps(info, indent=1))
    if a.check:
        same = out.read_bytes() == (PROTOCOL_DIR / "manifest.csv").read_bytes()
        print(f"identical to protocol/manifest.csv: {same}")
        return 0 if same else 1
    return 0


def cmd_rebuild(a: argparse.Namespace) -> int:
    from .loader import build_index, compare_with_committed, decision_table

    def progress(i: int, n: int) -> None:
        if i % 100 == 0 or i == n:
            print(f"  {i}/{n} instances", file=sys.stderr, flush=True)

    index, elig = build_index(a.dataset_root, progress)
    table = decision_table(index, elig)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    result = compare_with_committed(index, table)
    (out / "rebuild_report.json").write_text(json.dumps(jsonable(result), indent=1))
    if a.save:
        (out / "index_labels.json").write_text(json.dumps(index, separators=(",", ":")))
        np.savez_compressed(out / "decisions.npz", **table)
    print(json.dumps(jsonable(result), indent=1))
    return 0 if result["all_equal"] else 1


def cmd_baselines(a: argparse.Namespace) -> int:
    from .baselines.run import MODELS, run

    run(
        a.dataset_root,
        a.out,
        models=a.models or MODELS,
        smoke=a.smoke,
        workers=a.workers,
        boot=a.boot,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m held_out_well",
        description="Held-out-well benchmark for the 3W dataset.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="score one prediction file")
    s.add_argument("predictions", help=".npz with score (and pred)")
    s.add_argument("--split", choices=["test", "dev", "all"], default="test")
    s.add_argument(
        "--theta", type=float, help="alarm threshold (default: chosen on dev)"
    )
    s.add_argument("--boot", type=int, default=BOOT_B, help="resamples (0: none)")
    s.add_argument("--json", help="write the full report here")
    s.set_defaults(func=cmd_score)

    c = sub.add_parser("compare", help="paired difference of two predictions")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--split", choices=["test", "dev", "all"], default="test")
    c.add_argument("--boot", type=int, default=BOOT_B)
    c.set_defaults(func=cmd_compare)

    m = sub.add_parser("manifest", help="regenerate the well split")
    m.add_argument("--out", help="where to write manifest.csv")
    m.add_argument(
        "--check", action="store_true", help="compare with the committed file"
    )
    m.set_defaults(func=cmd_manifest)

    r = sub.add_parser("rebuild-index", help="rebuild the label index with the toolkit")
    r.add_argument("--dataset-root", required=True, help="folder with 0/ ... 9/")
    r.add_argument("--out", required=True, help="folder for rebuild_report.json")
    r.add_argument("--save", action="store_true", help="also write the rebuilt tables")
    r.set_defaults(func=cmd_rebuild)
    b = sub.add_parser("baselines", help="run the reference baselines")
    b.add_argument("--dataset-root", required=True, help="folder with 0/ ... 9/")
    b.add_argument("--out", required=True, help="folder for predictions and results")
    b.add_argument(
        "--models", nargs="*", choices=["simple", "volatility", "rf", "lgbm"]
    )
    b.add_argument(
        "--smoke", action="store_true", help="development wells only, tiny models"
    )
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--boot", type=int, default=BOOT_B)
    b.set_defaults(func=cmd_baselines)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
