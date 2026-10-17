import numpy as np
import pytest
from held_out_well.detection import alarm_state, well_statistics
from held_out_well.grid import n_decisions
from held_out_well.scoring import bootstrap, report, select_threshold
from hw_helpers import fake_file, make_bench, scenario
from hw_oracles import alarm_state_reference

WELLS = ["WELL-00001"]


def test_alarm_state_matches_reference_on_random_sequences():
    rng = np.random.default_rng(0)
    files = [
        fake_file(
            f"d/{i}/WELL-0000{i}_x.parquet",
            f"WELL-0000{i}",
            3600 + 30 * n,
            [[-1, 0, 3600], [0, 3600, 3600 + 30 * n]],
        )
        for i, n in [(1, 300), (2, 200), (3, 450)]
    ]
    elig = {f["path"]: rng.random(n_decisions(f["rows"])) > 0.15 for f in files}
    b = make_bench(files, elig)
    for _ in range(40):
        score = rng.random(b.n)
        score[rng.random(b.n) < 0.05] = np.nan
        theta = float(rng.uniform(0.2, 0.9))
        state, rise = alarm_state(b, score, theta)
        ref = alarm_state_reference(b.inst, b.elig, score, theta)
        assert np.array_equal(state, ref)
        same = np.r_[False, b.inst[1:] == b.inst[:-1]]
        prev = np.r_[False, ref[:-1]] & same
        assert np.array_equal(rise, ref & ~prev)
    state, _ = alarm_state(b, score, None)
    assert not state.any()


def test_perfect_detector_and_deterministic_bootstrap():
    b = scenario()
    score = np.where(b.y > 0, 0.99, 0.01).astype(float)
    pred = np.where(b.y >= 0, b.y, 0)
    p = report(b, score, pred, WELLS, theta=0.5, boot=False)["point"]
    assert p["macro_f1"] == 1.0
    assert p["balanced_accuracy"] == 1.0
    assert p["episode_coverage"] == 1.0
    assert p["false_alarms_per_normal_day"] == 0.0
    assert p["normal_time_occupation"] == 0.0
    assert p["auroc"] == pytest.approx(1.0)
    assert p["average_precision"] == pytest.approx(1.0)
    assert p["auroc_exact"] == pytest.approx(1.0)
    # an alarm needs two decisions, so the onset is seen at the second
    assert p["onset_detection_300s"] == 1.0
    assert p["onset_delay_median_s"] <= 60
    r1 = report(b, score, pred, WELLS, 0.5, boot=True, B=30, seed=1)
    r2 = report(b, score, pred, WELLS, 0.5, boot=True, B=30, seed=1)
    assert r1["ci95"] == r2["ci95"]
    assert r1["confusion"][2][2] > 0


def test_constant_predictors_and_false_alarm_accounting():
    b = scenario()
    zeros = np.zeros(b.n)
    p = report(b, zeros, np.zeros(b.n, int), WELLS, 0.5, boot=False)["point"]
    assert p["episode_coverage"] == 0.0
    assert p["false_alarms_per_normal_day"] == 0.0
    # always alarm: one rising edge per instance, in its normal period
    ones = np.ones(b.n)
    p = report(b, ones, np.ones(b.n, int), WELLS, 0.5, boot=False)["point"]
    assert p["episode_coverage"] == 1.0
    assert p["normal_time_occupation"] > 0.98
    assert p["false_alarms_per_normal_day"] == pytest.approx(2 / p["normal_days"])
    # an alarm that starts only after the event carries nothing in
    late = np.where(b.r.astype(int) >= 6600, 0.99, 0.0) * (b.inst == 1)
    pred = np.where(late > 0.5, 2, 0)
    p = report(b, late, pred, WELLS, 0.5, boot=False)["point"]
    assert p["episode_coverage"] == 1.0
    assert p["onset_detection_300s"] == 0.0
    assert p["false_alarms_per_normal_day"] == 0.0


def test_ineligible_decisions_reset_state_and_are_excluded():
    a = fake_file(
        "dataset/2/WELL-00001_a.parquet",
        "WELL-00001",
        9000,
        [[-1, 0, 3600], [0, 3600, 5100], [102, 5100, 6600], [2, 6600, 9000]],
    )
    elig = np.ones(n_decisions(9000), bool)
    elig[:10] = False
    b = make_bench([a], {a["path"]: elig})
    score = np.full(b.n, 0.9)
    p = report(b, score, np.zeros(b.n, int), WELLS, 0.5, boot=False)["point"]
    assert p["normal_eligibility"] < 1.0
    assert p["event_eligibility"] == 1.0
    bad = score.copy()
    bad[20] = np.nan
    with pytest.raises(ValueError):
        report(b, bad, np.zeros(b.n, int), WELLS, 0.5, boot=False)
    with pytest.raises(ValueError):
        report(b, score * 2, np.zeros(b.n, int), WELLS, 0.5, boot=False)
    with pytest.raises(ValueError):
        report(b, score, np.full(b.n, 10), WELLS, 0.5, boot=False)


def test_threshold_selection_respects_budgets():
    b = scenario()
    rng = np.random.default_rng(3)
    score = np.where(b.y > 0, rng.beta(8, 2, b.n), rng.beta(2, 8, b.n))
    theta, table = select_threshold(
        b, score, WELLS, fa_per_day_max=5.0, occupation_max=0.05, min_normal_days=0.05
    )
    assert theta is not None and table
    p = report(b, score, None, WELLS, theta, boot=False)["point"]
    assert p["false_alarms_per_normal_day"] <= 5.0
    assert p["normal_time_occupation"] <= 0.05
    assert np.isnan(p["macro_f1"])  # a pure detector has no class output


def test_no_feasible_threshold_returns_none():
    b = scenario()
    theta, table = select_threshold(b, np.zeros(b.n), WELLS, min_normal_days=1e6)
    assert theta is None and table


def test_paired_bootstrap_is_zero_for_identical_predictions():
    b = scenario()
    score = np.where(b.y > 0, 0.99, 0.01).astype(float)
    pred = np.where(b.y >= 0, b.y, 0)
    stats, _, _ = well_statistics(b, score, pred, WELLS, 0.5)
    both = np.vstack([stats, stats])
    ci = bootstrap(both, None, B=20, seed=2, S2=both)
    for key in ("macro_f1", "auroc"):
        assert ci[key][0] == 0.0 and ci[key][1] == 0.0
