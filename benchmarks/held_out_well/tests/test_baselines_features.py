"""Causal features and the training-free detector of the baselines."""

import numpy as np
from held_out_well.baselines import features as F
from held_out_well.baselines import recipes as R
from held_out_well.grid import decision_rows
from hw_helpers import scenario
from hw_oracles import reference_channel_features


def synth(rows=9000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(rows)
    x = 1e7 + 3e5 * np.sin(t / 200.0) + 2e4 * rng.normal(size=rows) + 50.0 * t / 1000
    x[2000:2100] = np.nan
    x[6000:6003] = np.nan
    return x


def test_channel_features_match_the_slice_implementation():
    x = synth()
    r = decision_rows(len(x))
    got = F.channel_features(x, r).astype(np.float64)
    ref = reference_channel_features(x, r)
    for j, name in enumerate(F.PER_CHANNEL):
        a, b = got[:, j], ref[:, j]
        both = np.isnan(a) & np.isnan(b)
        assert np.array_equal(np.isnan(a), np.isnan(b)), name
        scale = max(1.0, float(np.nanmax(np.abs(b[~both]))))
        ok = np.allclose(a[~both], b[~both], rtol=2e-4, atol=1e-5 * scale)
        assert ok, (name, np.nanmax(np.abs(a - b)))


def test_future_rows_do_not_change_past_features():
    rng = np.random.default_rng(1)
    cont = np.stack([synth(9000, s) for s in range(3)] + [np.full(9000, np.nan)], 1)
    state = np.zeros((9000, 2))
    state[4000:, 0] = 1
    r = decision_rows(9000)
    columns = range(cont.shape[1])
    base = np.concatenate([F.channel_features(cont[:, j], r) for j in columns], 1)
    cut = r[40]
    cont2 = cont.copy()
    cont2[cut + 1 :] = rng.normal(size=cont2[cut + 1 :].shape) * 1e9
    alt = np.concatenate([F.channel_features(cont2[:, j], r) for j in columns], 1)
    k = np.flatnonzero(r <= cut)
    assert np.array_equal(np.nan_to_num(base[k], nan=-7), np.nan_to_num(alt[k], nan=-7))
    later = np.where(np.arange(9000) <= cut, state[:, 0], 5.0)
    s1, s2 = F.state_features(state[:, 0], r), F.state_features(later, r)
    assert np.array_equal(s1[k], s2[k])


def test_all_nan_channel_and_shapes():
    r = decision_rows(5000)
    out = F.channel_features(np.full(5000, np.nan), r)
    finite = F.PER_CHANNEL.index("finite_w3")
    assert out.shape == (len(r), len(F.PER_CHANNEL)) and (out[:, finite] == 0).all()
    cont = np.full((5000, len(F.CONT)), np.nan)
    state = np.full((5000, len(F.STATE)), np.nan)
    full = F.instance_features(cont, state)
    assert full.shape == (len(r), len(F.feature_names())) == (len(r), 333)


def test_simple_detector_matches_naive_and_detects_a_shift():
    rng = np.random.default_rng(0)
    rows = 9000
    cont = np.full((rows, 18), np.nan)
    cont[:, 0] = 1e7 + rng.normal(size=rows) * 1e3
    cont[5000:, 0] += 8e3
    cont[:, 3] = 50 + rng.normal(size=rows) * 0.1
    cont[7000:, 3] = 50 + rng.normal(size=rows - 7000) * 0.6
    z = F.simple_detector_score(cont)
    r = decision_rows(rows)

    def naive(k):
        rr = r[k]
        best = -np.inf
        for j in range(16):
            xs = cont[rr - 299 : rr + 1, j]
            xr = cont[rr - 3599 : rr - 299, j]
            if np.isfinite(xs).mean() < 0.8 or np.isfinite(xr).mean() < 0.8:
                continue
            xs, xr = xs[np.isfinite(xs)], xr[np.isfinite(xr)]
            mr, sr, ss = xr.mean(), xr.std(), xs.std()
            d = sr + 1e-3 * abs(mr) + 1e-12
            best = max(best, abs(xs.mean() - mr) / d, 2 * max(0, ss / d - 1))
        return best if np.isfinite(best) else np.nan

    for k in (0, 10, 40, 60, 100, 120, 150):
        assert np.isclose(z[k], naive(k), rtol=1e-6, equal_nan=True)
    shifted = np.nanmean(z[(r > 5100) & (r < 5500)])
    assert shifted > 10 * np.nanmean(z[(r > 4000) & (r < 4800)])
    assert np.isnan(F.simple_detector_score(np.full((5000, 18), np.nan))).all()
    u = F.simple_score_to_unit(np.array([0.0, 3.0, np.nan, 1e9]))
    assert u[0] == 0.0 and u[1] == 0.5 and np.isnan(u[2]) and u[3] < 1.0 + 1e-6


def test_volatility_detector_scores_variable_channels_higher():
    names = F.feature_names()
    cols = [i for i, n in enumerate(names) if n.endswith("|cv_w3")]
    assert len(cols) == 18
    rng = np.random.default_rng(0)
    X = np.full((400, len(names)), np.nan)
    X[:, cols] = rng.uniform(0.1, 1.0, (400, 18))
    y = np.zeros(400, dtype=int)
    det = R.VolatilityDetector(names).fit(X, y)
    calm = X[:5].copy()
    wild = X[:5].copy()
    wild[:, cols[3]] = 50.0
    s_calm = det.predict_proba(calm)[:, 1]
    s_wild = det.predict_proba(wild)[:, 1]
    assert (s_wild > s_calm).all() and (s_wild < 1.0).all()
    assert np.allclose(det.predict_proba(calm).sum(1), 1.0)


def test_fold_plan_is_grouped_by_well():
    from held_out_well.benchmark import Benchmark

    plan = R.fold_plan(Benchmark.load())
    assert [name for name, _, _ in plan] == [f"dev{k}" for k in range(5)] + ["test"]
    for name, train, held in plan:
        assert not set(train) & set(held), name
    test_wells = set(plan[-1][2])
    for name, train, held in plan[:-1]:
        assert not test_wells & (set(train) | set(held))
    assert len(plan[-1][1]) == 30 and len(plan[-1][2]) == 10


def test_training_sample_respects_caps_and_wells():
    b = scenario()
    have = np.ones(b.n, bool)
    rng = np.random.default_rng(0)
    idx = R.train_sample(b, have, ["WELL-00001"], 20, rng, every=1)
    assert len(idx) <= 20 * 10 and (b.elig[idx]).all()
    assert (np.bincount(b.y[idx], minlength=10) <= 20).all()
    assert np.array_equal(idx, np.sort(idx))
