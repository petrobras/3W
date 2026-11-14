"""The toolkit-based metrics against independent formulas."""

import numpy as np
import pytest
from held_out_well.constants import N_CLASSES, NBINS
from held_out_well.detection import DIM, LAYOUT, logit_bin
from held_out_well.scoring import (
    bootstrap,
    classification_metrics,
    metrics_from_sums,
    resample_counts,
    threshold_free_metrics,
)
from hw_oracles import metrics_numpy
from ThreeWToolkit.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

COMPARED = [
    "macro_f1",
    "macro_f1_event_classes",
    "balanced_accuracy",
    "accuracy",
    "auroc",
    "average_precision",
] + [f"f1_c{c}" for c in range(N_CLASSES)]


def random_statistics(rng, n_rows):
    """Random per-well statistics with absent and spurious classes."""
    S = np.zeros((n_rows, DIM))
    a, b = LAYOUT["conf"]
    for i in range(n_rows):
        conf = rng.poisson(rng.uniform(0, 30, (N_CLASSES, N_CLASSES)))
        conf[rng.random(N_CLASSES) < 0.3] = 0  # truth class absent here
        conf = conf * (rng.random((N_CLASSES, N_CLASSES)) < 0.6)
        S[i, a:b] = conf.ravel()
        for key in ("hist_pos", "hist_neg"):
            lo, hi = LAYOUT[key]
            bins = rng.integers(0, NBINS, 40)
            S[i, lo:hi] = np.bincount(bins, minlength=NBINS) * rng.integers(1, 9)
    return S


@pytest.mark.parametrize("classes", [None, [0, 1, 2, 4, 5, 7, 8, 9]])
def test_metrics_equal_numpy_formulas(classes):
    S = random_statistics(np.random.default_rng(0), 7)
    for stats in (S, S.sum(0)):
        got = metrics_from_sums(stats, classes)
        want = metrics_numpy(stats, classes)
        for key in COMPARED:
            np.testing.assert_allclose(
                got[key],
                want[key],
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
                err_msg=key,
            )


def test_metrics_are_undefined_without_data():
    out = metrics_from_sums(np.zeros((2, DIM)))
    for key in ("macro_f1", "balanced_accuracy", "auroc", "average_precision"):
        assert np.isnan(out[key]).all()
    nothing = threshold_free_metrics(np.zeros(NBINS), np.ones(NBINS))
    assert all(np.isnan(v) for v in nothing)
    auc, ap = threshold_free_metrics(np.ones(NBINS), np.zeros(NBINS))
    assert np.isnan(auc) and ap == pytest.approx(1.0)


def test_primary_metric_uses_classes_present_in_truth():
    conf = np.zeros((N_CLASSES, N_CLASSES))
    conf[0, 0], conf[0, 3] = 90, 10  # normal; some called class 3
    conf[2, 2], conf[2, 0] = 8, 2
    got = classification_metrics(conf)
    # class 3 is predicted but absent from the truth: it is not averaged
    f1_0 = 2 * 90 / (2 * 90 + 10 + 2)
    f1_2 = 2 * 8 / (2 * 8 + 0 + 2)
    assert got["macro_f1"] == pytest.approx((f1_0 + f1_2) / 2)
    assert got["macro_f1_event_classes"] == pytest.approx(f1_2)
    assert got["balanced_accuracy"] == pytest.approx((0.9 + 0.8) / 2)
    assert np.isnan(got["f1"][3]) and not np.isnan(got["f1"][0])
    sub = classification_metrics(conf, classes=[0])
    assert sub["macro_f1"] == pytest.approx(f1_0)
    assert sub["balanced_accuracy"] == pytest.approx(0.9)


def test_toolkit_calls_on_cells_equal_calls_on_decisions():
    """Weights on confusion cells equal the metric of the decisions."""
    rng = np.random.default_rng(1)
    truth = rng.integers(0, N_CLASSES, 5000)
    noise = rng.integers(0, N_CLASSES, 5000)
    pred = np.where(rng.random(5000) < 0.5, truth, noise)
    conf = np.zeros((N_CLASSES, N_CLASSES))
    np.add.at(conf, (truth, pred), 1)
    got = classification_metrics(conf)
    present = sorted(set(truth.tolist()))
    want = f1_score(truth, pred, labels=present, average="macro", zero_division=0)
    assert got["macro_f1"] == pytest.approx(want)
    assert got["balanced_accuracy"] == pytest.approx(
        balanced_accuracy_score(truth, pred)
    )


def test_bootstrap_matches_vectorised_formulas():
    S = random_statistics(np.random.default_rng(5), 6)
    B, seed = 25, 7
    got = bootstrap(S, None, B=B, seed=seed, chunk=10)
    want = metrics_numpy(resample_counts(S.shape[0], B, seed) @ S)
    for key in ("macro_f1", "balanced_accuracy", "auroc", "average_precision"):
        v = want[key]
        assert got[key][0] == pytest.approx(np.nanpercentile(v, 2.5), abs=1e-10)
        assert got[key][1] == pytest.approx(np.nanpercentile(v, 97.5), abs=1e-10)
        assert got[key][2] == int(np.isfinite(v).sum())


def test_bootstrap_rng_stream_is_frozen():
    c = resample_counts(4, 3, 611)
    assert c.shape == (3, 4) and (c.sum(1) == 4).all()
    first = np.random.default_rng(611).integers(0, 4, 4)
    assert np.array_equal(c[0], np.bincount(first, minlength=4))


def test_well_bootstrap_on_decisions_equals_sufficient_statistics():
    """Resampling wells of raw decisions gives the same replicates."""
    rng = np.random.default_rng(11)
    n_wells, per = 4, 600
    well = np.repeat(np.arange(n_wells), per)
    truth = rng.integers(0, 4, well.size)
    noise = rng.integers(0, 4, well.size)
    pred = np.where(rng.random(well.size) < 0.6, truth, noise)
    # distinct scores lie far apart, so no two of them share a bin
    levels = np.linspace(-9, 9, 40)
    gap = levels[1] - levels[0]
    z = rng.choice(levels, well.size) + 0.5 * gap * (truth > 0)
    score = 1.0 / (1.0 + np.exp(-z))
    bins = logit_bin(score)
    S = np.zeros((n_wells, DIM))
    a, b = LAYOUT["conf"]
    for w in range(n_wells):
        m = well == w
        conf = np.zeros((N_CLASSES, N_CLASSES))
        np.add.at(conf, (truth[m], pred[m]), 1)
        S[w, a:b] = conf.ravel()
        for key, sel in (("hist_pos", truth > 0), ("hist_neg", truth == 0)):
            lo, hi = LAYOUT[key]
            S[w, lo:hi] = np.bincount(bins[m & sel], minlength=NBINS)
    counts = resample_counts(n_wells, 50, 3)
    fast = metrics_from_sums(counts @ S)
    for r in range(50):
        times = counts[r][well].astype(int)  # how often each decision is drawn
        idx = np.repeat(np.arange(well.size), times)
        present = sorted(set(truth[idx].tolist()))
        is_event = (truth[idx] > 0).astype(int)
        f1 = f1_score(
            truth[idx], pred[idx], labels=present, average="macro", zero_division=0
        )
        assert fast["macro_f1"][r] == pytest.approx(f1, abs=1e-12)
        auc = roc_auc_score(is_event, score[idx])
        assert fast["auroc"][r] == pytest.approx(auc, abs=1e-12)
        ap = average_precision_score(is_event, score[idx])
        assert fast["average_precision"][r] == pytest.approx(ap, abs=1e-12)
