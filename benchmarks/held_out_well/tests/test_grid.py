import numpy as np
from held_out_well.grid import (
    array_digest,
    decision_rows,
    eligibility_from_any_process,
    n_decisions,
)
from hw_helpers import fake_file, make_bench


def test_grid_counts_and_rows():
    assert n_decisions(3600) == 0
    assert n_decisions(3601) == 1
    assert n_decisions(3630) == 1
    assert n_decisions(3631) == 2
    assert list(decision_rows(3700)) == [3600, 3630, 3660, 3690]


def test_eligibility_window():
    rows = 6000
    any_proc = np.ones(rows, bool)
    assert eligibility_from_any_process(any_proc).all()
    any_proc[3550 : 3600 + 95] = False
    elig = eligibility_from_any_process(any_proc)
    for k, r in enumerate(decision_rows(rows)):
        missing = (~any_proc[r - 899 : r + 1]).sum()
        assert elig[k] == (missing <= 90), (r, missing)


def test_decision_table_labels():
    f = fake_file(
        "a/WELL-00001_1.parquet",
        "WELL-00001",
        8000,
        [[-1, 0, 3600], [0, 3600, 5000], [102, 5000, 6000], [2, 6000, 8000]],
    )
    b = make_bench([f])
    assert b.n == n_decisions(8000)
    r = b.r.astype(int)
    assert b.y[r == 3600][0] == 0
    assert b.y[r == 4980][0] == 0
    assert b.y[r == 5010][0] == 2
    assert b.y[r == 7980][0] == 2
    assert len(b.ep_start) == 1
    assert b.ep_start[0] == 5000 and b.ep_end[0] == 8000


def test_array_digest_depends_on_content_only():
    a = np.arange(10, dtype=np.int32)
    assert array_digest(a) == array_digest(a.copy())
    assert array_digest(a) != array_digest(a + 1)
