import numpy as np
from held_out_well.labels import (
    class_support,
    episodes_from_runs,
    event_class,
    runs_from_labels,
)


def test_event_class_merges_transients():
    got = event_class(np.array([-1, 0, 3, 103, 109]))
    assert list(got) == [-1, 0, 3, 3, 9]


def test_episodes_merge_transient_and_steady():
    runs = [
        [-1, 0, 3600],
        [0, 3600, 5000],
        [102, 5000, 6000],
        [2, 6000, 6500],
        [0, 6500, 7000],
        [4, 7000, 8000],
    ]
    assert episodes_from_runs(runs) == [
        {"event": 2, "start": 5000, "end": 6500, "known_onset": True},
        {"event": 4, "start": 7000, "end": 8000, "known_onset": True},
    ]
    rows, supported = class_support(runs)
    assert supported == [2, 4] and rows[2] == 1500


def test_unknown_onset_after_na_or_at_file_start():
    eps = episodes_from_runs([[-1, 0, 3600], [4, 3600, 9000]])
    assert eps[0]["known_onset"] is False
    eps = episodes_from_runs([[4, 0, 900], [0, 900, 1500]])
    assert eps[0]["known_onset"] is False


def test_short_episode_is_not_support():
    runs = [[0, 0, 4000], [5, 4000, 4200], [0, 4200, 6000]]
    rows, supported = class_support(runs)
    assert rows[5] == 200 and supported == []


def test_runs_from_labels_roundtrip():
    v = np.array([-1] * 3 + [0] * 4 + [102] * 2 + [2] * 5 + [0])
    runs = runs_from_labels(v)
    assert runs == [
        [-1, 0, 3],
        [0, 3, 7],
        [102, 7, 9],
        [2, 9, 14],
        [0, 14, 15],
    ]
    rebuilt = np.concatenate([[lab] * (e - s) for lab, s, e in runs])
    assert np.array_equal(rebuilt, v)
    assert runs_from_labels(np.array([], dtype=int)) == []
