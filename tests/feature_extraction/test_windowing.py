"""Tests for Windowing feature extraction class."""

import pytest
import pandas as pd
import numpy as np

from ThreeWToolkit.core.base_dataset import BaseDataset
from ThreeWToolkit.dataset.transformed_dataset import TransformedDataset

from ThreeWToolkit.feature_extraction.windowing import WindowingConfig


@pytest.fixture
def simple_dataset(mock_dataset_factory) -> BaseDataset:
    """Simple dataset for normalization tests."""
    return mock_dataset_factory(num_sensors=10)


@pytest.fixture
def multivariate_timeseries():
    """Multivariate time series for windowing tests."""
    return pd.DataFrame(
        {
            "var1": np.arange(100),
            "var2": np.arange(100, 200),
            "var3": np.arange(200, 300),
        }
    )


class TestWindowing:
    """Test basic windowing functionality."""

    @pytest.mark.parametrize("win_size", [64, 128, 256])
    def test_window_size(self, simple_dataset, win_size: int):

        windowing = WindowingConfig(window_size=win_size, overlap=0.5).build()
        windowed_dataset = TransformedDataset(simple_dataset, windowing.transform)

        for original, windowed in zip(simple_dataset, windowed_dataset):
            assert (
                "window" in windowed.signal.index.names
            ), "Expected 'window' in signal index names"
            assert (
                "variable" in windowed.signal.index.names
            ), "Expected 'variable' in signal index names"
            assert (
                windowed.signal.shape[1] == win_size
            ), f"Expected window size {win_size}, got {windowed.signal.shape[1]}"

            windowed_vars = windowed.signal.index.get_level_values("variable").unique()
            assert set(windowed_vars) == set(
                original.signal.columns
            ), "Expected windowed variables to match original signal columns"

    @pytest.mark.parametrize("win_size", [64, 128, 256])
    @pytest.mark.parametrize("overlap", [0.0, 0.25, 0.5, 0.75])
    @pytest.mark.parametrize("pad_start", [True, False])
    @pytest.mark.parametrize("pad_last", [True, False])
    def test_window_padding(
        self,
        simple_dataset,
        win_size: int,
        overlap: float,
        pad_start: bool,
        pad_last: bool,
    ):

        windowing = WindowingConfig(
            window_size=win_size,
            overlap=overlap,
            pad_start=pad_start,
            pad_last_window=pad_last,
        ).build()
        windowed_dataset = TransformedDataset(simple_dataset, windowing.transform)

        step = np.floor(win_size * (1.0 - overlap)).astype(int)
        step = max(step, 1)  # ensure step is at least 1 to avoid infinite loops

        start_padding = win_size - 1 if pad_start else 0

        for original, windowed in zip(simple_dataset, windowed_dataset):
            n_samples = start_padding + original.signal.shape[0] - win_size
            if pad_last:  # round up for last window if padding
                expected_n_windows = int(np.ceil(n_samples / step) + 1)
            else:  # round down for last window if no padding
                expected_n_windows = int(np.floor(n_samples / step) + 1)

            actual_windows = windowed.signal.index.get_level_values("window").nunique()
            assert (
                actual_windows == expected_n_windows
            ), f"Expected {expected_n_windows} windows, got {actual_windows}"
