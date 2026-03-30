"""Tests for FillLabels preprocessing class."""

import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal

from ThreeWToolkit.preprocessing import FillLabels, FillLabelsConfig


# Module-level fixtures

@pytest.fixture
def series_with_nan():
    """Series with NaN values for label filling tests."""
    return pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])


@pytest.fixture
def series_all_nan():
    """Series with all NaN values."""
    return pd.Series([np.nan, np.nan, np.nan])


# Tests for FillLabels functionality


class TestFillLabelsStrategies:
    """Test different fill strategies."""

    def test_nearest_fill_strategy(self, series_with_nan):
        """Test nearest interpolation + bfill + ffill strategy."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    def test_ffill_strategy(self, series_with_nan):
        """Test forward-fill then backward-fill strategy."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    def test_bfill_strategy(self, series_with_nan):
        """Test backward-fill then forward-fill strategy."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    def test_constant_fill_strategy(self, series_with_nan):
        """Test filling with a constant value."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    @pytest.mark.parametrize(
        "strategy",
        ["nearest", "ffill", "bfill"],
    )
    def test_various_fill_strategies(self, series_with_nan, strategy):
        """Test different fill strategies parametrized."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")


class TestFillLabelsEdgeCases:
    """Test edge cases and error handling."""

    def test_all_nan_series(self, series_all_nan):
        """Test filling when all values are NaN."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    def test_no_nan_series(self):
        """Test filling when there are no NaN values."""
        series = pd.Series([1.0, 2.0, 3.0])
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    def test_single_nan_at_start(self):
        """Test filling with NaN at the beginning."""
        series = pd.Series([np.nan, 2.0, 3.0])
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    def test_single_nan_at_end(self):
        """Test filling with NaN at the end."""
        series = pd.Series([1.0, 2.0, np.nan])
        # TODO: Implement test
        pytest.skip("API adaptation needed")
