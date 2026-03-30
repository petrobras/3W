"""Tests for CleanSignals preprocessing class."""

import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

from ThreeWToolkit.preprocessing import CleanSignals, CleanSignalsConfig


# Module-level fixtures

@pytest.fixture
def df_with_frozen_signals():
    """DataFrame with frozen (constant) signals."""
    return pd.DataFrame({
        "normal": [1.0, 2.0, 3.0, 4.0, 5.0],
        "frozen": [10.0, 10.0, 10.0, 10.0, 10.0],
        "varying": [1.0, 1.5, 2.0, 2.5, 3.0],
    })


@pytest.fixture
def df_with_outliers():
    """DataFrame with outlier values."""
    return pd.DataFrame({
        "normal": [1.0, 2.0, 3.0, 4.0, 5.0],
        "outlier": [1.0, 1.0, 1.0, 100.0, 1.0],
    })


# Tests for CleanSignals functionality


class TestCleanSignalsIQRThresholds:
    """Test IQR threshold detection and filtering."""

    def test_frozen_signal_detection(self, df_with_frozen_signals):
        """Test detection and filtering of frozen (constant) signals."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    def test_outlier_signal_detection(self, df_with_outliers):
        """Test detection and filtering of signals with outliers."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    @pytest.mark.parametrize(
        "iqr_multiplier",
        [1.5, 2.0, 3.0],
    )
    def test_various_iqr_thresholds(self, iqr_multiplier):
        """Test CleanSignals with different IQR threshold multipliers."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")


class TestCleanSignalsCategoricalExclusion:
    """Test categorical feature exclusion."""

    def test_categorical_columns_excluded(self):
        """Test that categorical columns are excluded from cleaning."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    def test_estado_columns_excluded(self):
        """Test that ESTADO-* columns are excluded from cleaning."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")


class TestCleanSignalsMissingThresholds:
    """Test missing value threshold filtering."""

    def test_high_missing_columns_dropped(self):
        """Test that columns with >60% missing values are dropped."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    def test_custom_missing_threshold(self):
        """Test custom missing value threshold."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")
