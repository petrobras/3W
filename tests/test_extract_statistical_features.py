import pytest
import numpy as np
import pandas as pd
import torch
from pandas.testing import assert_frame_equal
from scipy.stats import skew, kurtosis

from pydantic import BaseModel
from abc import ABC

from ThreeWToolkit.feature_extraction.extract_statistical_features import (
    ExtractStatisticalFeatures,
    StatisticalConfig)

# --- Test Suite ---

class TestExtractStatisticalFeatures:
    """
    Unit tests for the ExtractStatisticalFeatures class.
    """

    def test_basic_extraction(self):
        """
        Tests that all statistical features are calculated correctly for a single window.
        """
        # Create data for exactly one window
        window_size = 10
        data_array = np.arange(window_size, dtype=np.float64)
        data = pd.DataFrame({"signal": data_array})

        config = StatisticalConfig(window_size=window_size, stride=1, offset=0)
        extractor = ExtractStatisticalFeatures(config)

        result = extractor(data)

        # Note: Expected values were calculated with `numpy` and `scipy`
        # Note: `torch.std(unbiased=False)` is equivalent to `np.std()`
        # Note: torch's moment calculation matches scipy's with `bias=True`
        expected_mean = np.mean(data_array)
        expected_std = np.std(data_array)
        expected_skew = skew(data_array, bias=True) # bias=True matches moment calculation
        expected_kurt = kurtosis(data_array, fisher=False, bias=True) # Not excess kurtosis
        q = np.quantile(data_array, [0, 0.25, 0.5, 0.75, 1.0])

        assert len(result) == 1
        res = result.iloc[0]

        # Assert each calculated value is close to the expected value
        assert np.isclose(res["signal_mean"], expected_mean)
        assert np.isclose(res["signal_std"], expected_std)
        assert np.isclose(res["signal_skew"], expected_skew)
        assert np.isclose(res["signal_kurt"], expected_kurt)
        assert np.isclose(res["signal_min"], q[0])
        assert np.isclose(res["signal_1qrt"], q[1])
        assert np.isclose(res["signal_med"], q[2])
        assert np.isclose(res["signal_3qrt"], q[3])
        assert np.isclose(res["signal_max"], q[4])

    def test_stride_parameter(self):
        """
        Tests if the "stride" parameter correctly reduces the number of output windows.
        """
        data = pd.DataFrame({"signal": np.arange(30)}) # 30 data points
        config = StatisticalConfig(window_size=10, stride=10, offset=0)
        extractor = ExtractStatisticalFeatures(config)
        result = extractor(data)

        # With 30 points, window 10, stride 10, we expect 3 windows:
        # [0-9], [10-19], [20-29]
        assert len(result) == 3
        # Check if the index corresponds to the end of each window
        assert result.index[0] == 9
        assert result.index[1] == 19
        assert result.index[2] == 29

    def test_offset_parameter(self):
        """
        Tests if the "offset" parameter correctly ignores initial data points.
        """
        data = pd.DataFrame({"signal": np.arange(25, dtype=np.float64)})
        # The data from index 5 onwards is [5, 6, ..., 14]
        config = StatisticalConfig(window_size=10, stride=10, offset=5)
        extractor = ExtractStatisticalFeatures(config)
        result = extractor(data)

        # The first window should be data[5:15], which is np.arange(5, 15)
        expected_first_mean = np.mean(np.arange(5, 15))

        assert len(result) == 2 # Windows [5-14] and [15-24]
        assert result.index[0] == 14 # End of the first window (5+10-1)
        assert np.isclose(result["signal_mean"].iloc[0], expected_first_mean)

    def test_multiple_columns(self):
        """
        Tests if the extractor works correctly for a DataFrame with multiple columns.
        """
        data = pd.DataFrame({
            "s1": np.arange(10, dtype=np.float64),
            "s2": np.arange(20, 30, dtype=np.float64)
        })
        config = StatisticalConfig(window_size=10, stride=1)
        extractor = ExtractStatisticalFeatures(config)
        result = extractor(data)

        # Expect (num_features * num_columns) columns in the output
        # 9 features * 2 columns = 18 output columns
        assert len(result.columns) == 18
        assert "s1_mean" in result.columns
        assert "s2_mean" in result.columns
        assert "s2_max" in result.columns

        # Check a specific value for the second column
        expected_s2_mean = np.mean(np.arange(20, 30))
        assert np.isclose(result["s2_mean"].iloc[0], expected_s2_mean)

    def test_insufficient_data(self):
        """
        Tests that an empty DataFrame is returned if data is not sufficient for a single window.
        """
        data = pd.DataFrame({"signal": np.arange(5)}) # 5 data points
        data.index.name = "idx"
        config = StatisticalConfig(window_size=10, stride=1)
        extractor = ExtractStatisticalFeatures(config)
        result = extractor(data)

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        # Check if the index name is preserved
        assert result.index.name == "idx"
        # Check if expected columns are created, even if empty
        assert "signal_mean" in result.columns