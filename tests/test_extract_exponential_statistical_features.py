# testes/test_extract_ew_statistical_features.py
import pytest
import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, field_validator
from abc import ABC

from ThreeWToolkit.feature_extraction.extract_exponential_statistics_features import (
    ExtractEWStatisticalFeatures,
    EWStatisticalConfig)


class TestExtractEWStatisticalFeatures:
    """Unit tests for the ExtractEWStatisticalFeatures class."""

    def test_basic_extraction(self):
        """Tests that all EW statistical features are calculated correctly."""
        window_size = 10
        decay = 0.9
        data_array = np.arange(window_size, dtype=np.float64)
        data = pd.DataFrame({"signal": data_array})

        config = EWStatisticalConfig(window_size=window_size, decay=decay, overlap=0.0)
        extractor = ExtractEWStatisticalFeatures(config)
        result = extractor(data)

        # Replicate EW calculations for validation
        ew_weights = decay ** np.arange(window_size, 0, -1)
        weights = ew_weights / np.sum(np.abs(ew_weights))
        
        expected_mean = np.dot(data_array, weights)
        expected_std = np.sqrt(np.dot((data_array - expected_mean)**2, weights))
        
        c_data = (data_array - expected_mean) / (expected_std + 1e-6)
        expected_skew = np.dot(c_data**3, weights)
        expected_kurt = np.dot(c_data**4, weights)

        q_expected = np.quantile(c_data, [0.00, 0.25, 0.50, 0.75, 1.00])
        
        assert len(result) == 1
        res = result.iloc[0]
        assert np.isclose(res["signal_ew_mean"], expected_mean)
        assert np.isclose(res["signal_ew_std"], expected_std)
        assert np.isclose(res["signal_ew_skew"], expected_skew)
        assert np.isclose(res["signal_ew_kurt"], expected_kurt)
        assert np.isclose(res["signal_ew_min"], q_expected[0])

    def test_offset_parameter(self):
        """Tests if the offset correctly ignores initial data points."""
        data = pd.DataFrame({"signal": np.arange(25, dtype=np.float64)})
        config = EWStatisticalConfig(window_size=10, decay=0.9, overlap=0.0, offset=5)
        extractor = ExtractEWStatisticalFeatures(config)
        result = extractor(data)

        raw_window_data = np.arange(5, 15)
        ew_weights = 0.9 ** np.arange(10, 0, -1)
        weights = ew_weights / np.sum(np.abs(ew_weights))
        expected_first_mean = np.dot(raw_window_data, weights)
        
        assert np.isclose(result["signal_ew_mean"].iloc[0], expected_first_mean)

    def test_multiple_columns(self):
        """Tests if the extractor works for a DataFrame with multiple columns."""
        data = pd.DataFrame({
            "s1": np.arange(20, dtype=np.float64),
            "s2": np.arange(20, 40, dtype=np.float64)
        })
        config = EWStatisticalConfig(window_size=10, decay=0.9, overlap=0.8)
        extractor = ExtractEWStatisticalFeatures(config)
        result = extractor(data)
        
        assert "s1_ew_mean" in result.columns
        assert "s2_ew_mean" in result.columns

    def test_insufficient_data(self):
        """Tests that an empty DataFrame is returned for insufficient data."""
        data = pd.DataFrame({"signal": np.arange(5)})
        data.index.name = "idx"
        config = EWStatisticalConfig(window_size=10, decay=0.9, overlap=0.0)
        extractor = ExtractEWStatisticalFeatures(config)
        result = extractor(data)
        assert result.empty
        assert result.index.name == "idx"

    def test_invalid_config_raises_error(self):
        """
        Tests that the Pydantic validators raise a ValueError for invalid
        overlap and eps values, and also covers the valid eps path.
        """
        
        # Test cases for invalid overlap
        with pytest.raises(ValueError, match="Overlap must be in the range"):
            EWStatisticalConfig(window_size=10, decay=0.9, overlap=1.0)
        
        with pytest.raises(ValueError, match="Overlap must be in the range"):
            EWStatisticalConfig(window_size=10, decay=0.9, overlap=-0.1)
            
        # Test case for invalid offset
        with pytest.raises(ValueError, match="Offset must be a non-negative integer"):
            EWStatisticalConfig(window_size=10, decay=0.9, overlap=0.5, offset=-1)

        # Test case for the valid offset path
        try:
            EWStatisticalConfig(window_size=10, decay=0.9, overlap=0.5, offset=0)
            EWStatisticalConfig(window_size=10, decay=0.9, overlap=0.5, offset=10)
        except ValueError:
            pytest.fail("A ValueError was raised for a valid non-negative offset.")
            
        # Test cases for invalid eps
        with pytest.raises(ValueError, match="Epsilon .* must be positive"):
            EWStatisticalConfig(window_size=10, decay=0.9, overlap=0.5, eps=0)

        with pytest.raises(ValueError, match="Epsilon .* must be positive"):
            EWStatisticalConfig(window_size=10, decay=0.9, overlap=0.5, eps=-1e-9)
        
        # Test case for the valid eps path
        try:
            # Create a config with a valid, positive epsilon
            EWStatisticalConfig(window_size=10, decay=0.9, overlap=0.5, eps=0.1)
        except ValueError:
            pytest.fail("A ValueError was raised for a valid positive epsilon.")

    def test_output_column_names(self):
        """Tests that the output DataFrame has correctly formatted column names."""
        input_cols = ["alpha", "beta"]
        data = pd.DataFrame({"alpha": np.arange(10), "beta": np.arange(10, 20)})
        config = EWStatisticalConfig(window_size=5, decay=0.9, overlap=0.0)
        extractor = ExtractEWStatisticalFeatures(config)
        result = extractor(data)
        feature_suffixes = ExtractEWStatisticalFeatures.FEATURES
        expected_columns = [f"{col}_{feat}" for feat in feature_suffixes for col in input_cols]
        assert list(result.columns) == expected_columns

    def test_handles_empty_windows_from_toolkit_function(self, monkeypatch):
        """Tests the early-return path if the windowing function returns empty."""
        def mock_windowing(*args, **kwargs):
            return pd.DataFrame()

        monkeypatch.setattr("ThreeWToolkit.feature_extraction.extract_exponential_statistics_features.windowing",
                            mock_windowing)
        
        data = pd.DataFrame({"s1": np.arange(20)})
        config = EWStatisticalConfig(window_size=10, decay=0.9, overlap=0.5)
        extractor = ExtractEWStatisticalFeatures(config)
        result = extractor(data)
        assert result.empty

    def test_handles_mixed_success_from_windowing(self, monkeypatch):
        """Tests the mixed scenario (one column succeeds, one fails)."""
        def mock_windowing_mixed(X: pd.Series, *args, **kwargs):
            if X.name == "s1":
                window_size = kwargs.get("window_size", 10)
                return pd.DataFrame(np.random.rand(3, window_size))
            else:
                return pd.DataFrame()
        
        # Replacing the windowing function using monkeypatch.setattr
        monkeypatch.setattr("ThreeWToolkit.feature_extraction.extract_exponential_statistics_features.windowing",
                            mock_windowing_mixed)
        
        data = pd.DataFrame({"s1": np.arange(20), "s2": np.arange(20, 40)})
        config = EWStatisticalConfig(window_size=10, decay=0.9, overlap=0.5)
        extractor = ExtractEWStatisticalFeatures(config)
        result = extractor(data)
        
        assert not result.empty
        assert "s1_ew_mean" in result.columns
        assert "s2_ew_mean" not in result.columns
