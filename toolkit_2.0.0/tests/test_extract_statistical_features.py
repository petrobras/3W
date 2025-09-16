import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from ThreeWToolkit.feature_extraction.extract_statistical_features import (
    ExtractStatisticalFeatures,
    StatisticalConfig
)


class TestExtractStatisticalFeatures:
    """
    Unit tests for the ExtractStatisticalFeatures class.
    """

    def test_basic_extraction(self):
        """Tests that statistical features are calculated correctly."""
        window_size = 10
        data_array = np.arange(window_size * 2, dtype=np.float64)
        data = pd.DataFrame({"signal": data_array})

        y = pd.Series(np.arange(window_size * 2), index=data.index)

        config = StatisticalConfig(window_size=window_size, overlap=0.8)
        extractor = ExtractStatisticalFeatures(config)
        X, y_out = extractor(data, y=y)

        expected_mean = np.mean(np.arange(window_size))

        assert np.isclose(X["signal_mean"].iloc[0], expected_mean)
        assert list(X.index) == list(y_out.index)

    def test_offset_parameter(self):
        """Tests offset with a boxcar window."""
        data = pd.DataFrame({"signal": np.arange(25, dtype=np.float64)})
        y = pd.Series(np.arange(25), index=data.index)
        config = StatisticalConfig(window_size=10, overlap=0.0, offset=5)
        extractor = ExtractStatisticalFeatures(config)
        X, y_out = extractor(data, y=y)

        expected_first_mean = np.mean(np.arange(5, 15))

        assert np.isclose(X["signal_mean"].iloc[0], expected_first_mean)
        assert list(X.index) == list(y_out.index)

    def test_multiple_columns(self):
        """Tests multiple columns with a boxcar window."""
        data = pd.DataFrame({
            "s1": np.arange(20, dtype=np.float64),
            "s2": np.arange(20, 40, dtype=np.float64)
        })
        y = pd.Series(np.arange(20), index=data.index)
        config = StatisticalConfig(window_size=10, overlap=0.8)
        extractor = ExtractStatisticalFeatures(config)
        X, y_out = extractor(data, y=y)

        expected_s2_mean = np.mean(np.arange(20, 30))
        assert np.isclose(X["s2_mean"].iloc[0], expected_s2_mean)
        assert list(X.index) == list(y_out.index)

    def test_insufficient_data(self):
        """Tests that empty X and y are returned if data is not sufficient."""
        data = pd.DataFrame({"signal": np.arange(5)})
        data.index.name = "idx"
        y = pd.Series(np.arange(5), index=data.index)
        config = StatisticalConfig(window_size=10, overlap=0.0)
        extractor = ExtractStatisticalFeatures(config)
        X, y_out = extractor(data, y=y)

        assert X.empty
        assert y_out.empty
        assert X.index.name == "idx"
        assert y_out.index.name == "idx"

    def test_invalid_config_raises_error(self):
        """Tests that the Pydantic validators raise a ValueError for invalid cases."""

        with pytest.raises(ValueError, match="Overlap must be in the range"):
            StatisticalConfig(window_size=10, overlap=1.0)

        with pytest.raises(ValueError, match="Overlap must be in the range"):
            StatisticalConfig(window_size=10, overlap=-0.1)

        with pytest.raises(ValueError, match="Offset must be a non-negative integer"):
            StatisticalConfig(window_size=10, overlap=0.5, offset=-1)

        try:
            StatisticalConfig(window_size=10, overlap=0.5, offset=0)
            StatisticalConfig(window_size=10, overlap=0.5, offset=10)
        except ValueError:
            pytest.fail("A ValueError was raised for a valid non-negative offset.")

        with pytest.raises(ValueError, match="Epsilon .* must be positive"):
            StatisticalConfig(window_size=10, overlap=0.5, eps=0)

        with pytest.raises(ValueError, match="Epsilon .* must be positive"):
            StatisticalConfig(window_size=10, overlap=0.5, eps=-1e-9)

        try:
            StatisticalConfig(window_size=10, overlap=0.5, eps=0.1)
        except ValueError:
            pytest.fail("A ValueError was raised for a valid positive epsilon.")

    def test_output_column_names(self):
        """Tests that the output DataFrame has correctly formatted column names."""
        input_cols = ["sensor_alpha", "sensor_beta"]
        data = pd.DataFrame({
            "sensor_alpha": np.arange(10),
            "sensor_beta": np.arange(10, 20)
        })
        y = pd.Series(np.arange(10), index=data.index)
        config = StatisticalConfig(window_size=5, overlap=0.0)
        extractor = ExtractStatisticalFeatures(config)
        X, y_out = extractor(data, y=y)

        feature_suffixes = ExtractStatisticalFeatures.FEATURES
        expected_columns = [f"{col}_{feat}" for feat in feature_suffixes for col in input_cols]
        assert list(X.columns) == expected_columns
        assert list(X.index) == list(y_out.index)

    def test_handles_empty_windows_from_toolkit_function(self, monkeypatch):
        """Tests handling of empty windows from the toolkit function."""
        def mock_windowing(*args, **kwargs):
            return pd.DataFrame()

        monkeypatch.setattr(
            "ThreeWToolkit.feature_extraction.extract_statistical_features.windowing",
            mock_windowing,
        )
        data = pd.DataFrame({"s1": np.arange(20), "s2": np.arange(20, 40)})
        data.index.name = "my_idx"
        y = pd.Series(np.arange(20), index=data.index)
        config = StatisticalConfig(window_size=10, overlap=0.5)
        extractor = ExtractStatisticalFeatures(config)
        X, y_out = extractor(data, y=y)

        assert X.empty
        assert y_out.empty
        assert X.index.name == "my_idx"
        assert y_out.index.name == "my_idx"

    def test_handles_mixed_success_from_windowing(self, monkeypatch):
        """Tests scenario where windowing succeeds for one column but returns empty for another."""
        def mock_windowing_mixed(X: pd.Series, *args, **kwargs):
            col_name = X.name
            if col_name == "s1":
                window_size = kwargs.get("window_size", 10)
                return pd.DataFrame(np.random.rand(3, window_size))
            else:
                return pd.DataFrame()

        monkeypatch.setattr(
            "ThreeWToolkit.feature_extraction.extract_statistical_features.windowing",
            mock_windowing_mixed,
        )
        data = pd.DataFrame({"s1": np.arange(20), "s2": np.arange(20, 40)})
        y = pd.Series(np.arange(20), index=data.index)
        config = StatisticalConfig(window_size=10, overlap=0.5)
        extractor = ExtractStatisticalFeatures(config)
        X, y_out = extractor(data, y=y)

        assert not X.empty
        assert "s1_mean" in X.columns
        assert "s2_mean" not in X.columns
        assert list(X.index) == list(y_out.index)
        
    def test_raises_error_if_y_is_not_provided(self):
        """Tests that a ValueError is raised if the 'y' labels are not passed."""
        
        tags = pd.DataFrame({"signal": np.arange(20)})
        config = StatisticalConfig(window_size=10, overlap=0.5)
        extractor = ExtractStatisticalFeatures(config)

        with pytest.raises(ValueError, match="The 'y' series .* must be provided"):
            extractor(tags=tags, y=None)
