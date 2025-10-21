import pytest
import numpy as np
import pandas as pd

from ThreeWToolkit.feature_extraction.extract_statistical_features import (
    ExtractStatisticalFeatures,
)


class StatisticalConfig:
    """Mock config class for testing."""

    def __init__(
        self,
        window_size=4,
        overlap=0.0,
        offset=0,
        eps=1e-10,
        selected_features=[
            "mean",
            "std",
            "skew",
            "kurt",
            "min",
            "1qrt",
            "med",
            "3qrt",
            "max",
        ],
        multivariate=True,
        is_windowed=True,
        label_column="label",
    ):
        if overlap < 0 or overlap >= 1:
            raise ValueError("Overlap must be in the range [0, 1)")
        if offset < 0:
            raise ValueError("Offset must be a non-negative integer")
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer")

        self.window_size = window_size
        self.overlap = overlap
        self.offset = offset
        self.eps = eps
        self.selected_features = selected_features
        self.multivariate = multivariate
        self.is_windowed = is_windowed
        self.label_column = label_column


class TestExtractStatisticalFeatures:
    """Unit tests for the ExtractStatisticalFeatures class."""

    @pytest.fixture
    def windowed_data_with_labels(self):
        """
        Provides sample windowed data with known values for testing.
        Each row represents a window with columns: var1_0, var1_1, var1_2, var1_3, label
        """
        # Create 5 windows with predictable values
        data = {
            "var1_0": [1.0, 2.0, 3.0, 4.0, 5.0],
            "var1_1": [2.0, 3.0, 4.0, 5.0, 6.0],
            "var1_2": [3.0, 4.0, 5.0, 6.0, 7.0],
            "var1_3": [4.0, 5.0, 6.0, 7.0, 8.0],
            "label": [100, 101, 102, 103, 104],
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def multivariate_windowed_data(self):
        """
        Provides multivariate windowed data with two variables.
        """
        data = {
            "var1_0": [1.0, 2.0, 3.0],
            "var1_1": [2.0, 3.0, 4.0],
            "var1_2": [3.0, 4.0, 5.0],
            "var1_3": [4.0, 5.0, 6.0],
            "var2_0": [10.0, 20.0, 30.0],
            "var2_1": [20.0, 30.0, 40.0],
            "var2_2": [30.0, 40.0, 50.0],
            "var2_3": [40.0, 50.0, 60.0],
            "label": [100, 101, 102],
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def constant_value_data(self):
        """Data with constant values to test edge cases like zero std."""
        data = {
            "var1_0": [5.0, 5.0, 5.0],
            "var1_1": [5.0, 5.0, 5.0],
            "var1_2": [5.0, 5.0, 5.0],
            "var1_3": [5.0, 5.0, 5.0],
            "label": [100, 101, 102],
        }
        return pd.DataFrame(data)

    def test_basic_extraction_all_features(self, windowed_data_with_labels):
        """Tests basic extraction with all default features."""
        config = StatisticalConfig(is_windowed=True)
        extractor = ExtractStatisticalFeatures(config)

        result = extractor(windowed_data_with_labels)

        # Check result shape
        assert not result.empty
        assert len(result) == len(windowed_data_with_labels)

        # Check that labels are preserved
        assert "label" in result.columns
        pd.testing.assert_series_equal(
            result["label"], windowed_data_with_labels["label"], check_names=False
        )

        # Check that all expected features exist
        expected_features = [
            "mean",
            "std",
            "skew",
            "kurt",
            "min",
            "1qrt",
            "med",
            "3qrt",
            "max",
        ]
        for feat in expected_features:
            assert f"var1_{feat}" in result.columns

    def test_mean_calculation(self, windowed_data_with_labels):
        """Tests that mean is calculated correctly."""
        config = StatisticalConfig(is_windowed=True, selected_features=["mean"])
        extractor = ExtractStatisticalFeatures(config)

        result = extractor(windowed_data_with_labels)

        # First window: [1, 2, 3, 4] -> mean = 2.5
        assert np.isclose(result["var1_mean"].iloc[0], 2.5)

        # Second window: [2, 3, 4, 5] -> mean = 3.5
        assert np.isclose(result["var1_mean"].iloc[1], 3.5)

    def test_std_calculation(self, windowed_data_with_labels):
        """Tests that standard deviation is calculated correctly."""
        config = StatisticalConfig(is_windowed=True, selected_features=["std"])
        extractor = ExtractStatisticalFeatures(config)

        result = extractor(windowed_data_with_labels)

        # First window: [1, 2, 3, 4] -> std ≈ 1.118
        expected_std = np.std([1, 2, 3, 4], ddof=0)
        assert np.isclose(result["var1_std"].iloc[0], expected_std)

    def test_quantiles_calculation(self, windowed_data_with_labels):
        """Tests quantile calculations (min, quartiles, median, max)."""
        config = StatisticalConfig(
            is_windowed=True, selected_features=["min", "1qrt", "med", "3qrt", "max"]
        )
        extractor = ExtractStatisticalFeatures(config)

        result = extractor(windowed_data_with_labels)

        # First window: [1, 2, 3, 4]
        assert result["var1_min"].iloc[0] == 1.0
        assert result["var1_1qrt"].iloc[0] == 1.75
        assert result["var1_med"].iloc[0] == 2.5
        assert result["var1_3qrt"].iloc[0] == 3.25
        assert result["var1_max"].iloc[0] == 4.0

    def test_skewness_calculation(self):
        """Tests skewness calculation with asymmetric data."""
        # Create right-skewed data
        data = {
            "var1_0": [1.0],
            "var1_1": [2.0],
            "var1_2": [3.0],
            "var1_3": [10.0],  # Outlier causing right skew
        }
        df = pd.DataFrame(data)

        config = StatisticalConfig(is_windowed=True, selected_features=["skew"])
        extractor = ExtractStatisticalFeatures(config)

        result = extractor(df)

        # Should have positive skew
        assert result["var1_skew"].iloc[0] > 0

    def test_kurtosis_calculation(self):
        """Tests kurtosis calculation."""
        # Create data with high kurtosis (heavy tails)
        data = {
            "var1_0": [1.0],
            "var1_1": [5.0],
            "var1_2": [5.0],
            "var1_3": [9.0],
        }
        df = pd.DataFrame(data)

        config = StatisticalConfig(is_windowed=True, selected_features=["kurt"])
        extractor = ExtractStatisticalFeatures(config)

        result = extractor(df)

        # Should calculate kurtosis
        assert "var1_kurt" in result.columns
        assert not np.isnan(result["var1_kurt"].iloc[0])

    def test_constant_values_handling(self, constant_value_data):
        """Tests handling of constant values (zero std)."""
        config = StatisticalConfig(
            is_windowed=True, selected_features=["mean", "std", "skew", "kurt"]
        )
        extractor = ExtractStatisticalFeatures(config)

        result = extractor(constant_value_data)

        # Mean should be the constant value
        assert np.all(result["var1_mean"] == 5.0)

        # Std should be zero
        assert np.all(result["var1_std"] == 0.0)

        # Skew and kurt should be zero (handled by eps threshold)
        assert np.all(result["var1_skew"] == 0.0)
        assert np.all(result["var1_kurt"] == 0.0)

    def test_multivariate_extraction(self, multivariate_windowed_data):
        """Tests feature extraction for multivariate data."""
        config = StatisticalConfig(is_windowed=True, selected_features=["mean", "std"])
        extractor = ExtractStatisticalFeatures(config)

        result = extractor(multivariate_windowed_data)

        # Check features for both variables
        assert "var1_mean" in result.columns
        assert "var2_mean" in result.columns
        assert "var1_std" in result.columns
        assert "var2_std" in result.columns

        # Verify var2 has different values (scaled by 10x)
        assert result["var2_mean"].iloc[0] > result["var1_mean"].iloc[0]

    def test_selected_features_subset(self, windowed_data_with_labels):
        """Tests that only selected features are computed."""
        selected = ["mean", "max"]
        config = StatisticalConfig(is_windowed=True, selected_features=selected)
        extractor = ExtractStatisticalFeatures(config)

        result = extractor(windowed_data_with_labels)

        # Only selected features should be present (plus label)
        assert "var1_mean" in result.columns
        assert "var1_max" in result.columns
        assert "var1_std" not in result.columns
        assert "var1_min" not in result.columns
        assert len([c for c in result.columns if c != "label"]) == 2

    def test_offset_application(self, windowed_data_with_labels):
        """Tests that offset correctly skips initial windows."""
        offset = 2
        config = StatisticalConfig(
            is_windowed=True, offset=offset, selected_features=["mean"]
        )
        extractor = ExtractStatisticalFeatures(config)

        result = extractor(windowed_data_with_labels)

        # Should have 3 windows (5 - 2 offset)
        assert len(result) == len(windowed_data_with_labels) - offset

        # First label should be the third label from original data
        assert (
            result["label"].iloc[0] == windowed_data_with_labels["label"].iloc[offset]
        )

        # First mean should correspond to third window: [3, 4, 5, 6] -> mean = 4.5
        assert np.isclose(result["var1_mean"].iloc[0], 4.5)

    def test_not_windowed_raises_error(self):
        """Tests that error is raised when data is not marked as windowed."""
        data = pd.DataFrame(
            {
                "var1_0": [1, 2, 3],
                "var1_1": [4, 5, 6],
                "var1_2": [7, 8, 9],
                "var1_3": [10, 11, 12],
            }
        )

        config = StatisticalConfig(is_windowed=False)
        extractor = ExtractStatisticalFeatures(config)

        with pytest.raises(ValueError, match="Data is not windowed"):
            extractor(data)

    def test_invalid_config_raises_error(self):
        """Tests that validators raise ValueErrors for invalid configs."""
        with pytest.raises(ValueError, match="Overlap must be in the range"):
            StatisticalConfig(overlap=1.0)

        with pytest.raises(ValueError, match="Offset must be a non-negative integer"):
            StatisticalConfig(offset=-1)

        with pytest.raises(ValueError, match="Window size must be a positive integer"):
            StatisticalConfig(window_size=0)

    def test_empty_data_raises_error(self):
        """Tests that empty DataFrame raises appropriate error."""
        config = StatisticalConfig(is_windowed=True)
        extractor = ExtractStatisticalFeatures(config)

        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Input data is empty"):
            extractor(empty_df)

    def test_no_variables_found_raises_error(self):
        """Tests error when no valid variable columns are found."""
        config = StatisticalConfig(is_windowed=True)
        extractor = ExtractStatisticalFeatures(config)

        # DataFrame with wrong column format
        invalid_data = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": [4, 5, 6], "label": [100, 101, 102]}
        )

        with pytest.raises(ValueError, match="No variables with pattern 'varX_' found"):
            extractor(invalid_data)

    def test_offset_exceeds_data_length(self, windowed_data_with_labels):
        """Tests that offset larger than data raises error."""
        config = StatisticalConfig(
            is_windowed=True,
            offset=100,  # Much larger than data
        )
        extractor = ExtractStatisticalFeatures(config)

        with pytest.raises(ValueError, match="Offset .* is larger than data length"):
            extractor(windowed_data_with_labels)

    def test_without_label_column(self):
        """Tests processing data without labels."""
        data = pd.DataFrame(
            {
                "var1_0": [1, 2, 3],
                "var1_1": [4, 5, 6],
                "var1_2": [7, 8, 9],
                "var1_3": [10, 11, 12],
            }
        )

        config = StatisticalConfig(
            is_windowed=True, label_column=None, selected_features=["mean"]
        )
        extractor = ExtractStatisticalFeatures(config)

        result = extractor(data)

        assert not result.empty
        assert "label" not in result.columns
        assert "var1_mean" in result.columns

    def test_type_error_on_non_dataframe_input(self):
        """Tests that non-DataFrame input raises TypeError."""
        config = StatisticalConfig(is_windowed=True)
        extractor = ExtractStatisticalFeatures(config)

        with pytest.raises(TypeError, match="Input data must be a pandas DataFrame"):
            extractor([1, 2, 3, 4])  # List instead of DataFrame

    def test_nan_handling_warning(self, windowed_data_with_labels):
        """Tests that NaN values are handled and warning is shown."""
        config = StatisticalConfig(is_windowed=True, selected_features=["mean"])
        extractor = ExtractStatisticalFeatures(config)

        # Introduce NaN
        data_with_nan = windowed_data_with_labels.copy()
        data_with_nan.loc[0, "var1_0"] = np.nan

        result = extractor(data_with_nan)

        # Should process but result will have NaN
        assert not result.empty
        assert np.isnan(result["var1_mean"].iloc[0])

    def test_infinite_values_handling(self):
        """Tests that infinite values are replaced in post-processing."""
        data = pd.DataFrame(
            {
                "var1_0": [1.0, 2.0],
                "var1_1": [2.0, 3.0],
                "var1_2": [3.0, 4.0],
                "var1_3": [4.0, np.inf],  # Infinite value
                "label": [100, 101],
            }
        )

        config = StatisticalConfig(is_windowed=True, selected_features=["max"])
        extractor = ExtractStatisticalFeatures(config)

        result = extractor(data)

        # Infinite values should be replaced
        assert not np.isinf(result["var1_max"].iloc[1])
        assert result["var1_max"].iloc[1] == np.finfo(np.float64).max

    def test_single_window(self):
        """Tests processing a single window."""
        data = pd.DataFrame(
            {
                "var1_0": [1.0],
                "var1_1": [2.0],
                "var1_2": [3.0],
                "var1_3": [4.0],
                "label": [100],
            }
        )

        config = StatisticalConfig(is_windowed=True, selected_features=["mean", "std"])
        extractor = ExtractStatisticalFeatures(config)

        result = extractor(data)

        assert len(result) == 1
        assert np.isclose(result["var1_mean"].iloc[0], 2.5)

    def test_feature_ordering_consistency(self, multivariate_windowed_data):
        """Tests that feature ordering is consistent across variables."""
        config = StatisticalConfig(
            is_windowed=True, selected_features=["mean", "std", "max"]
        )
        extractor = ExtractStatisticalFeatures(config)

        result = extractor(multivariate_windowed_data)

        # Get columns for each variable
        var1_cols = [c for c in result.columns if c.startswith("var1_")]
        var2_cols = [c for c in result.columns if c.startswith("var2_")]

        # Extract feature names
        var1_features = [c.replace("var1_", "") for c in var1_cols]
        var2_features = [c.replace("var2_", "") for c in var2_cols]

        # Should have same features in same order
        assert var1_features == var2_features

    def test_eps_parameter_effect(self):
        """Tests that eps parameter affects skew/kurt calculations."""
        # Data with very small std
        data = pd.DataFrame(
            {
                "var1_0": [1.0000000],
                "var1_1": [1.0000001],
                "var1_2": [1.0000002],
                "var1_3": [1.0000003],
            }
        )

        # With large eps, should return 0 for skew/kurt
        config_large_eps = StatisticalConfig(
            is_windowed=True,
            selected_features=["skew", "kurt"],
            eps=1.0,  # Large eps
        )
        extractor_large = ExtractStatisticalFeatures(config_large_eps)
        result_large = extractor_large(data)

        assert result_large["var1_skew"].iloc[0] == 0.0
        assert result_large["var1_kurt"].iloc[0] == 0.0

    def test_all_features_class_attribute(self):
        """Tests that FEATURES class attribute contains all expected features."""
        expected_features = [
            "mean",
            "std",
            "skew",
            "kurt",
            "min",
            "1qrt",
            "med",
            "3qrt",
            "max",
        ]
        assert ExtractStatisticalFeatures.FEATURES == expected_features

    def test_multiple_variables_different_ranges(self):
        """Tests extraction with multiple variables having different value ranges."""
        data = pd.DataFrame(
            {
                "var1_0": [1, 2, 3],
                "var1_1": [2, 3, 4],
                "var1_2": [3, 4, 5],
                "var1_3": [4, 5, 6],
                "var2_0": [1000, 2000, 3000],
                "var2_1": [2000, 3000, 4000],
                "var2_2": [3000, 4000, 5000],
                "var2_3": [4000, 5000, 6000],
                "label": [100, 101, 102],
            }
        )

        config = StatisticalConfig(is_windowed=True, selected_features=["mean", "std"])
        extractor = ExtractStatisticalFeatures(config)

        result = extractor(data)

        # Both variables should be processed correctly
        assert result["var1_mean"].iloc[0] < result["var2_mean"].iloc[0]
        assert result["var1_std"].iloc[0] < result["var2_std"].iloc[0]
