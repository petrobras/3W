# testes/test_extract_ew_statistical_features.py
import pytest
import numpy as np
import pandas as pd

from ThreeWToolkit.feature_extraction.extract_exponential_statistics_features import (
    ExtractEWStatisticalFeatures,
)


class EWStatisticalConfig:
    """Mock config class for testing."""

    # Default features matching the class
    DEFAULT_FEATURES = [
        "ew_mean",
        "ew_std",
        "ew_skew",
        "ew_kurt",
        "ew_min",
        "ew_1qrt",
        "ew_med",
        "ew_3qrt",
        "ew_max",
    ]

    def __init__(
        self,
        window_size=4,
        overlap=0.0,
        offset=0,
        eps=1e-10,
        decay=0.9,
        selected_features=None,
        is_windowed=True,
        label_column="label",
    ):
        if overlap < 0 or overlap >= 1:
            raise ValueError("Overlap must be in the range [0, 1)")
        if offset < 0:
            raise ValueError("Offset must be a non-negative integer")
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer")
        if decay <= 0 or decay > 1:
            raise ValueError("Decay must be in the range (0, 1]")

        self.window_size = window_size
        self.overlap = overlap
        self.offset = offset
        self.eps = eps
        self.decay = decay
        # If None, use all default features
        self.selected_features = (
            selected_features
            if selected_features is not None
            else self.DEFAULT_FEATURES
        )
        self.is_windowed = is_windowed
        self.label_column = label_column


class TestExtractEWStatisticalFeatures:
    """Unit tests for the ExtractEWStatisticalFeatures class."""

    @pytest.fixture
    def windowed_data_with_labels(self):
        """
        Provides sample windowed data with known values.
        Each row represents a window with columns: var1_0, var1_1, var1_2, var1_3, label
        """
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
        """Provides multivariate windowed data with two variables."""
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
        """Data with constant values to test edge cases."""
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
        config = EWStatisticalConfig(is_windowed=True, window_size=4)
        extractor = ExtractEWStatisticalFeatures(config)

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
            "ew_mean",
            "ew_std",
            "ew_skew",
            "ew_kurt",
            "ew_min",
            "ew_1qrt",
            "ew_med",
            "ew_3qrt",
            "ew_max",
        ]
        for feat in expected_features:
            assert f"var1_{feat}" in result.columns

    def test_ew_mean_calculation(self, windowed_data_with_labels):
        """Tests that exponentially weighted mean is calculated correctly."""
        config = EWStatisticalConfig(
            is_windowed=True, window_size=4, decay=0.9, selected_features=["ew_mean"]
        )
        extractor = ExtractEWStatisticalFeatures(config)

        result = extractor(windowed_data_with_labels)

        # EW mean should be different from regular mean
        # and should be closer to recent values (higher indices)
        first_window = windowed_data_with_labels.iloc[0, :4].values
        regular_mean = np.mean(first_window)
        ew_mean = result["var1_ew_mean"].iloc[0]

        # EW mean should be > regular mean for increasing sequence
        assert ew_mean > regular_mean

    def test_ew_std_calculation(self, windowed_data_with_labels):
        """Tests exponentially weighted standard deviation."""
        config = EWStatisticalConfig(
            is_windowed=True, window_size=4, decay=0.9, selected_features=["ew_std"]
        )
        extractor = ExtractEWStatisticalFeatures(config)

        result = extractor(windowed_data_with_labels)

        # EW std should be positive for non-constant data
        assert np.all(result["var1_ew_std"] >= 0)

    def test_ew_skewness_calculation(self):
        """Tests exponentially weighted skewness with asymmetric data."""
        # Create right-skewed data
        data = {
            "var1_0": [1.0],
            "var1_1": [2.0],
            "var1_2": [3.0],
            "var1_3": [10.0],  # Outlier
        }
        df = pd.DataFrame(data)

        config = EWStatisticalConfig(
            is_windowed=True, window_size=4, decay=0.9, selected_features=["ew_skew"]
        )
        extractor = ExtractEWStatisticalFeatures(config)

        result = extractor(df)

        # Should calculate skewness
        assert "var1_ew_skew" in result.columns
        assert not np.isnan(result["var1_ew_skew"].iloc[0])

    def test_ew_kurtosis_calculation(self):
        """Tests exponentially weighted kurtosis."""
        data = {
            "var1_0": [1.0],
            "var1_1": [5.0],
            "var1_2": [5.0],
            "var1_3": [9.0],
        }
        df = pd.DataFrame(data)

        config = EWStatisticalConfig(
            is_windowed=True, window_size=4, decay=0.9, selected_features=["ew_kurt"]
        )
        extractor = ExtractEWStatisticalFeatures(config)

        result = extractor(df)

        assert "var1_ew_kurt" in result.columns
        assert not np.isnan(result["var1_ew_kurt"].iloc[0])

    def test_ew_quantiles_calculation(self, windowed_data_with_labels):
        """Tests exponentially weighted quantile calculations."""
        config = EWStatisticalConfig(
            is_windowed=True,
            window_size=4,
            decay=0.9,
            selected_features=["ew_min", "ew_1qrt", "ew_med", "ew_3qrt", "ew_max"],
        )
        extractor = ExtractEWStatisticalFeatures(config)

        result = extractor(windowed_data_with_labels)

        # Check all quantiles exist
        assert "var1_ew_min" in result.columns
        assert "var1_ew_1qrt" in result.columns
        assert "var1_ew_med" in result.columns
        assert "var1_ew_3qrt" in result.columns
        assert "var1_ew_max" in result.columns

        # Quantiles should be ordered: min < 1qrt < med < 3qrt < max
        for i in range(len(result)):
            assert result["var1_ew_min"].iloc[i] <= result["var1_ew_1qrt"].iloc[i]
            assert result["var1_ew_1qrt"].iloc[i] <= result["var1_ew_med"].iloc[i]
            assert result["var1_ew_med"].iloc[i] <= result["var1_ew_3qrt"].iloc[i]
            assert result["var1_ew_3qrt"].iloc[i] <= result["var1_ew_max"].iloc[i]

    def test_constant_values_handling(self, constant_value_data):
        """Tests handling of constant values (zero std)."""
        config = EWStatisticalConfig(
            is_windowed=True,
            window_size=4,
            decay=0.9,
            selected_features=["ew_mean", "ew_std", "ew_skew", "ew_kurt"],
        )
        extractor = ExtractEWStatisticalFeatures(config)

        result = extractor(constant_value_data)

        # Mean should be the constant value
        assert np.all(np.isclose(result["var1_ew_mean"], 5.0))

        # Std should be zero or very close to zero
        assert np.all(result["var1_ew_std"] < 1e-6)

        # Skew and kurt should be finite (handled by eps)
        assert not np.any(np.isnan(result["var1_ew_skew"]))
        assert not np.any(np.isnan(result["var1_ew_kurt"]))

    def test_multivariate_extraction(self, multivariate_windowed_data):
        """Tests feature extraction for multivariate data."""
        config = EWStatisticalConfig(
            is_windowed=True,
            window_size=4,
            decay=0.9,
            selected_features=["ew_mean", "ew_std"],
        )
        extractor = ExtractEWStatisticalFeatures(config)

        result = extractor(multivariate_windowed_data)

        # Check features for both variables
        assert "var1_ew_mean" in result.columns
        assert "var2_ew_mean" in result.columns
        assert "var1_ew_std" in result.columns
        assert "var2_ew_std" in result.columns

        # Verify var2 has different (larger) values
        assert result["var2_ew_mean"].iloc[0] > result["var1_ew_mean"].iloc[0]

    def test_selected_features_subset(self, windowed_data_with_labels):
        """Tests that only selected features are computed."""
        selected = ["ew_mean", "ew_max"]
        config = EWStatisticalConfig(
            is_windowed=True, window_size=4, decay=0.9, selected_features=selected
        )
        extractor = ExtractEWStatisticalFeatures(config)

        result = extractor(windowed_data_with_labels)

        # Only selected features should be present (plus label)
        assert "var1_ew_mean" in result.columns
        assert "var1_ew_max" in result.columns
        assert "var1_ew_std" not in result.columns
        assert "var1_ew_min" not in result.columns

    def test_decay_parameter_effect(self, windowed_data_with_labels):
        """Tests that different decay values produce different results."""
        # High decay (more uniform weights)
        config_high = EWStatisticalConfig(
            is_windowed=True, window_size=4, decay=0.95, selected_features=["ew_mean"]
        )
        extractor_high = ExtractEWStatisticalFeatures(config_high)
        result_high = extractor_high(windowed_data_with_labels)

        # Low decay (more weight concentrated on recent values)
        config_low = EWStatisticalConfig(
            is_windowed=True, window_size=4, decay=0.5, selected_features=["ew_mean"]
        )
        extractor_low = ExtractEWStatisticalFeatures(config_low)
        result_low = extractor_low(windowed_data_with_labels)

        # For increasing sequence [1,2,3,4], lower decay concentrates more weight
        # on the last value (4), so it should give a higher mean
        assert result_low["var1_ew_mean"].iloc[0] > result_high["var1_ew_mean"].iloc[0]

    def test_offset_application(self, windowed_data_with_labels):
        """Tests that offset correctly skips initial windows."""
        offset = 2
        config = EWStatisticalConfig(
            is_windowed=True,
            window_size=4,
            decay=0.9,
            offset=offset,
            selected_features=["ew_mean"],
        )
        extractor = ExtractEWStatisticalFeatures(config)

        result = extractor(windowed_data_with_labels)

        # Should have 3 windows (5 - 2 offset)
        assert len(result) == len(windowed_data_with_labels) - offset

        # First label should be the third label from original data
        assert (
            result["label"].iloc[0] == windowed_data_with_labels["label"].iloc[offset]
        )

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

        config = EWStatisticalConfig(is_windowed=False, window_size=4)
        extractor = ExtractEWStatisticalFeatures(config)

        with pytest.raises(ValueError, match="Data is not windowed"):
            extractor(data)

    def test_invalid_config_raises_error(self):
        """Tests that validators raise ValueErrors for invalid configs."""
        with pytest.raises(ValueError, match="Overlap must be in the range"):
            EWStatisticalConfig(overlap=1.0)

        with pytest.raises(ValueError, match="Offset must be a non-negative integer"):
            EWStatisticalConfig(offset=-1)

        with pytest.raises(ValueError, match="Window size must be a positive integer"):
            EWStatisticalConfig(window_size=0)

        with pytest.raises(ValueError, match="Decay must be in the range"):
            EWStatisticalConfig(decay=1.5)

        with pytest.raises(ValueError, match="Decay must be in the range"):
            EWStatisticalConfig(decay=0.0)

    def test_empty_data_raises_error(self):
        """Tests that empty DataFrame raises appropriate error."""
        config = EWStatisticalConfig(is_windowed=True, window_size=4)
        extractor = ExtractEWStatisticalFeatures(config)

        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Input data is empty"):
            extractor(empty_df)

    def test_no_variables_found_raises_error(self):
        """Tests error when no valid variable columns are found."""
        config = EWStatisticalConfig(is_windowed=True, window_size=4)
        extractor = ExtractEWStatisticalFeatures(config)

        invalid_data = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": [4, 5, 6], "label": [100, 101, 102]}
        )

        with pytest.raises(ValueError, match="No variables with pattern 'varX_' found"):
            extractor(invalid_data)

    def test_offset_exceeds_data_length(self, windowed_data_with_labels):
        """Tests that offset larger than data raises error."""
        config = EWStatisticalConfig(is_windowed=True, window_size=4, offset=100)
        extractor = ExtractEWStatisticalFeatures(config)

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

        config = EWStatisticalConfig(
            is_windowed=True,
            window_size=4,
            label_column=None,
            selected_features=["ew_mean"],
        )
        extractor = ExtractEWStatisticalFeatures(config)

        result = extractor(data)

        assert not result.empty
        assert "label" not in result.columns
        assert "var1_ew_mean" in result.columns

    def test_type_error_on_non_dataframe_input(self):
        """Tests that non-DataFrame input raises TypeError."""
        config = EWStatisticalConfig(is_windowed=True, window_size=4)
        extractor = ExtractEWStatisticalFeatures(config)

        with pytest.raises(TypeError, match="Input data must be a pandas DataFrame"):
            extractor([1, 2, 3, 4])

    def test_nan_handling_warning(self, windowed_data_with_labels):
        """Tests that NaN values are handled."""
        config = EWStatisticalConfig(
            is_windowed=True, window_size=4, decay=0.9, selected_features=["ew_mean"]
        )
        extractor = ExtractEWStatisticalFeatures(config)

        # Introduce NaN
        data_with_nan = windowed_data_with_labels.copy()
        data_with_nan.loc[0, "var1_0"] = np.nan

        result = extractor(data_with_nan)

        # Should process but result will have NaN
        assert not result.empty
        assert np.isnan(result["var1_ew_mean"].iloc[0])

    def test_infinite_values_handling(self):
        """Tests that infinite values are replaced in post-processing."""
        data = pd.DataFrame(
            {
                "var1_0": [1.0, 2.0],
                "var1_1": [2.0, 3.0],
                "var1_2": [3.0, 4.0],
                "var1_3": [4.0, np.inf],
                "label": [100, 101],
            }
        )

        config = EWStatisticalConfig(
            is_windowed=True, window_size=4, decay=0.9, selected_features=["ew_max"]
        )
        extractor = ExtractEWStatisticalFeatures(config)

        result = extractor(data)

        # Infinite values should be replaced
        assert not np.isinf(result.select_dtypes(include=[np.number]).values).any()

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

        config = EWStatisticalConfig(
            is_windowed=True,
            window_size=4,
            decay=0.9,
            selected_features=["ew_mean", "ew_std"],
        )
        extractor = ExtractEWStatisticalFeatures(config)

        result = extractor(data)

        assert len(result) == 1
        assert "var1_ew_mean" in result.columns

    def test_eps_parameter_prevents_division_by_zero(self):
        """Tests that eps parameter prevents division by zero."""
        data = pd.DataFrame(
            {
                "var1_0": [5.0],
                "var1_1": [5.0],
                "var1_2": [5.0],
                "var1_3": [5.0],
            }
        )

        config = EWStatisticalConfig(
            is_windowed=True,
            window_size=4,
            decay=0.9,
            eps=1e-10,
            selected_features=["ew_skew", "ew_kurt"],
        )
        extractor = ExtractEWStatisticalFeatures(config)

        result = extractor(data)

        # Should not raise division by zero error
        assert not np.any(np.isnan(result["var1_ew_skew"]))
        assert not np.any(np.isnan(result["var1_ew_kurt"]))

    def test_all_features_class_attribute(self):
        """Tests that FEATURES class attribute contains all expected features."""
        expected_features = [
            "ew_mean",
            "ew_std",
            "ew_skew",
            "ew_kurt",
            "ew_min",
            "ew_1qrt",
            "ew_med",
            "ew_3qrt",
            "ew_max",
        ]
        assert ExtractEWStatisticalFeatures.FEATURES == expected_features

    def test_feature_ordering_consistency(self, multivariate_windowed_data):
        """Tests that feature ordering is consistent across variables."""
        config = EWStatisticalConfig(
            is_windowed=True,
            window_size=4,
            decay=0.9,
            selected_features=["ew_mean", "ew_std", "ew_max"],
        )
        extractor = ExtractEWStatisticalFeatures(config)

        result = extractor(multivariate_windowed_data)

        # Get columns for each variable
        var1_cols = [c for c in result.columns if c.startswith("var1_")]
        var2_cols = [c for c in result.columns if c.startswith("var2_")]

        # Extract feature names
        var1_features = [c.replace("var1_", "") for c in var1_cols]
        var2_features = [c.replace("var2_", "") for c in var2_cols]

        # Should have same features in same order
        assert var1_features == var2_features
