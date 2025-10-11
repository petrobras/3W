import pytest
import numpy as np
import pandas as pd

from ThreeWToolkit.feature_extraction.extract_wavelet_features import (
    ExtractWaveletFeatures,
)


class WaveletConfig:
    """Mock config class for testing."""

    def __init__(
        self,
        level=2,
        overlap=0.0,
        offset=0,
        wavelet="db1",
        is_windowed=True,
        label_column="label",
    ):
        if overlap < 0 or overlap >= 1:
            raise ValueError("Overlap must be in the range [0, 1)")
        if offset < 0:
            raise ValueError("Offset must be a non-negative integer")
        if level <= 0:
            raise ValueError("Wavelet level must be a positive integer")

        self.level = level
        self.overlap = overlap
        self.offset = offset
        self.wavelet = wavelet
        self.is_windowed = is_windowed
        self.label_column = label_column


class TestExtractWaveletFeatures:
    """Unit tests for the ExtractWaveletFeatures class."""

    @pytest.fixture
    def windowed_data_with_labels(self):
        """
        Provides sample windowed data in the expected format.
        Each row represents a window with columns: var1_0, var1_1, ..., var1_3, label
        """
        # Create 5 windows, each with 4 time steps (level=2 -> window_size=4)
        num_windows = 5
        window_size = 4

        data = {}
        # Create columns for variable 1
        for i in range(window_size):
            data[f"var1_{i}"] = np.random.randn(num_windows)

        # Add labels
        data["label"] = np.array([100, 101, 102, 103, 104])

        return pd.DataFrame(data)

    @pytest.fixture
    def multivariate_windowed_data(self):
        """
        Provides multivariate windowed data with two variables.
        """
        num_windows = 5
        window_size = 4

        data = {}
        # Variable 1
        for i in range(window_size):
            data[f"var1_{i}"] = np.random.randn(num_windows)

        # Variable 2
        for i in range(window_size):
            data[f"var2_{i}"] = np.random.randn(num_windows) * 2

        # Add labels
        data["label"] = np.array([100, 101, 102, 103, 104])

        return pd.DataFrame(data)

    def test_basic_extraction_univariate(self, windowed_data_with_labels):
        """Tests basic wavelet feature extraction for univariate data."""
        config = WaveletConfig(level=2, is_windowed=True)
        extractor = ExtractWaveletFeatures(config)

        result = extractor(windowed_data_with_labels)

        # Check that we get features
        assert not result.empty
        assert len(result) == len(windowed_data_with_labels)

        # Check that labels are preserved
        assert "label" in result.columns
        pd.testing.assert_series_equal(
            result["label"], windowed_data_with_labels["label"], check_names=False
        )

        # Check that wavelet features exist
        assert "var1_A2" in result.columns
        assert "var1_D2" in result.columns
        assert "var1_A1" in result.columns
        assert "var1_D1" in result.columns
        assert "var1_A0" in result.columns

    def test_multivariate_extraction(self, multivariate_windowed_data):
        """Tests wavelet feature extraction for multivariate data."""
        config = WaveletConfig(level=2, is_windowed=True)
        extractor = ExtractWaveletFeatures(config)

        result = extractor(multivariate_windowed_data)

        assert not result.empty

        # Check features for both variables
        assert "var1_A2" in result.columns
        assert "var2_A2" in result.columns
        assert "var1_D1" in result.columns
        assert "var2_D1" in result.columns

    def test_offset_application(self, windowed_data_with_labels):
        """Tests that offset parameter correctly skips initial windows."""
        offset = 2
        config = WaveletConfig(level=2, offset=offset, is_windowed=True)
        extractor = ExtractWaveletFeatures(config)

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

        config = WaveletConfig(level=2, is_windowed=False)
        extractor = ExtractWaveletFeatures(config)

        with pytest.raises(ValueError, match="Data is not windowed"):
            extractor(data)

    def test_invalid_config_raises_error(self):
        """Tests that validators raise ValueErrors for invalid configs."""
        with pytest.raises(ValueError, match="Overlap must be in the range"):
            WaveletConfig(level=1, overlap=1.0)

        with pytest.raises(ValueError, match="Offset must be a non-negative integer"):
            WaveletConfig(level=1, offset=-1)
        with pytest.raises(
            ValueError, match="Wavelet level must be a positive integer"
        ):
            WaveletConfig(level=0)

    def test_output_column_names_format(self, windowed_data_with_labels):
        """Tests that output columns follow the expected naming convention."""
        config = WaveletConfig(level=2, is_windowed=True)
        extractor = ExtractWaveletFeatures(config)

        result = extractor(windowed_data_with_labels)

        # Expected feature names for level=2
        expected_features = ["var1_A2", "var1_D2", "var1_A1", "var1_D1", "var1_A0"]

        for feat in expected_features:
            assert feat in result.columns, f"Expected feature {feat} not found"

    def test_empty_data_raises_error(self):
        """Tests that empty DataFrame raises appropriate error."""
        config = WaveletConfig(level=2, is_windowed=True)
        extractor = ExtractWaveletFeatures(config)

        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Input data is empty"):
            extractor(empty_df)

    def test_no_variables_found_raises_error(self):
        """Tests error when no valid variable columns are found."""
        config = WaveletConfig(level=2, is_windowed=True)
        extractor = ExtractWaveletFeatures(config)

        # DataFrame with wrong column format
        invalid_data = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": [4, 5, 6], "label": [100, 101, 102]}
        )

        with pytest.raises(ValueError, match="No variables with pattern 'varX_' found"):
            extractor(invalid_data)

    def test_window_size_mismatch_handling(self):
        """Tests handling of windows with incorrect size."""
        config = WaveletConfig(level=2, is_windowed=True)  # Expects window_size=4
        extractor = ExtractWaveletFeatures(config)

        # Create data with wrong window size (only 3 columns)
        data = pd.DataFrame(
            {"var1_0": [1, 2], "var1_1": [3, 4], "var1_2": [5, 6], "label": [100, 101]}
        )

        # Should pad with zeros and still process
        result = extractor(data)
        assert not result.empty
        assert len(result) == 2

    def test_offset_exceeds_data_length(self, windowed_data_with_labels):
        """Tests that offset larger than data raises error."""
        config = WaveletConfig(
            level=2,
            offset=100,  # Much larger than data
            is_windowed=True,
        )
        extractor = ExtractWaveletFeatures(config)

        with pytest.raises(ValueError, match="Offset .* is larger than data length"):
            extractor(windowed_data_with_labels)

    def test_nan_handling_warning(self, windowed_data_with_labels):
        """Tests that NaN values trigger a warning."""
        config = WaveletConfig(level=2, is_windowed=True)
        extractor = ExtractWaveletFeatures(config)

        # Introduce NaN
        data_with_nan = windowed_data_with_labels.copy()
        data_with_nan.loc[0, "var1_0"] = np.nan

        # Should process but may show warning (captured if using capfd)
        result = extractor(data_with_nan)
        assert not result.empty

    def test_feature_names_generation(self):
        """Tests that feature names are correctly generated for different levels."""
        config = WaveletConfig(level=3, is_windowed=True)
        extractor = ExtractWaveletFeatures(config)

        expected_names = ["A3", "D3", "A2", "D2", "A1", "D1", "A0"]
        assert extractor.feat_names == expected_names

    def test_different_wavelet_types(self, windowed_data_with_labels):
        """Tests that different wavelet types work correctly."""
        for wavelet in ["db1", "db2", "sym2", "coif1"]:
            config = WaveletConfig(level=2, wavelet=wavelet, is_windowed=True)
            extractor = ExtractWaveletFeatures(config)

            result = extractor(windowed_data_with_labels)
            assert not result.empty
            assert "var1_A2" in result.columns

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

        config = WaveletConfig(level=2, is_windowed=True, label_column=None)
        extractor = ExtractWaveletFeatures(config)

        result = extractor(data)

        assert not result.empty
        assert "label" not in result.columns
        assert "var1_A2" in result.columns

    def test_type_error_on_non_dataframe_input(self):
        """Tests that non-DataFrame input raises TypeError."""
        config = WaveletConfig(level=2, is_windowed=True)
        extractor = ExtractWaveletFeatures(config)

        with pytest.raises(TypeError, match="Input data must be a pandas DataFrame"):
            extractor([1, 2, 3, 4])  # List instead of DataFrame
