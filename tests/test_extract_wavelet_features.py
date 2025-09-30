import pytest
import numpy as np
import pandas as pd

from ThreeWToolkit.feature_extraction.extract_wavelet_features import (
    ExtractWaveletFeatures,
    WaveletConfig,
)


# A mock for the windowing function is needed for the tests to run standalone
def windowing(X, window_size, overlap):
    step = int(window_size * (1 - overlap))
    if step == 0:
        return pd.DataFrame()  # Return empty if stride would be 0
    windows = []
    for i in range(0, len(X) - window_size + 1, step):
        windows.append(X.iloc[i : i + window_size].values)
    return pd.DataFrame(windows)


class TestExtractWaveletFeatures:
    """Unit tests for the ExtractWaveletFeatures class."""

    @pytest.fixture
    def data_with_labels(self):
        """Provides sample data (X) and corresponding labels (y)."""
        index = pd.to_datetime([f"2025-01-01 {h:02d}:00:00" for h in range(20)])
        tags = pd.DataFrame({"signal": np.arange(1.0, 21.0)}, index=index)
        y = pd.Series(100 + np.arange(20), index=index, name="target")
        return tags, y

    def test_basic_extraction_and_y_alignment(self, data_with_labels):
        """Tests that both features (X) and labels (y) are correctly windowed and aligned."""
        tags, y = data_with_labels
        config = WaveletConfig(level=2, overlap=0.5)  # window_size=4, stride=2
        extractor = ExtractWaveletFeatures(config)

        X_out, y_out = extractor(tags=tags, y=y)

        assert not X_out.empty and not y_out.empty
        assert len(X_out) == len(y_out)
        pd.testing.assert_index_equal(X_out.index, y_out.index)

        # First window covers original indices 0-3, so its index is y.index[3]
        # and its label value should be y.iloc[3]
        assert y_out.index[0] == y.index[3]
        assert y_out.iloc[0] == y.iloc[3]

        # Second window covers original indices 2-5, so its index is y.index[5]
        # and its label value should be y.iloc[5]
        assert y_out.index[1] == y.index[5]
        assert y_out.iloc[1] == y.iloc[5]

    def test_offset_is_applied_to_y(self, data_with_labels):
        """Tests that the offset parameter is correctly applied to both X and y."""
        tags, y = data_with_labels
        config = WaveletConfig(level=2, overlap=0.5, offset=5)
        extractor = ExtractWaveletFeatures(config)
        X_out, y_out = extractor(tags=tags, y=y)

        # The first window starts after the offset, covering original indices [5, 6, 7, 8].
        # The corresponding label should be the original y at index 8.
        expected_first_label = y.iloc[8]

        assert y_out.iloc[0] == expected_first_label

    def test_insufficient_data_with_y(self, data_with_labels):
        """Tests that for insufficient data, both returned X and y are empty."""
        tags, y = data_with_labels
        short_tags, short_y = tags.head(3), y.head(3)

        config = WaveletConfig(level=2)  # window_size=4
        extractor = ExtractWaveletFeatures(config)
        X_out, y_out = extractor(tags=short_tags, y=short_y)

        assert X_out.empty
        assert y_out.empty
        assert isinstance(y_out, pd.Series)

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

    def test_output_column_names(self, data_with_labels):
        """Tests that the output DataFrame has correctly formatted column names."""
        # Get the input data and its column names from the fixture
        tags, y = data_with_labels
        input_cols = tags.columns

        config = WaveletConfig(level=2)
        extractor = ExtractWaveletFeatures(config)

        X_out, _ = extractor(tags=tags, y=y)

        feature_names = extractor.feat_names

        expected_columns = [
            f"{col}_{feat}" for feat in feature_names for col in input_cols
        ]

        assert list(X_out.columns) == expected_columns

    def test_handles_empty_windows_from_toolkit_function(
        self, monkeypatch, data_with_labels
    ):
        """
        Tests the early-return path if the windowing function returns empty for all columns.
        """

        def mock_windowing(*args, **kwargs):
            return pd.DataFrame()

        monkeypatch.setattr(
            "ThreeWToolkit.feature_extraction.extract_wavelet_features.windowing",
            mock_windowing,
        )

        tags, y = data_with_labels

        config = WaveletConfig(level=2)
        extractor = ExtractWaveletFeatures(config)
        X_out, y_out = extractor(tags=tags, y=y)

        assert X_out.empty
        assert y_out.empty

    def test_handles_mixed_success_from_windowing(self, monkeypatch):
        """Tests the mixed scenario where one column succeeds and one fails."""

        def mock_windowing_mixed(X: pd.Series, *args, **kwargs):
            if X.name == "signal_ok":
                return pd.DataFrame(np.random.rand(5, kwargs.get("window_size", 4)))
            else:
                return pd.DataFrame()

        monkeypatch.setattr(
            "ThreeWToolkit.feature_extraction.extract_wavelet_features.windowing",
            mock_windowing_mixed,
        )

        tags = pd.DataFrame(
            {"signal_ok": np.arange(20), "signal_fail": np.arange(20, 40)}
        )
        y = pd.Series(np.arange(20))
        config = WaveletConfig(level=2)
        extractor = ExtractWaveletFeatures(config)
        X_out, y_out = extractor(tags=tags, y=y)

        assert not X_out.empty
        assert "signal_ok_A1" in X_out.columns
        assert "signal_fail_A1" not in X_out.columns
        assert len(X_out) == len(y_out)

    def test_raises_error_if_y_is_not_provided(self):
        """Tests that a ValueError is raised if the 'y' labels are not passed."""
        tags = pd.DataFrame({"signal": np.arange(20)})
        config = WaveletConfig(level=2, overlap=0.5)
        extractor = ExtractWaveletFeatures(config)

        with pytest.raises(ValueError, match="The 'y' series .* must be provided"):
            extractor(tags=tags, y=None)
