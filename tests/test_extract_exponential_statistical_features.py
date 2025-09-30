# testes/test_extract_ew_statistical_features.py
import pytest
import numpy as np
import pandas as pd

from ThreeWToolkit.feature_extraction.extract_exponential_statistics_features import (
    ExtractEWStatisticalFeatures,
    EWStatisticalConfig,
)


class TestExtractEWStatisticalFeatures:
    """Unit tests for the ExtractEWStatisticalFeatures class."""

    @pytest.fixture
    def data_with_labels(self):
        """Provides sample data (X) and corresponding labels (y)."""
        tags = pd.DataFrame({"signal": np.arange(1.0, 21.0)})
        y = pd.Series(100 + tags.index.to_series(), index=tags.index, name="target")
        return tags, y

    def test_y_alignment_and_basic_extraction(self, data_with_labels):
        """Tests that features (X) and labels (y) are correctly aligned."""
        tags, y = data_with_labels
        config = EWStatisticalConfig(window_size=10, decay=0.9, overlap=0.5)  # stride=5
        extractor = ExtractEWStatisticalFeatures(config)

        X_out, y_out = extractor(tags=tags, y=y)

        assert not X_out.empty and not y_out.empty
        assert len(X_out) == len(y_out)
        pd.testing.assert_index_equal(X_out.index, y_out.index)

        # First window ends at index 9, label should be y[9]=109
        assert y_out.iloc[0] == y.iloc[9]
        # Second window ends at index 14, label should be y[14]=114
        assert y_out.iloc[1] == y.iloc[14]

    def test_offset_is_applied_to_y(self, data_with_labels):
        """Tests that the offset parameter is correctly applied to both X and y."""
        tags, y = data_with_labels
        config = EWStatisticalConfig(window_size=10, decay=0.9, overlap=0.0, offset=5)
        extractor = ExtractEWStatisticalFeatures(config)
        X_out, y_out = extractor(tags=tags, y=y)

        # First window ends at index 14 (5 offset + 10 size - 1). Label should be y[14]=114.
        assert y_out.iloc[0] == y.iloc[14]

    def test_multiple_columns(self, data_with_labels):
        """Tests the extractor works for a DataFrame with multiple columns."""
        tags, y = data_with_labels
        tags["signal2"] = tags["signal"] * 2  # Add a second column

        config = EWStatisticalConfig(window_size=10, decay=0.9, overlap=0.8)
        extractor = ExtractEWStatisticalFeatures(config)
        X_out, y_out = extractor(tags=tags, y=y)

        assert "signal_ew_mean" in X_out.columns
        assert "signal2_ew_mean" in X_out.columns
        assert len(X_out) == len(y_out)

    def test_insufficient_data_with_y(self):
        """Tests that for insufficient data, both returned X and y are empty."""
        tags = pd.DataFrame({"signal": np.arange(5.0)})
        y = pd.Series(np.arange(5), name="target")

        config = EWStatisticalConfig(window_size=10, decay=0.9)
        extractor = ExtractEWStatisticalFeatures(config)
        X_out, y_out = extractor(tags=tags, y=y)

        assert X_out.empty
        assert y_out.empty
        assert isinstance(y_out, pd.Series)

    def test_raises_error_if_y_is_not_provided(self):
        """Tests that a ValueError is raised if 'y' is not passed."""
        tags = pd.DataFrame({"signal": np.arange(20)})
        config = EWStatisticalConfig(window_size=10, decay=0.9)
        extractor = ExtractEWStatisticalFeatures(config)

        with pytest.raises(ValueError, match="The 'y' series .* must be provided"):
            extractor(tags=tags, y=None)

    def test_invalid_config_raises_error(self):
        """Tests that Pydantic validators raise ValueErrors for invalid configs."""
        with pytest.raises(ValueError, match="Overlap must be in the range"):
            EWStatisticalConfig(window_size=10, decay=0.9, overlap=1.0)
        with pytest.raises(ValueError, match="Offset must be a non-negative integer"):
            EWStatisticalConfig(window_size=10, decay=0.9, offset=-1)
        with pytest.raises(ValueError, match="Epsilon .* must be positive"):
            EWStatisticalConfig(window_size=10, decay=0.9, eps=0)

    def test_output_column_names(self):
        """Tests that the output DataFrame has correctly formatted column names."""
        input_cols = ["alpha", "beta"]
        tags = pd.DataFrame(np.random.rand(10, 2), columns=input_cols)
        y = pd.Series(np.arange(10))

        config = EWStatisticalConfig(window_size=5, decay=0.9)
        extractor = ExtractEWStatisticalFeatures(config)
        X_out, _ = extractor(tags=tags, y=y)

        feature_suffixes = ExtractEWStatisticalFeatures.FEATURES
        expected_columns = [
            f"{col}_{feat}" for feat in feature_suffixes for col in input_cols
        ]
        assert list(X_out.columns) == expected_columns

    def test_handles_mixed_success_from_windowing(self, monkeypatch):
        """Tests the mixed scenario where one column succeeds and one fails."""

        def mock_windowing_mixed(X: pd.Series, *args, **kwargs):
            if X.name == "s1":
                return pd.DataFrame(np.random.rand(3, kwargs.get("window_size", 10)))
            else:
                return pd.DataFrame()

        monkeypatch.setattr(
            "ThreeWToolkit.feature_extraction.extract_exponential_statistics_features.windowing",
            mock_windowing_mixed,
        )

        tags = pd.DataFrame({"s1": np.arange(20), "s2": np.arange(20, 40)})
        y = pd.Series(np.arange(20))
        config = EWStatisticalConfig(window_size=10, decay=0.9, overlap=0.5)
        extractor = ExtractEWStatisticalFeatures(config)
        X_out, y_out = extractor(tags=tags, y=y)

        assert not X_out.empty
        assert "s1_ew_mean" in X_out.columns
        assert "s2_ew_mean" not in X_out.columns
        assert len(X_out) == len(y_out)
