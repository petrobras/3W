import pytest


from ThreeWToolkit.core.base_dataset import BaseDataset

from ThreeWToolkit.feature_extraction.adapters import SequentialFeatureAdapterConfig

from ThreeWToolkit.feature_extraction.windowing import WindowingConfig
from ThreeWToolkit.feature_extraction.exponential_statistics import (
    EWStatisticalConfig,
    _AVAILABLE_FEATURES,
)

_NUM_SENSORS = 10


@pytest.fixture
def simple_dataset(mock_dataset_factory) -> BaseDataset:
    """Simple label series for remapping."""
    return mock_dataset_factory(num_sensors=_NUM_SENSORS)


class TestExtractStatisticalFeatures:
    """Unit tests for the ExtractStatisticalFeatures class."""

    def test_ew_stats_without_windowing(self, simple_dataset):
        """Test that statistical feature extraction raises an error if data is not windowed."""
        extractor = EWStatisticalConfig().build()

        for event in simple_dataset:
            with pytest.raises(ValueError):
                extractor.transform(event)

    @pytest.mark.parametrize("window_size", [64, 128, 256])
    @pytest.mark.parametrize("decay", [0.9, 0.95, 0.99])
    @pytest.mark.parametrize(
        "features",
        [
            ["ew_mean", "ew_std"],
            ["ew_skew", "ew_kurt"],
            ["ew_min", "ew_1qrt", "ew_med", "ew_3qrt", "ew_max"],
            list(_AVAILABLE_FEATURES),
        ],
    )
    def test_ew_stats_with_windowing(
        self, simple_dataset, window_size, features, decay
    ):
        """Test that feature extraction works correctly with windowed data."""

        feature_extractor = SequentialFeatureAdapterConfig(
            steps=[
                WindowingConfig(
                    window_size=window_size,
                    overlap=0.5,
                    pad_start=True,
                    pad_last_window=True,
                ),
                EWStatisticalConfig(
                    window_size=window_size, decay=decay, features=features
                ),
            ]
        ).build()

        expected_names = [
            f"{feat}_sensor_{var}" for feat in features for var in range(_NUM_SENSORS)
        ]

        for event in simple_dataset.events:
            transformed = feature_extractor.transform(event)

            assert (
                not transformed.signal.isna().any().any()
            ), "Transformed data should not contain NaN values."
            assert (
                transformed.signal.shape[0] > 0
            ), "Transformed data should have at least one row."
            assert transformed.signal.shape[1] == len(
                expected_names
            ), f"Expected {len(expected_names)} features,\
                    but got {transformed.signal.shape[1]}."
            assert set(transformed.signal.columns) == set(
                expected_names
            ), "Expected feature names do not match actual feature names."
