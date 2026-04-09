import pytest


from ThreeWToolkit.core.base_dataset import BaseDataset

from ThreeWToolkit.feature_extraction.adapters import SequentialFeatureAdapterConfig

from ThreeWToolkit.feature_extraction.windowing import WindowingConfig
from ThreeWToolkit.feature_extraction.wavelet import WaveletConfig

_NUM_SENSORS = 10


@pytest.fixture
def simple_dataset(mock_dataset_factory) -> BaseDataset:
    """Simple label series for remapping."""
    return mock_dataset_factory(num_sensors=_NUM_SENSORS)


class TestExtractWaveletFeatures:
    """Unit tests for the ExtractWaveletFeatures class."""

    def test_wavelet_without_windowing(self, simple_dataset):
        """Test that wavelet feature extraction raises an error if data is not windowed."""
        extractor = WaveletConfig(wavelet="haar", level=2, full=False).build()

        for event in simple_dataset:
            with pytest.raises(ValueError):
                extractor.transform(event)

    @pytest.mark.parametrize("level", [3, 4, 5, 6, 7, 8])
    @pytest.mark.parametrize("overlap", [0.25, 0.5, 0.75])
    @pytest.mark.parametrize("wavelet", ["haar", "db4", "sym5"])
    @pytest.mark.parametrize("full", [True, False])
    def test_wavelet_with_windowing(
        self, simple_dataset, level, overlap, wavelet, full
    ):
        """Test that wavelet feature extraction works correctly with windowed data."""

        feature_extractor = SequentialFeatureAdapterConfig(
            steps=[
                WindowingConfig(
                    window_size=2**level,
                    overlap=overlap,
                    pad_start=True,
                    pad_last_window=True,
                ),
                WaveletConfig(wavelet=wavelet, level=level, full=full),
            ]
        ).build()

        if full:
            bases = ["A", "D"]  # A for approximation, D for detail coefficients
            names = [
                f"{base}{lvl}" for base in bases for lvl in range(1, level + 1)
            ] + ["A0"]
        else:
            bases = ["D"]  # Only detail coefficients
            names = (
                [f"A{level}"]
                + [f"{base}{lvl}" for base in bases for lvl in range(1, level + 1)]
                + ["A0"]
            )

        expected_names = [
            f"{feat}_sensor_{var}" for feat in names for var in range(_NUM_SENSORS)
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
