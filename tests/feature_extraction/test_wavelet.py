import pytest
import numpy as np
import pandas as pd


from ThreeWToolkit.core.base_dataset import BaseDataset

from ThreeWToolkit.feature_extraction.adapters import SequentialFeatureAdapterConfig

from ThreeWToolkit.feature_extraction.windowing import WindowConfig
from ThreeWToolkit.feature_extraction.wavelet import WaveletConfig

@pytest.fixture
def simple_dataset(mock_dataset_factory) -> BaseDataset:
    """Simple label series for remapping."""
    return mock_dataset_factory(num_sensors=10)


class TestExtractWaveletFeatures:
    """Unit tests for the ExtractWaveletFeatures class."""

    def test_wavelet_without_windowing(self, simple_dataset):
        """Test that wavelet feature extraction raises an error if data is not windowed."""
        extractor = WaveletConfig(wavelet="haar", level=2, full=False).build()
        with pytest.raises(ValueError):
            extractor.transform(simple_dataset)

