"""Tests for feature extraction adapter classes."""

import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

from ThreeWToolkit.feature_extraction import (
    SequentialFeatureAdapter,
    SequentialFeatureAdapterConfig,
    ConcatFeatureAdapter,
    ConcatFeatureAdapterConfig,
    StatisticalFeatures,
    StatisticalConfig,
    WaveletFeatures,
    WaveletConfig,
)


# Module-level fixtures

@pytest.fixture
def windowed_data():
    """Windowed data with multi-index for feature extraction."""
    # TODO: Create proper windowed data structure
    return pd.DataFrame({
        "sensor1": np.random.randn(100),
        "sensor2": np.random.randn(100),
    })


# Tests for SequentialFeatureAdapter


class TestSequentialFeatureAdapter:
    """Test sequential feature adapter."""

    def test_sequential_two_extractors(self, windowed_data):
        """Test sequential application of two feature extractors."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    def test_sequential_multiple_extractors(self):
        """Test sequential application of multiple feature extractors."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    @pytest.mark.parametrize(
        "num_extractors",
        [1, 2, 3],
    )
    def test_various_pipeline_lengths(self, num_extractors):
        """Test pipelines of various lengths."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")


class TestSequentialFeatureAdapterDataFlow:
    """Test data flow through sequential adapter."""

    def test_data_transformation_chain(self):
        """Test that data is properly transformed through the chain."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    def test_output_from_final_extractor(self):
        """Test that output comes from the final extractor."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")


# Tests for ConcatFeatureAdapter


class TestConcatFeatureAdapter:
    """Test concatenation feature adapter."""

    def test_concat_two_extractors(self, windowed_data):
        """Test concatenation of two feature extractors."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    def test_concat_statistical_and_wavelet(self):
        """Test concatenating statistical and wavelet features."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    @pytest.mark.parametrize(
        "num_extractors",
        [2, 3, 4],
    )
    def test_various_extractor_counts(self, num_extractors):
        """Test concatenation with different numbers of extractors."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")


class TestConcatFeatureAdapterOutput:
    """Test output structure of concat adapter."""

    def test_output_concatenation_axis(self):
        """Test that features are concatenated along correct axis."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    def test_label_preservation(self):
        """Test that labels are preserved from first non-None source."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    def test_feature_name_uniqueness(self):
        """Test that concatenated features have unique names."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")


class TestFeatureAdaptersEdgeCases:
    """Test edge cases for feature adapters."""

    def test_empty_extractor_list(self):
        """Test behavior with empty extractor list."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    def test_single_extractor(self):
        """Test adapter with single extractor."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")
