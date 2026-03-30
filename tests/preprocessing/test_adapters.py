"""Tests for preprocessing adapter classes."""

import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

from ThreeWToolkit.preprocessing import (
    SequentialPreprocessingAdapter,
    SequentialPreprocessingAdapterConfig,
    ImputeMissing,
    ImputeMissingConfig,
    Normalize,
    NormalizeConfig,
)


# Module-level fixtures

@pytest.fixture
def df_with_missing_and_unnormalized():
    """DataFrame with missing values and unnormalized data."""
    return pd.DataFrame({
        "a": [1.0, np.nan, 3.0],
        "b": [10.0, 20.0, 30.0],
    })


# Tests for SequentialPreprocessingAdapter


class TestSequentialPreprocessingAdapter:
    """Test sequential preprocessing adapter pipeline."""

    def test_sequential_pipeline_two_steps(self, df_with_missing_and_unnormalized):
        """Test pipeline with two preprocessing steps."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    def test_sequential_pipeline_multiple_steps(self):
        """Test pipeline with multiple preprocessing steps."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    def test_fit_transform_pipeline(self, df_with_missing_and_unnormalized):
        """Test fit() and transform() methods on pipeline."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    @pytest.mark.parametrize(
        "num_steps",
        [1, 2, 3, 4],
    )
    def test_various_pipeline_lengths(self, num_steps):
        """Test pipelines of various lengths."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")


class TestSequentialPreprocessingAdapterEdgeCases:
    """Test edge cases for sequential adapter."""

    def test_empty_pipeline(self):
        """Test behavior with empty preprocessing list."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    def test_single_step_pipeline(self):
        """Test pipeline with only one preprocessing step."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    def test_pipeline_data_flow(self):
        """Test that data flows correctly through pipeline stages."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")
