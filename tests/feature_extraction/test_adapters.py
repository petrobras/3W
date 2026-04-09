"""Tests for feature extraction adapter classes."""

import pytest
import pandas as pd

from pydantic import PrivateAttr
from typing import Callable


from ThreeWToolkit.core.base_dataset import BaseDataset
from ThreeWToolkit.core.dataset_outputs import DatasetOutputs

from ThreeWToolkit.core.base_feature_extractor import (
    BaseFeatureExtractor,
    BaseFeatureExtractorConfig,
)

from ThreeWToolkit.feature_extraction.adapters import (
    SequentialFeatureAdapterConfig,
    ConcatFeatureAdapterConfig,
)


# Create a simple mock preprocessing step for testing
class MockFeatureConfig(BaseFeatureExtractorConfig):
    """Mock preprocessing configuration for testing."""

    function: Callable[[pd.DataFrame], pd.DataFrame]
    meta_tag: str | None = None
    wants_tag: str | None = None
    _target: type = PrivateAttr(default_factory=lambda: MockFeature)


class MockFeature(BaseFeatureExtractor):
    """Mock preprocessing step that applies a function to the data."""

    def __init__(self, config: MockFeatureConfig):
        self.config: MockFeatureConfig = config

    def transform(self, data: DatasetOutputs) -> DatasetOutputs:
        """Apply the configured function to the data."""
        assert (
            self.config.wants_tag is None or self.config.wants_tag in data.metadata
        ), f"Required metadata tag '{self.config.wants_tag}' not found in data."

        signal = self.config.function(data.signal)
        metadata = data.metadata.copy()
        if self.config.meta_tag is not None:
            metadata[self.config.meta_tag] = True
        return DatasetOutputs(signal=signal, label=data.label, metadata=metadata)


@pytest.fixture
def simple_dataset(mock_dataset_factory) -> BaseDataset:
    """Simple label series for remapping."""
    return mock_dataset_factory(num_sensors=10)


# Tests for SequentialFeatureAdapter
class TestSequentialFeatureAdapter:
    """Test data flow through ConcatFeatureAdapter."""

    def test_empty_sequential_extractor(self, simple_dataset):
        """Test sequential application of two feature extractors."""
        features = SequentialFeatureAdapterConfig(steps=[]).build()

        for event in simple_dataset:
            transformed = features.transform(event)
            assert (
                event == transformed
            ), "Data should be unchanged when no steps are applied."

    def test_sequential_single_extractor(self, simple_dataset):
        """Test application of single feature extractor."""
        features = SequentialFeatureAdapterConfig(
            steps=[
                MockFeatureConfig(function=lambda df: df * 2, meta_tag="multiply_two"),
            ]
        ).build()
        for event in simple_dataset:
            transformed = features.transform(event)
            assert (
                (transformed.signal == event.signal * 2).all().all()
            ), "Transformation should be applied."
            assert (
                "multiply_two" in transformed.metadata
            ), "Metadata tag should be added."

    def test_sequential_multiple_extractors(self, simple_dataset):
        """Test sequential application of two feature extractors."""
        features = SequentialFeatureAdapterConfig(
            steps=[
                MockFeatureConfig(function=lambda df: df + 1, meta_tag="add_one"),
                MockFeatureConfig(
                    function=lambda df: df * 2,
                    meta_tag="multiply_two",
                    wants_tag="add_one",
                ),
            ]
        ).build()

        for event in simple_dataset:
            transformed = features.transform(event)
            expected_signal = (event.signal + 1) * 2
            assert (
                (transformed.signal == expected_signal).all().all()
            ), "Transformations should be applied in sequence."
            assert (
                "add_one" in transformed.metadata
            ), "First metadata tag should be added."
            assert (
                "multiply_two" in transformed.metadata
            ), "Second metadata tag should be added."


class TestConcatFeatureAdapter:
    """Test data flow through ConcatFeatureAdapter."""

    def test_concat_single_extractor(self, simple_dataset):
        """Test application of single feature extractor."""
        features = ConcatFeatureAdapterConfig(
            steps=[
                MockFeatureConfig(function=lambda df: df * 2, meta_tag="multiply_two"),
            ]
        ).build()
        for event in simple_dataset:
            transformed = features.transform(event)
            assert (
                (transformed.signal == event.signal * 2).all().all()
            ), "Transformation should be applied."
            assert (
                "multiply_two" in transformed.metadata
            ), "Metadata tag should be added."

    def test_concat_multiple_extractors(self, simple_dataset):
        """Test sequential application of two feature extractors."""
        features = ConcatFeatureAdapterConfig(
            steps=[
                MockFeatureConfig(function=lambda df: df + 1, meta_tag="add_one"),
                MockFeatureConfig(
                    function=lambda df: df * 2,
                    meta_tag="multiply_two",
                ),
            ]
        ).build()

        for event in simple_dataset:
            transformed = features.transform(event)
            expected_signal = pd.concat([event.signal + 1, event.signal * 2], axis=1)
            assert (
                (transformed.signal == expected_signal).all().all()
            ), "Transformations should be applied and concatenated."
            assert (
                "add_one" in transformed.metadata
            ), "First metadata tag should be added."
            assert (
                "multiply_two" in transformed.metadata
            ), "Second metadata tag should be added."
