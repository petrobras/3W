"""Tests for preprocessing adapter classes."""

import pytest
from typing import Callable

import pandas as pd
from pydantic import PrivateAttr
import numpy as np

from ThreeWToolkit.core.base_dataset import BaseDataset
from ThreeWToolkit.core.dataset_outputs import DatasetOutputs

from ThreeWToolkit.core.base_preprocessing import (
    BasePreprocessing,
    BasePreprocessingConfig,
)
from ThreeWToolkit.preprocessing import SequentialPreprocessingAdapterConfig


# Create a simple mock preprocessing step for testing
class MockPreprocessingConfig(BasePreprocessingConfig):
    """Mock preprocessing configuration for testing."""

    function: Callable[[pd.DataFrame], pd.DataFrame]
    meta_tag: str | None = None
    wants_tag: str | None = None
    _target: type = PrivateAttr(default_factory=lambda: MockPreprocessing)


class MockPreprocessing(BasePreprocessing):
    """Mock preprocessing step that applies a function to the data."""

    def __init__(self, config: MockPreprocessingConfig):
        self.config: MockPreprocessingConfig = config
        self.fitted = False

    def fit(self, data: BaseDataset) -> None:
        for event in data:
            if self.config.wants_tag is not None:
                if self.config.wants_tag not in event.metadata:
                    raise ValueError(
                        f"Required metadata tag '{self.config.wants_tag}' not found in data."
                    )
        self.fitted = True

    def transform(self, data: DatasetOutputs) -> DatasetOutputs:
        """Apply the configured function to the data."""

        signal = self.config.function(data.signal)
        metadata = data.metadata.copy()
        if self.config.meta_tag is not None:
            metadata[self.config.meta_tag] = True
        return DatasetOutputs(signal=signal, label=data.label, metadata=metadata)


@pytest.fixture
def simple_dataset(mock_dataset_factory) -> BaseDataset:
    """Simple label series for remapping."""
    return mock_dataset_factory(num_sensors=10)


class TestSequentialPreprocessingAdapter:
    """Test sequential preprocessing adapter pipeline."""

    def test_sequential_pipeline_two_steps(self, simple_dataset):
        """Test pipeline with two preprocessing steps."""

        sequential = SequentialPreprocessingAdapterConfig(
            steps=[
                MockPreprocessingConfig(function=lambda df: df + 1, meta_tag="add_one"),
                MockPreprocessingConfig(
                    function=lambda df: df * 2,
                    meta_tag="multiply_two",
                    wants_tag="add_one",
                ),
            ]
        ).build()

        sequential.fit(simple_dataset)

        for step in sequential.steps:
            assert step.fitted, (
                "Each step should be fitted after calling fit on the sequential adapter."
            )

        for event in simple_dataset:
            transformed = sequential.transform(event)
            expected_signal = (event.signal + 1) * 2
            assert np.isclose(transformed.signal, expected_signal).all(), (
                "Transformed signal does not match expected output."
            )

            assert transformed.metadata.get("add_one") is True, (
                "First step should add 'add_one' tag to metadata."
            )
            assert transformed.metadata.get("multiply_two") is True, (
                "Second step should add 'multiply_two' tag to metadata."
            )

        # Invert order of transformations and test again
        sequential = SequentialPreprocessingAdapterConfig(
            steps=[
                MockPreprocessingConfig(
                    function=lambda df: df * 2, meta_tag="multiply_two"
                ),
                MockPreprocessingConfig(
                    function=lambda df: df + 1, wants_tag="multiply_two"
                ),
            ]
        ).build()
        sequential.fit(simple_dataset)

        for step in sequential.steps:
            assert step.fitted, (
                "Each step should be fitted after calling fit on the sequential adapter."
            )

        for event in simple_dataset:
            transformed = sequential.transform(event).signal
            expected_signal = (event.signal * 2) + 1
            assert np.isclose(transformed, expected_signal).all(), (
                "Transformed signal does not match expected output."
            )
