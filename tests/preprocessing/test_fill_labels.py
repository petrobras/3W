"""Tests for FillLabels preprocessing class."""

import pytest
import numpy as np

from ThreeWToolkit.core.base_dataset import BaseDataset
from ThreeWToolkit.core.dataset_outputs import DatasetOutputs
from ThreeWToolkit.dataset.transformed_dataset import TransformedDataset

from ThreeWToolkit.preprocessing import FillLabelsConfig


@pytest.fixture
def simple_dataset(mock_dataset_factory) -> BaseDataset:
    """Simple dataset for normalization tests."""
    return mock_dataset_factory(num_sensors=10)


@pytest.fixture
def dataset_with_nan_labels(simple_dataset):
    """Dataset with some NaN values in labels."""
    nan_ratio = 0.2

    def _transform(data: DatasetOutputs) -> DatasetOutputs:
        if data.label is None:
            return data
        _label = data.label.copy()
        num_labels = len(_label)
        nan_mask = np.random.rand(num_labels) < nan_ratio
        _label[nan_mask] = np.nan
        return DatasetOutputs(signal=data.signal, label=_label, metadata=data.metadata)

    return TransformedDataset(simple_dataset, _transform)


# Tests for FillLabels functionality
class TestFillLabelsStrategies:
    """Test different fill strategies."""

    def test_constant_fill_strategy(self, dataset_with_nan_labels):
        """Test filling with a constant value."""

        fill_value = 999
        fill_labels = FillLabelsConfig(
            fill_method="constant", fill_value=fill_value
        ).build()
        fill_labels.fit(dataset_with_nan_labels)  # Noop

        for original in dataset_with_nan_labels:
            filled = fill_labels.transform(original)

            assert original.label is not None, "Original labels should not be None."
            assert filled.label is not None, "Filled labels should not be None."

            assert (
                not filled.label.isna().any()
            ), "Filled labels should not contain NaN values."

            nan_mask = original.label.isna()
            assert (
                filled.label[nan_mask] == fill_value
            ).all(), "Filled values should match the specified constant fill value."

    @pytest.mark.parametrize(
        "fill_method",
        ["nearest", "ffill", "bfill"],
    )
    def test_nearest_fill_strategy(self, dataset_with_nan_labels, fill_method):
        """Test nearest interpolation + bfill + ffill strategy."""

        fill_labels = FillLabelsConfig(fill_method=fill_method).build()

        fill_labels.fit(dataset_with_nan_labels)  # Noop

        for original in dataset_with_nan_labels:
            filled = fill_labels.transform(original)

            assert original.label is not None, "Original labels should not be None."
            assert filled.label is not None, "Filled labels should not be None."

            assert (
                not filled.label.isna().any()
            ), "Filled labels should not contain NaN values."

            original_na_mask = original.label.isna()
            nan_indices = original_na_mask[original_na_mask].index
            non_nan_indices = original_na_mask[~original_na_mask].index

            for idx in nan_indices:
                nearest_left = max(non_nan_indices[non_nan_indices < idx], default=None)
                nearest_right = min(
                    non_nan_indices[non_nan_indices > idx], default=None
                )

                if nearest_left is None:  # always bfill
                    assert filled.label[idx] == original.label[nearest_right]
                elif nearest_right is None:  # always ffill
                    assert filled.label[idx] == original.label[nearest_left]
                else:
                    if fill_method == "ffill":
                        assert (
                            filled.label[idx] == original.label[nearest_left]
                        ), "Filled value should match nearest left non-NaN value for ffill strategy."
                    elif fill_method == "bfill":
                        assert (
                            filled.label[idx] == original.label[nearest_right]
                        ), "Filled value should match nearest right non-NaN value for bfill strategy."
                    else:  # nearest
                        assert (filled.label[idx] == original.label[nearest_left]) or (
                            filled.label[idx] == original.label[nearest_right]
                        ), "Filled value should match either nearest left or right non-NaN value."
