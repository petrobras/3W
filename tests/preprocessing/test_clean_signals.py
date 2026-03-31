"""Tests for CleanSignals preprocessing class."""

import pytest
import numpy as np
from pandas.testing import assert_frame_equal

from ThreeWToolkit.core.base_dataset import BaseDataset
from ThreeWToolkit.core.dataset_outputs import DatasetOutputs
from ThreeWToolkit.dataset.transformed_dataset import TransformedDataset
from ThreeWToolkit.preprocessing import CleanSignalsConfig


@pytest.fixture
def simple_dataset(mock_dataset_factory) -> BaseDataset:
    """Simple dataset for normalization tests."""
    return mock_dataset_factory(num_sensors=10)


@pytest.fixture
def df_with_missing_column(simple_dataset) -> BaseDataset:
    """Dataset with a column that has 100% missing values."""
    missing_column = "sensor_5"

    def _introduce_missing_column(data: DatasetOutputs) -> DatasetOutputs:
        signal = data.signal.assign(**{missing_column: np.nan})
        return DatasetOutputs(signal=signal, label=data.label, metadata=data.metadata)

    return TransformedDataset(simple_dataset, _introduce_missing_column)


@pytest.fixture
def df_with_some_frozen_signals(simple_dataset) -> BaseDataset:
    """Dataset with some frozen (constant) signals."""
    frozen_column = "sensor_2"
    num_frozen_events = 5
    # Randomly select 5 events to freeze the signal for the specified column
    frozen_event_indices = np.random.choice(
        np.arange(len(simple_dataset)), size=num_frozen_events, replace=False
    )

    def _freeze_signal(data: DatasetOutputs) -> DatasetOutputs:
        if data.metadata["event_id"] in frozen_event_indices:
            signal = data.signal.assign(
                **{frozen_column: 4200.0}
            )  # low variance, out of range value
            return DatasetOutputs(
                signal=signal, label=data.label, metadata=data.metadata
            )
        else:
            return data

    return TransformedDataset(simple_dataset, _freeze_signal)


class TestCleanSignalsIQRThresholds:
    """Test IQR threshold detection and filtering."""

    def test_simple_dataset_no_frozen_signals(self, simple_dataset):
        """Test that a simple dataset with no frozen signals passes through unchanged."""

        cleaner = CleanSignalsConfig(
            exclude_features=[],
        ).build()
        cleaner.fit(simple_dataset)

        cleaned_dataset = TransformedDataset(simple_dataset, cleaner.transform)

        # Since there are no frozen signals, the cleaned dataset should be the same as the original
        for original, cleaned in zip(simple_dataset, cleaned_dataset):
            assert_frame_equal(original.signal, cleaned.signal)

    def test_simple_dataset_with_all_frozen_signals(self, df_with_missing_column):
        """Test that a dataset with one column of all NaN values is correctly identified and dropped."""

        cleaner = CleanSignalsConfig(
            exclude_features=[],
        ).build()
        cleaner.fit(df_with_missing_column)

        print(cleaner.drop_list)

        cleaned_dataset = TransformedDataset(df_with_missing_column, cleaner.transform)

        # The column with all NaN values should be dropped
        for _, cleaned in zip(df_with_missing_column, cleaned_dataset):
            assert "sensor_5" not in cleaned.signal.columns

    def test_simple_dataset_with_some_frozen_signals(self, df_with_some_frozen_signals):
        """Test that a dataset with some frozen signals is correctly identified and filtered."""

        cleaner = CleanSignalsConfig(
            exclude_features=[],
        ).build()
        cleaner.fit(df_with_some_frozen_signals)

        cleaned_dataset = TransformedDataset(
            df_with_some_frozen_signals, cleaner.transform
        )

        # The column with frozen signals should have NaN values in the cleaned dataset
        has_nan = []
        for _, cleaned in zip(df_with_some_frozen_signals, cleaned_dataset):
            assert "sensor_2" in cleaned.signal.columns
            has_nan.append(cleaned.signal["sensor_2"].isna().all())
        assert (
            sum(has_nan) == 5
        ), "Exactly 5 events should have the frozen signal column set to NaN"

    def test_exclude_features(self, df_with_missing_column):
        """Test that excluded features are not dropped even if they meet the IQR-based criteria."""

        cleaner = CleanSignalsConfig(
            exclude_features=["sensor_5"],
        ).build()
        cleaner.fit(df_with_missing_column)

        cleaned_dataset = TransformedDataset(df_with_missing_column, cleaner.transform)

        # The frozen column should not be dropped since it's in the exclude_features list
        for _, cleaned in zip(df_with_missing_column, cleaned_dataset):
            assert "sensor_5" in cleaned.signal.columns
