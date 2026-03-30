"""Tests for RenameColumns preprocessing class."""

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from ThreeWToolkit.core.base_dataset import BaseDataset
from ThreeWToolkit.preprocessing import RenameColumns, RenameColumnsConfig

@pytest.fixture
def simple_dataset(mock_dataset_factory) -> BaseDataset:
    """Simple label series for remapping."""
    return mock_dataset_factory(num_sensors=10, known_labels=[0, 1, 2, 101, 102])


class TestRenameColumnsFunctionality:
    """Test basic renaming functionality."""

    def test_functional_case(self, simple_dataset):
        """Test renaming multiple columns while keeping others unchanged."""
        columns_map = {"sensor_0": "X", "sensor_1": "Y"}
        rename = RenameColumnsConfig(columns_map=columns_map).build()

        rename.fit(simple_dataset) # should be no-op

        for event in simple_dataset:
            event = rename.transform(event)
            assert "sensor_0" not in event.signal.columns
            assert "sensor_1" not in event.signal.columns

            assert "X" in event.signal.columns
            assert "Y" in event.signal.columns

    def test_missing_column(self, simple_dataset):
        """Test behavior when columns_map contains a column not in the signal DataFrame."""
        columns_map = {"sensor_0": "X", "sensor_999": "Z"}  # sensor_999 does not exist
        rename = RenameColumnsConfig(columns_map=columns_map).build()

        rename.fit(simple_dataset) # should be no-op

        for event in simple_dataset:
            with pytest.raises(ValueError):
                rename.transform(event)
