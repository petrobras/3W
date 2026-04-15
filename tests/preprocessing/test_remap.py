"""Tests for RemapClass preprocessing class."""

import pytest

from ThreeWToolkit.core.base_dataset import BaseDataset
from ThreeWToolkit.preprocessing import RemapClassConfig


@pytest.fixture
def simple_dataset(mock_dataset_factory) -> BaseDataset:
    """Simple label series for remapping."""
    return mock_dataset_factory(num_sensors=10, known_labels=[0, 1, 2, 101, 102])


# Tests for RemapClass functionality
class TestRemapClassManualMapping:
    """Test manual class mapping."""

    def test_manual_mapping_integers(self, simple_dataset):
        """Test manual integer label remapping."""
        remapper = RemapClassConfig(
            class_map={0: 10, 1: 20, 2: 30, 101: 20, 102: 30}
        ).build()

        remapper.fit(simple_dataset)
        assert remapper.class_map == {0: 10, 1: 20, 2: 30, 101: 20, 102: 30}

        for event in simple_dataset:
            transformed = remapper.transform(event)
            assert all(transformed.label.isin([10, 20, 30]))

    def test_unknown_label_mapper(self, simple_dataset):
        """Test behavior when encountering an unknown label value."""
        remapper = RemapClassConfig(
            class_map={200: 0}
        ).build()  # dataset does not have label 200.

        with pytest.raises(ValueError):
            remapper.fit(simple_dataset)

        assert remapper.class_map == {200: 0}

        for event in simple_dataset:
            with pytest.raises(
                ValueError, match="Some labels were not in the class_map"
            ):
                remapper.transform(event)


class TestRemapClassAutoMapping:
    """Test automatic class mapping."""

    def test_auto_mapping(self, simple_dataset):
        """Test automatic mapping of unique labels to integers."""
        remapper = RemapClassConfig().build()

        remapper.fit(simple_dataset)
        expected_map = {
            0: 0,
            1: 1,
            2: 2,
            101: 3,
            102: 4,
        }  # unique labels sorted and mapped to integers
        assert remapper.class_map == expected_map

        for event in simple_dataset:
            transformed = remapper.transform(event)
            assert all(transformed.label.isin(expected_map.values()))
