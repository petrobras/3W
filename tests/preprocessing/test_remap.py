"""Tests for RemapClass preprocessing class."""

import pytest
import pandas as pd

from ThreeWToolkit.core.base_dataset import BaseDataset
from ThreeWToolkit.preprocessing import RemapClassConfig


# Module-level fixtures

@pytest.fixture
def simple_dataset(mock_dataset_factory) -> BaseDataset:
    """Simple label series for remapping."""
    return mock_dataset_factory(num_sensors=10)


@pytest.fixture
def string_labels():
    """String label series for remapping."""
    return pd.Series(["A", "B", "C", "B", "A"])


# Tests for RemapClass functionality


class TestRemapClassManualMapping:
    """Test manual class mapping."""

    def test_manual_mapping_integers(self, simple_labels):
        """Test manual integer label remapping."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    def test_manual_mapping_strings(self, string_labels):
        """Test manual string label remapping."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    @pytest.mark.parametrize(
        "class_map,expected",
        [
            ({0: 10, 1: 20, 2: 30}, [10, 20, 30, 20, 10, 30]),
            ({0: "zero", 1: "one", 2: "two"}, ["zero", "one", "two", "one", "zero", "two"]),
        ],
    )
    def test_various_mappings(self, simple_labels, class_map, expected):
        """Test different mapping configurations parametrized."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")


class TestRemapClassAutoGeneration:
    """Test automatic class mapping generation."""

    def test_auto_generate_mapping(self, simple_labels):
        """Test automatic generation of class mapping from data."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    def test_auto_generate_with_fit(self, simple_labels):
        """Test fit() method for automatic mapping generation."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")


class TestRemapClassEdgeCases:
    """Test edge cases and error handling."""

    def test_unmapped_class_raises_error(self):
        """Test error when encountering unmapped class value."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    def test_empty_mapping(self):
        """Test behavior with empty mapping."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")

    def test_partial_mapping(self):
        """Test behavior with partial class mapping."""
        # TODO: Implement test
        pytest.skip("API adaptation needed")
