"""Tests for Normalize preprocessing class."""

import pytest
import numpy as np

from ThreeWToolkit.core.base_dataset import BaseDataset
from ThreeWToolkit.dataset.transformed_dataset import TransformedDataset
from ThreeWToolkit.preprocessing import NormalizeConfig


@pytest.fixture
def simple_dataset(mock_dataset_factory) -> BaseDataset:
    """Simple dataset for normalization tests."""
    return mock_dataset_factory(num_sensors=10)


class TestNormalizeStrategies:
    """Test different normalization strategies."""

    @pytest.mark.parametrize("norm", ["l1", "l2", "max", 1.0, 2.0, 3.0])
    def test_normalize_dataset_named(self, simple_dataset, norm):
        """Test L2 normalization."""
        normalizer = NormalizeConfig(norm=norm).build()
        normalizer.fit(simple_dataset)

        normalized_dataset = TransformedDataset(simple_dataset, normalizer.transform)

        second_normalizer = NormalizeConfig(norm=norm).build()
        second_normalizer.fit(normalized_dataset)

        # The second normalizer should find that the global average is close to 0 and the global moments are close to 1 after Lp normalization
        assert np.isclose(second_normalizer.global_average.values, 0.0).all()
        assert np.isclose(second_normalizer.global_moment.values, 1.0).all()
