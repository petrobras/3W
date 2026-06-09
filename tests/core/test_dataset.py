"""Tests for BaseDataset class."""

import pytest
import pandas as pd

from ThreeWToolkit.core import BaseDataset, DatasetOutputs


class TestBaseDatasetImplementation:
    """Test BaseDataset implementation requirements."""

    def test_abstract_methods_required(self):
        """Test that abstract methods must be implemented."""

        class IncompleteDataset(BaseDataset):
            pass

        with pytest.raises(TypeError):
            IncompleteDataset()

    def test_complete_implementation(self):
        """Test a complete implementation of BaseDataset."""

        class SimpleDataset(BaseDataset):
            def __init__(self, data: list[DatasetOutputs]):
                self._data = data

            def __len__(self):
                return len(self._data)

            def __getitem__(self, idx):
                return self._data[idx]

        signal = pd.DataFrame({"sensor_0": [1.0, 2.0]})
        label = pd.Series([0, 1])
        event = DatasetOutputs(signal=signal, label=label)

        dataset = SimpleDataset([event, event, event])

        assert len(dataset) == 3
        assert isinstance(dataset[0], DatasetOutputs)


class TestBaseDatasetIteration:
    """Test BaseDataset iteration functionality."""

    def test_iter_yields_all_items(self):
        """Test that __iter__ yields all items."""

        class SimpleDataset(BaseDataset):
            def __init__(self, data: list[DatasetOutputs]):
                self._data = data

            def __len__(self):
                return len(self._data)

            def __getitem__(self, idx):
                return self._data[idx]

        events = []
        for i in range(5):
            signal = pd.DataFrame({"sensor_0": [float(i)]})
            label = pd.Series([i % 2])
            events.append(
                DatasetOutputs(signal=signal, label=label, metadata={"id": i})
            )

        dataset = SimpleDataset(events)
        iterated = list(dataset)

        assert len(iterated) == 5
        for i, event in enumerate(iterated):
            assert event.metadata["id"] == i

    def test_iter_on_empty_dataset(self):
        """Test iteration on empty dataset."""

        class SimpleDataset(BaseDataset):
            def __init__(self, data: list[DatasetOutputs]):
                self._data = data

            def __len__(self):
                return len(self._data)

            def __getitem__(self, idx):
                return self._data[idx]

        dataset = SimpleDataset([])
        iterated = list(dataset)

        assert iterated == []

    def test_for_loop_iteration(self):
        """Test for loop iteration over dataset."""

        class SimpleDataset(BaseDataset):
            def __init__(self, data: list[DatasetOutputs]):
                self._data = data

            def __len__(self):
                return len(self._data)

            def __getitem__(self, idx):
                return self._data[idx]

        events = [
            DatasetOutputs(
                signal=pd.DataFrame({"s": [1.0]}),
                label=pd.Series([0]),
            )
            for _ in range(3)
        ]
        dataset = SimpleDataset(events)

        count = 0
        for event in dataset:
            assert isinstance(event, DatasetOutputs)
            count += 1

        assert count == 3


class TestMockDataset:
    """Test using MockDataset from conftest."""

    def test_mock_dataset_factory_basic(self, mock_dataset_factory):
        """Test creating mock dataset with factory."""
        dataset = mock_dataset_factory(num_events=10, num_sensors=5)

        assert len(dataset) == 10
        assert isinstance(dataset[0], DatasetOutputs)
        assert dataset[0].signal.shape[1] == 5

    def test_mock_dataset_factory_with_nan(self, mock_dataset_factory):
        """Test creating mock dataset with NaN columns."""
        dataset = mock_dataset_factory(
            num_events=5,
            num_sensors=10,
            all_nan_columns=[0, 1],
        )

        for event in dataset:
            assert event.signal.iloc[:, 0].isna().all()
            assert event.signal.iloc[:, 1].isna().all()
            assert not event.signal.iloc[:, 2].isna().all()

    def test_mock_dataset_factory_reproducible(self, mock_dataset_factory):
        """Test that seed makes dataset reproducible."""
        dataset1 = mock_dataset_factory(num_events=5, seed=42)
        dataset2 = mock_dataset_factory(num_events=5, seed=42)

        for e1, e2 in zip(dataset1, dataset2):
            pd.testing.assert_frame_equal(e1.signal, e2.signal)
            pd.testing.assert_series_equal(e1.label, e2.label)

    def test_mock_dataset_iteration(self, mock_dataset_factory):
        """Test iterating over mock dataset."""
        dataset = mock_dataset_factory(num_events=3)

        events = list(dataset)
        assert len(events) == 3

        for event in events:
            assert event.signal is not None
            assert event.label is not None
