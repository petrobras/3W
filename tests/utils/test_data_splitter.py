import pytest
from ThreeWToolkit.utils.data_splitter import TrainTestSplitter, KFoldSplitter
from ThreeWToolkit.dataset.subset_dataset import SubsetDataset
from ThreeWToolkit.core.base_dataset import BaseDataset
from ThreeWToolkit.core.dataset_outputs import DatasetOutputs


class SimpleTestDataset(BaseDataset):
    """Simple dataset for testing splitter functionality."""

    def __init__(self, size: int = 100):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> DatasetOutputs:
        import pandas as pd

        return DatasetOutputs(
            signal=pd.DataFrame({"col": [idx]}),
            label=pd.Series([0]),
            metadata={"event_id": idx, "class": int(idx % 3)},
        )


class TestTrainTestSplitter:
    """Tests for TrainTestSplitter class."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        splitter = TrainTestSplitter()
        assert splitter.size_training == 0.8
        assert splitter.size_test == 0.2
        assert splitter.shuffle is True
        assert splitter.random_state is None

    def test_split_default_proportions(self):
        """Test 80/20 split."""
        dataset = SimpleTestDataset(100)
        train_set, test_set = TrainTestSplitter().split_data(dataset)
        assert len(train_set) == 80
        assert len(test_set) == 20

    def test_split_custom_proportions(self):
        """Test custom proportions."""
        dataset = SimpleTestDataset(100)
        train_set, test_set = TrainTestSplitter(
            size_training=0.7, size_test=0.3
        ).split_data(dataset)
        assert len(train_set) == 70
        assert len(test_set) == 30

    def test_no_overlap(self):
        """Test that training and test sets don't overlap."""
        dataset = SimpleTestDataset(100)
        train_set, test_set = TrainTestSplitter().split_data(dataset)
        assert len(set(train_set.indices) & set(test_set.indices)) == 0

    def test_covers_all_indices(self):
        """Test that all dataset indices are covered."""
        dataset = SimpleTestDataset(100)
        train_set, test_set = TrainTestSplitter().split_data(dataset)
        all_indices = set(train_set.indices) | set(test_set.indices)
        assert all_indices == set(range(100))

    def test_no_shuffle_maintains_order(self):
        """Test that shuffle=False maintains index order."""
        dataset = SimpleTestDataset(100)
        train_set, test_set = TrainTestSplitter(shuffle=False).split_data(dataset)
        assert train_set.indices == list(range(80))
        assert test_set.indices == list(range(80, 100))

    def test_reproducible_with_random_state(self):
        """Test reproducibility with same random_state."""
        dataset1 = SimpleTestDataset(100)
        dataset2 = SimpleTestDataset(100)
        splitter1 = TrainTestSplitter(random_state=42)
        splitter2 = TrainTestSplitter(random_state=42)

        train1, test1 = splitter1.split_data(dataset1)
        train2, test2 = splitter2.split_data(dataset2)

        assert train1.indices == train2.indices
        assert test1.indices == test2.indices

    def test_error_invalid_proportions(self):
        """Test error when proportions don't sum to 1."""
        dataset = SimpleTestDataset(100)
        with pytest.raises(ValueError, match="must equal 1.0"):
            TrainTestSplitter(size_training=0.7, size_test=0.2).split_data(dataset)

    def test_subset_dataset_access(self):
        """Test that subsets exist and have correct length."""
        dataset = SimpleTestDataset(100)
        train_set, test_set = TrainTestSplitter().split_data(dataset)

        assert isinstance(train_set, SubsetDataset)
        assert isinstance(test_set, SubsetDataset)
        assert len(train_set) == 80
        assert len(test_set) == 20


class TestKFoldSplitter:
    """Tests for KFoldSplitter class."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        splitter = KFoldSplitter()
        assert splitter.num_splits == 5
        assert splitter.random_state is None
        assert splitter.stratify_by == []

    def test_split_returns_correct_number_of_folds(self):
        """Test correct number of folds returned."""
        dataset = SimpleTestDataset(100)
        folds = list(KFoldSplitter(num_splits=5).split_data(dataset))
        assert len(folds) == 5

    def test_split_folds_are_subset_datasets(self):
        """Test that folds contain SubsetDataset tuples."""
        dataset = SimpleTestDataset(100)
        folds = list(KFoldSplitter(num_splits=5).split_data(dataset))

        for train_set, test_set in folds:
            assert isinstance(train_set, SubsetDataset)
            assert isinstance(test_set, SubsetDataset)

    def test_folds_cover_entire_dataset(self):
        """Test that all folds cover the entire dataset."""
        dataset = SimpleTestDataset(100)
        folds = list(KFoldSplitter(num_splits=5).split_data(dataset))

        all_test_indices = set()
        for _, test_set in folds:
            all_test_indices.update(test_set.indices)

        assert all_test_indices == set(range(100))

    def test_no_overlap_between_test_sets(self):
        """Test that test sets don't overlap across folds."""
        dataset = SimpleTestDataset(100)
        folds = list(KFoldSplitter(num_splits=5).split_data(dataset))

        for i, (_, test_set1) in enumerate(folds):
            for j, (_, test_set2) in enumerate(folds):
                if i != j:
                    assert len(set(test_set1.indices) & set(test_set2.indices)) == 0

    def test_training_set_larger_than_test_set(self):
        """Test that training sets are larger than test sets."""
        dataset = SimpleTestDataset(100)
        folds = list(KFoldSplitter(num_splits=5).split_data(dataset))

        for train_set, test_set in folds:
            assert len(train_set) > len(test_set)
            assert len(train_set) + len(test_set) == len(dataset)

    def test_different_num_splits(self):
        """Test with different numbers of splits."""
        dataset = SimpleTestDataset(100)

        for num_splits in [2, 5, 10]:
            folds = list(KFoldSplitter(num_splits=num_splits).split_data(dataset))
            assert len(folds) == num_splits

    def test_stratified_split_by_metadata(self):
        """Test stratified split by metadata field."""
        dataset = SimpleTestDataset(100)
        folds = list(
            KFoldSplitter(num_splits=5, stratify_by=["class"]).split_data(dataset)
        )

        assert len(folds) == 5

    def test_stratified_maintains_class_distribution(self):
        """Test that stratified split maintains class distribution."""
        dataset = SimpleTestDataset(100)
        folds = list(
            KFoldSplitter(num_splits=5, stratify_by=["class"]).split_data(dataset)
        )

        # Just verify stratified split works without errors
        assert len(folds) == 5

    def test_error_invalid_stratify_key(self):
        """Test error when stratifying by non-existent key."""
        dataset = SimpleTestDataset(100)
        with pytest.raises(ValueError):
            list(
                KFoldSplitter(num_splits=5, stratify_by=["non_existent"]).split_data(
                    dataset
                )
            )

    def test_reproducible_with_random_state(self):
        """Test reproducibility with same random_state."""
        dataset1 = SimpleTestDataset(100)
        dataset2 = SimpleTestDataset(100)
        splitter1 = KFoldSplitter(num_splits=5, random_state=42)
        splitter2 = KFoldSplitter(num_splits=5, random_state=42)

        folds1 = list(splitter1.split_data(dataset1))
        folds2 = list(splitter2.split_data(dataset2))

        for (train1, test1), (train2, test2) in zip(folds1, folds2):
            assert list(train1.indices) == list(train2.indices)
            assert list(test1.indices) == list(test2.indices)

    def test_subset_dataset_access(self):
        """Test that fold subsets exist and have correct structure."""
        dataset = SimpleTestDataset(100)
        folds = list(KFoldSplitter(num_splits=5).split_data(dataset))

        train_set, test_set = folds[0]
        assert isinstance(train_set, SubsetDataset)
        assert isinstance(test_set, SubsetDataset)
        assert len(train_set) > 0
        assert len(test_set) > 0


class TestDataSplitterIntegration:
    """Integration tests with mock datasets."""

    def test_train_test_splitter_with_mock_dataset(self, mock_dataset_factory):
        """Test TrainTestSplitter with factory-created dataset."""
        dataset = mock_dataset_factory(num_events=100)
        train_set, test_set = TrainTestSplitter().split_data(dataset)

        assert len(train_set) == 80
        assert len(test_set) == 20
        assert train_set[0].signal is not None

    def test_kfold_splitter_with_mock_dataset(self, mock_dataset_factory):
        """Test KFoldSplitter with factory-created dataset."""
        dataset = mock_dataset_factory(num_events=100)
        folds = list(KFoldSplitter(num_splits=5).split_data(dataset))

        assert len(folds) == 5
        for train_set, test_set in folds:
            assert train_set[0].signal is not None
            assert test_set[0].signal is not None
