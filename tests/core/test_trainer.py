"""Tests for BaseTrainer and TrainingResult."""

import pytest
import pandas as pd
import numpy as np

from ThreeWToolkit.core.base_trainer import PredictionResult

from ThreeWToolkit.core import (
    BaseDataset,
    BaseTrainer,
    BaseTrainerConfig,
    TrainingResult,
    TrainingHistory,
    BaseModels,
    DatasetOutputs,
)


# Create a concrete trainer for testing
class ConcreteTrainer(BaseTrainer):
    def _prepare_data(self, dataset, shuffle=True):
        return dataset

    def predict(self, dataset: BaseDataset) -> PredictionResult:
        return PredictionResult(
            y_pred=np.random.randint(0, 2, size=len(dataset)),
            y_true=np.random.randint(0, 2, size=len(dataset)),
        )

    def _initialize_training_state(
        self, train_data, train_dataset: BaseDataset
    ) -> None:
        return super()._initialize_training_state(train_data, train_dataset)

    def _execute_training(self, train_data, val_data) -> TrainingHistory:
        return TrainingHistory(train_loss=[0.5, 0.3, 0.1], val_loss=[0.6, 0.4, 0.2])


class ConcreteTrainerConfig(BaseTrainerConfig):
    """Concrete trainer config for testing."""

    _target: type = ConcreteTrainer


class TestBaseTrainerConfig:
    """Test BaseTrainerConfig validation."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ConcreteTrainerConfig()

        assert config.seed == 42
        assert config.use_class_weights is False
        assert config.class_weight_strategy == "balanced"
        assert config.manual_class_weights is None

    def test_custom_seed(self):
        """Test custom seed value."""
        config = ConcreteTrainerConfig(seed=123)
        assert config.seed == 123

    def test_use_class_weights_enabled(self):
        """Test enabling class weights."""
        config = ConcreteTrainerConfig(use_class_weights=True)
        assert config.use_class_weights is True

    def test_manual_class_weights(self):
        """Test manual class weights configuration."""
        weights = {0: 1.0, 1: 2.0}
        config = ConcreteTrainerConfig(
            class_weight_strategy="manual",
            manual_class_weights=weights,
        )

        assert config.class_weight_strategy == "manual"
        assert config.manual_class_weights == weights


class TestTrainingResult:
    """Test TrainingResult class."""

    def test_training_result_creation(self):
        """Test creating TrainingResult."""

        class DummyModel(BaseModels):
            def save(self, filename):
                pass

            def load(self, filename):
                pass

        model = DummyModel()
        history = TrainingHistory(train_loss=[0.5, 0.3, 0.1])

        result = TrainingResult(
            model=model,
            history=history,
            train_dataset_size=100,
            val_dataset_size=20,
        )

        assert result.model is model
        assert result.history.train_loss == [0.5, 0.3, 0.1]
        assert result.train_dataset_size == 100
        assert result.val_dataset_size == 20
        assert result.metadata == {}

    def test_training_result_with_metadata(self):
        """Test TrainingResult with metadata."""

        class DummyModel(BaseModels):
            def save(self, filename):
                pass

            def load(self, filename):
                pass

        model = DummyModel()
        metadata = {"epochs": 50, "learning_rate": 0.001}

        result = TrainingResult(
            model=model,
            history=TrainingHistory(train_loss=[0.5, 0.3, 0.1]),
            train_dataset_size=100,
            val_dataset_size=0,
            metadata=metadata,
        )

        assert result.metadata["epochs"] == 50
        assert result.metadata["learning_rate"] == 0.001

    def test_training_result_no_validation(self):
        """Test TrainingResult without validation dataset."""

        class DummyModel(BaseModels):
            def save(self, filename):
                pass

            def load(self, filename):
                pass

        result = TrainingResult(
            model=DummyModel(),
            history=TrainingHistory(train_loss=[0.5, 0.3, 0.1]),
            train_dataset_size=100,
            val_dataset_size=0,
        )

        assert result.val_dataset_size == 0


class TestBaseTrainerValidation:
    """Test BaseTrainer dataset validation."""

    def test_validate_empty_train_dataset(self, mock_dataset_factory):
        """Test validation fails for empty training dataset."""

        class EmptyDataset(BaseDataset):
            def __len__(self):
                return 0

            def __getitem__(self, idx):
                raise IndexError("Empty dataset")

        trainer = ConcreteTrainer(ConcreteTrainerConfig())

        with pytest.raises(ValueError, match="Training dataset is empty"):
            trainer._validate_datasets(EmptyDataset(), None)

    def test_validate_empty_val_dataset(self, mock_dataset_factory):
        """Test validation fails for empty validation dataset."""

        class EmptyDataset(BaseDataset):
            def __len__(self):
                return 0

            def __getitem__(self, idx):
                raise IndexError("Empty dataset")

        train_dataset = mock_dataset_factory(num_events=5)
        trainer = ConcreteTrainer(ConcreteTrainerConfig())

        with pytest.raises(ValueError, match="Validation dataset is empty"):
            trainer._validate_datasets(train_dataset, EmptyDataset())

    def test_validate_mismatched_columns(self, mock_dataset_factory):
        """Test validation fails for mismatched signal columns."""
        train_dataset = mock_dataset_factory(num_events=5, num_sensors=10)

        # Create val dataset with different columns
        class DifferentColumnsDataset(BaseDataset):
            def __len__(self):
                return 3

            def __getitem__(self, idx):
                signal = pd.DataFrame({"different_col": [1.0, 2.0]})
                label = pd.Series([0, 1])
                return DatasetOutputs(signal=signal, label=label)

        trainer = ConcreteTrainer(ConcreteTrainerConfig())

        with pytest.raises(ValueError, match="different signal columns"):
            trainer._validate_datasets(train_dataset, DifferentColumnsDataset())

    def test_validate_mismatched_label_presence(self, mock_dataset_factory):
        """Test validation fails when label presence differs."""
        train_dataset = mock_dataset_factory(num_events=5, num_sensors=3)

        # Create val dataset without labels
        class NoLabelDataset(BaseDataset):
            def __len__(self):
                return 3

            def __getitem__(self, idx):
                signal = pd.DataFrame(
                    {
                        "sensor_0": [1.0],
                        "sensor_1": [2.0],
                        "sensor_2": [3.0],
                    }
                )
                return DatasetOutputs(signal=signal, label=None)

        trainer = ConcreteTrainer(ConcreteTrainerConfig())

        with pytest.raises(ValueError, match="both have labels"):
            trainer._validate_datasets(train_dataset, NoLabelDataset())


class TestBaseTrainerClassWeights:
    """Test class weight computation."""

    def test_compute_balanced_class_weights(self, mock_dataset_factory):
        """Test computing balanced class weights."""
        trainer = ConcreteTrainer(ConcreteTrainerConfig(use_class_weights=True))
        dataset = mock_dataset_factory(num_events=10, known_labels=[0, 1])

        weights = trainer._compute_class_weights(dataset)

        assert isinstance(weights, dict)
        assert 0 in weights or 1 in weights

    def test_compute_manual_class_weights(self, mock_dataset_factory):
        """Test using manual class weights."""
        manual_weights = {0: 1.0, 1: 3.0}
        config = ConcreteTrainerConfig(
            class_weight_strategy="manual",
            manual_class_weights=manual_weights,
        )
        trainer = ConcreteTrainer(config)
        dataset = mock_dataset_factory(num_events=5)

        weights = trainer._compute_class_weights(dataset)

        assert weights == manual_weights

    def test_manual_weights_missing_raises_error(self, mock_dataset_factory):
        """Test that manual strategy without weights raises error."""
        config = ConcreteTrainerConfig(class_weight_strategy="manual")
        trainer = ConcreteTrainer(config)
        dataset = mock_dataset_factory(num_events=5)

        with pytest.raises(ValueError, match="manual_class_weights must be provided"):
            trainer._compute_class_weights(dataset)
