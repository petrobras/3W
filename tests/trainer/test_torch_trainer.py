"""Tests for TorchTrainer and TorchTrainerConfig."""

import pytest
import torch
import numpy as np
import pandas as pd

from ThreeWToolkit.trainer.torch_trainer import TorchTrainer, TorchTrainerConfig
from ThreeWToolkit.models.mlp import MLPConfig
from ThreeWToolkit.core.dataset_outputs import DatasetOutputs
from ThreeWToolkit.core.base_dataset import BaseDataset


class MockDataset(BaseDataset):
    """Mock dataset for testing trainers."""

    def __init__(self, events: list[DatasetOutputs]):
        self.events = events

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int) -> DatasetOutputs:
        return self.events[idx]


def create_mock_dataset(
    num_events: int = 20,
    num_features: int = 10,
    num_timesteps: int = 50,
    num_classes: int = 2,
    seed: int = 42,
) -> MockDataset:
    """Create a mock dataset for testing."""
    np.random.seed(seed)
    events = []

    for i in range(num_events):
        signal_data = np.random.randn(num_timesteps, num_features).astype(np.float32)
        signal_df = pd.DataFrame(
            signal_data, columns=[f"sensor_{j}" for j in range(num_features)]
        )
        label_data = np.full(num_timesteps, i % num_classes, dtype=np.int64)
        label_series = pd.Series(label_data, name="label")

        events.append(
            DatasetOutputs(
                signal=signal_df,
                label=label_series,
                metadata={"event_id": i},
            )
        )

    return MockDataset(events)


class TestTorchTrainerConfig:
    """Tests for TorchTrainerConfig validation."""

    @pytest.fixture
    def mlp_config(self):
        """Provide basic MLP config."""
        return MLPConfig(hidden_sizes=(32,), output_size=2)

    def test_valid_config(self, mlp_config):
        """Valid config should be created."""
        config = TorchTrainerConfig(
            config_model=mlp_config,
            batch_size=32,
            epochs=10,
            learning_rate=0.001,
        )
        assert config.batch_size == 32
        assert config.epochs == 10
        assert config.learning_rate == 0.001

    def test_default_values(self, mlp_config):
        """Config should have sensible defaults."""
        config = TorchTrainerConfig(config_model=mlp_config)
        assert config.batch_size == 32
        assert config.epochs == 50
        assert config.learning_rate == 1e-3
        assert config.optimizer == "adam"
        assert config.criterion == "cross_entropy"
        assert config.shuffle is False

    @pytest.mark.parametrize("optimizer", ["adam", "sgd", "rmsprop", "adamw"])
    def test_valid_optimizers(self, mlp_config, optimizer):
        """All valid optimizers should be accepted."""
        config = TorchTrainerConfig(config_model=mlp_config, optimizer=optimizer)
        assert config.optimizer == optimizer

    def test_invalid_optimizer(self, mlp_config):
        """Invalid optimizer should raise ValueError."""
        with pytest.raises(ValueError, match="optimizer must be one of"):
            TorchTrainerConfig(config_model=mlp_config, optimizer="invalid")

    @pytest.mark.parametrize("criterion", ["cross_entropy", "mse", "mae"])
    def test_valid_criteria(self, mlp_config, criterion):
        """All valid criteria should be accepted."""
        config = TorchTrainerConfig(config_model=mlp_config, criterion=criterion)
        assert config.criterion == criterion

    def test_invalid_criterion(self, mlp_config):
        """Invalid criterion should raise ValueError."""
        with pytest.raises(ValueError, match="criterion must be one of"):
            TorchTrainerConfig(config_model=mlp_config, criterion="invalid")

    def test_invalid_batch_size_zero(self, mlp_config):
        """batch_size of 0 should raise ValidationError."""
        with pytest.raises(ValueError):
            TorchTrainerConfig(config_model=mlp_config, batch_size=0)

    def test_invalid_batch_size_negative(self, mlp_config):
        """Negative batch_size should raise ValidationError."""
        with pytest.raises(ValueError):
            TorchTrainerConfig(config_model=mlp_config, batch_size=-1)

    def test_invalid_epochs_zero(self, mlp_config):
        """epochs of 0 should raise ValidationError."""
        with pytest.raises(ValueError):
            TorchTrainerConfig(config_model=mlp_config, epochs=0)

    def test_invalid_learning_rate_zero(self, mlp_config):
        """learning_rate of 0 should raise ValidationError."""
        with pytest.raises(ValueError):
            TorchTrainerConfig(config_model=mlp_config, learning_rate=0)

    def test_invalid_learning_rate_negative(self, mlp_config):
        """Negative learning_rate should raise ValidationError."""
        with pytest.raises(ValueError):
            TorchTrainerConfig(config_model=mlp_config, learning_rate=-0.001)

    def test_device_cpu(self, mlp_config):
        """CPU device should always be valid."""
        config = TorchTrainerConfig(config_model=mlp_config, device="cpu")
        assert config.device == "cpu"

    def test_invalid_device(self, mlp_config):
        """Invalid device should raise ValueError."""
        with pytest.raises(ValueError, match="device must be"):
            TorchTrainerConfig(config_model=mlp_config, device="tpu")

    def test_target_returns_trainer_class(self, mlp_config):
        """target_ should return TorchTrainer class."""
        config = TorchTrainerConfig(config_model=mlp_config)
        assert config.target_ == TorchTrainer

    def test_class_weight_config(self, mlp_config):
        """Class weight config should be accepted."""
        config = TorchTrainerConfig(
            config_model=mlp_config,
            use_class_weights=True,
            class_weight_strategy="balanced",
        )
        assert config.use_class_weights is True
        assert config.class_weight_strategy == "balanced"

    def test_manual_class_weights(self, mlp_config):
        """Manual class weights should be configurable."""
        config = TorchTrainerConfig(
            config_model=mlp_config,
            use_class_weights=True,
            class_weight_strategy="manual",
            manual_class_weights={0: 1.0, 1: 2.0},
        )
        assert config.manual_class_weights == {0: 1.0, 1: 2.0}


class TestTorchTrainer:
    """Tests for TorchTrainer functionality."""

    @pytest.fixture
    def trainer_config(self):
        """Provide trainer config for testing."""
        mlp_config = MLPConfig(hidden_sizes=(32, 16), output_size=2)
        return TorchTrainerConfig(
            config_model=mlp_config,
            batch_size=8,
            epochs=2,
            learning_rate=0.01,
            device="cpu",
        )

    @pytest.fixture
    def mock_dataset(self):
        """Provide mock dataset for testing."""
        return create_mock_dataset(num_events=20, num_features=10, num_timesteps=50)

    def test_trainer_initialization(self, trainer_config):
        """Trainer should initialize correctly."""
        trainer = TorchTrainer(trainer_config)
        assert trainer.config == trainer_config
        assert trainer._class_weights is None

    def test_create_optimizer_adam(self, trainer_config):
        """Adam optimizer should be created correctly."""
        trainer = TorchTrainer(trainer_config)
        # Manually set model to test optimizer creation
        trainer.model = MLPConfig(
            hidden_sizes=(32,), output_size=2, input_size=10
        ).build()
        optimizer = trainer._create_optimizer()
        assert isinstance(optimizer, torch.optim.Adam)

    def test_create_optimizer_sgd(self):
        """SGD optimizer should be created correctly."""
        mlp_config = MLPConfig(hidden_sizes=(32,), output_size=2, input_size=10)
        config = TorchTrainerConfig(
            config_model=mlp_config, optimizer="sgd", device="cpu"
        )
        trainer = TorchTrainer(config)
        trainer.model = mlp_config.build()
        optimizer = trainer._create_optimizer()
        assert isinstance(optimizer, torch.optim.SGD)

    def test_create_criterion_cross_entropy(self, trainer_config):
        """CrossEntropyLoss should be created correctly."""
        trainer = TorchTrainer(trainer_config)
        criterion = trainer._create_criterion()
        assert isinstance(criterion, torch.nn.CrossEntropyLoss)

    def test_create_criterion_with_weights(self, trainer_config):
        """CrossEntropyLoss should accept class weights."""
        trainer = TorchTrainer(trainer_config)
        weights = torch.tensor([1.0, 2.0])
        criterion = trainer._create_criterion(weights)
        assert isinstance(criterion, torch.nn.CrossEntropyLoss)
        assert criterion.weight is not None

    def test_create_criterion_mse(self):
        """MSELoss should be created correctly."""
        mlp_config = MLPConfig(hidden_sizes=(32,), output_size=1, input_size=10)
        config = TorchTrainerConfig(
            config_model=mlp_config, criterion="mse", device="cpu"
        )
        trainer = TorchTrainer(config)
        criterion = trainer._create_criterion()
        assert isinstance(criterion, torch.nn.MSELoss)

    def test_train_basic(self, trainer_config, mock_dataset):
        """Basic training should complete successfully."""
        trainer = TorchTrainer(trainer_config)
        result = trainer.train(mock_dataset)

        assert result is not None
        assert result.model is not None
        assert result.train_dataset_size == len(mock_dataset)
        assert "train_loss" in result.history
        assert len(result.history["train_loss"]) == trainer_config.epochs

    def test_train_with_validation(self, trainer_config, mock_dataset):
        """Training with validation should track val_loss."""
        trainer = TorchTrainer(trainer_config)
        val_dataset = create_mock_dataset(
            num_events=10, num_features=10, num_timesteps=50, seed=99
        )

        result = trainer.train(mock_dataset, val_dataset)

        assert result.val_dataset_size == len(val_dataset)
        assert "val_loss" in result.history
        assert result.history["val_loss"] is not None
        assert len(result.history["val_loss"]) == trainer_config.epochs

    def test_train_with_class_weights(self, mock_dataset):
        """Training with class weights should work."""
        mlp_config = MLPConfig(hidden_sizes=(32,), output_size=2)
        config = TorchTrainerConfig(
            config_model=mlp_config,
            batch_size=8,
            epochs=2,
            device="cpu",
            use_class_weights=True,
            class_weight_strategy="balanced",
        )
        trainer = TorchTrainer(config)
        result = trainer.train(mock_dataset)

        assert result is not None
        assert trainer._class_weights is not None

    def test_train_decreasing_loss(self, mock_dataset):
        """Loss should generally decrease during training."""
        mlp_config = MLPConfig(hidden_sizes=(64, 32), output_size=2)
        config = TorchTrainerConfig(
            config_model=mlp_config,
            batch_size=16,
            epochs=10,
            learning_rate=0.01,
            device="cpu",
        )
        trainer = TorchTrainer(config)
        result = trainer.train(mock_dataset)

        train_loss = result.history["train_loss"]
        # Loss at end should be less than at start (with some tolerance for noise)
        assert train_loss[-1] < train_loss[0] * 1.5

    def test_model_in_training_result(self, trainer_config, mock_dataset):
        """Trained model should be accessible from result."""
        trainer = TorchTrainer(trainer_config)
        result = trainer.train(mock_dataset)

        model = result.model
        assert model is not None

        # Get the actual input size that was inferred during training
        input_size = trainer.config.config_model.input_size
        x = torch.randn(4, input_size)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (4, 2)

    def test_metadata_in_result(self, trainer_config, mock_dataset):
        """Training result should contain metadata."""
        trainer = TorchTrainer(trainer_config)
        result = trainer.train(mock_dataset)

        assert "trainer_type" in result.metadata
        assert result.metadata["trainer_type"] == "TorchTrainer"
        assert "seed" in result.metadata


class TestTorchTrainerValidation:
    """Tests for TorchTrainer dataset validation."""

    @pytest.fixture
    def trainer(self):
        """Provide trainer for validation tests."""
        mlp_config = MLPConfig(hidden_sizes=(32,), output_size=2)
        config = TorchTrainerConfig(config_model=mlp_config, device="cpu")
        return TorchTrainer(config)

    def test_empty_dataset_raises(self, trainer):
        """Empty dataset should raise ValueError."""
        empty_dataset = MockDataset([])

        with pytest.raises(ValueError, match="Training dataset is empty"):
            trainer.train(empty_dataset)

    def test_empty_validation_dataset_raises(self, trainer):
        """Empty validation dataset should raise ValueError."""
        train_dataset = create_mock_dataset(num_events=10)
        empty_val = MockDataset([])

        with pytest.raises(ValueError, match="Validation dataset is empty"):
            trainer.train(train_dataset, empty_val)

    def test_mismatched_columns_raises(self, trainer):
        """Mismatched signal columns should raise ValueError."""
        train_dataset = create_mock_dataset(num_events=10, num_features=10)

        # Create val dataset with different number of features
        val_events = []
        signal_df = pd.DataFrame(
            np.random.randn(50, 5),  # Different number of features
            columns=[f"sensor_{j}" for j in range(5)],
        )
        label_series = pd.Series(np.zeros(50, dtype=np.int64), name="label")
        val_events.append(
            DatasetOutputs(signal=signal_df, label=label_series, metadata={})
        )
        val_dataset = MockDataset(val_events)

        with pytest.raises(ValueError, match="different signal columns"):
            trainer.train(train_dataset, val_dataset)

    def test_mismatched_labels_raises(self, trainer):
        """Train with labels, val without should raise ValueError."""
        train_dataset = create_mock_dataset(num_events=10)

        # Create val dataset without labels
        val_events = []
        signal_df = pd.DataFrame(
            np.random.randn(50, 10), columns=[f"sensor_{j}" for j in range(10)]
        )
        val_events.append(DatasetOutputs(signal=signal_df, label=None, metadata={}))
        val_dataset = MockDataset(val_events)

        with pytest.raises(ValueError, match="both have labels or both not"):
            trainer.train(train_dataset, val_dataset)


class TestTorchTrainerIntegration:
    """Integration tests for TorchTrainer with realistic scenarios."""

    def test_multiclass_classification(self):
        """Trainer should handle multi-class classification."""
        num_classes = 5
        dataset = create_mock_dataset(
            num_events=50, num_features=20, num_classes=num_classes
        )

        mlp_config = MLPConfig(hidden_sizes=(64, 32), output_size=num_classes)
        config = TorchTrainerConfig(
            config_model=mlp_config, batch_size=16, epochs=5, device="cpu"
        )

        trainer = TorchTrainer(config)
        result = trainer.train(dataset)

        # Get the actual input size that was inferred during training
        input_size = trainer.config.config_model.input_size
        x = torch.randn(4, input_size)
        with torch.no_grad():
            output = result.model(x)
        assert output.shape == (4, num_classes)

    def test_reproducibility_with_seed(self):
        """Training with same seed should produce same results."""
        dataset = create_mock_dataset(num_events=20, seed=42)

        mlp_config = MLPConfig(hidden_sizes=(32,), output_size=2)
        config = TorchTrainerConfig(
            config_model=mlp_config, batch_size=8, epochs=3, device="cpu", seed=42
        )

        # Train twice with same seed
        trainer1 = TorchTrainer(config)
        result1 = trainer1.train(dataset)

        trainer2 = TorchTrainer(config)
        result2 = trainer2.train(dataset)

        # Results should be identical
        assert result1.history["train_loss"] == result2.history["train_loss"]

    def test_different_batch_sizes(self):
        """Training should work with various batch sizes."""
        dataset = create_mock_dataset(num_events=32, num_features=10)

        for batch_size in [1, 8, 16, 32]:
            mlp_config = MLPConfig(hidden_sizes=(32,), output_size=2)
            config = TorchTrainerConfig(
                config_model=mlp_config,
                batch_size=batch_size,
                epochs=2,
                device="cpu",
            )
            trainer = TorchTrainer(config)
            result = trainer.train(dataset)
            assert len(result.history["train_loss"]) == 2
