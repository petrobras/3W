import pytest
import torch
import torch.nn as nn
import numpy as np
from ThreeWToolkit.models.mlp import (
    MLP,
    MLPConfig,
    LabeledSubset,
    ActivationFunctionEnum,
)
from ThreeWToolkit.trainer.trainer import ModelTrainer, TrainerConfig


@pytest.fixture
def trainer_setup():
    """
    Pytest fixture to set up a standard ModelTrainer and data for testing.
    """
    num_samples = 100
    input_size = 10
    data = np.random.rand(num_samples, input_size)
    target_data = np.random.randint(0, 2, num_samples)
    x_tensor = torch.tensor(data, dtype=torch.float32)
    y_tensor = torch.tensor(target_data, dtype=torch.float32).unsqueeze(1)
    config = MLPConfig(
        input_size=input_size,
        hidden_sizes=(32, 16),
        output_size=1,
        activation_function="relu",
        regularization=None,
        random_seed=42,
    )
    trainer_config = TrainerConfig(
        batch_size=16,
        epochs=3,
        seed=42,
        learning_rate=1e-3,
        config_model=config,
        optimizer="adam",
        criterion="mse",
        device="cpu",
    )
    trainer = ModelTrainer(trainer_config)
    return {
        "x_tensor": x_tensor,
        "y_tensor": y_tensor,
        "config": config,
        "trainer": trainer,
    }


class TestLabeledSubset:
    def test_init_value_error(self):
        """
        Tests if LabeledSubset raises ValueError for inputs of different lengths.
        """
        samples = torch.rand(10, 2)
        labels = torch.rand(9)
        with pytest.raises(
            ValueError, match="Samples and labels must have the same length."
        ):
            LabeledSubset(samples, labels)


class TestMLP:
    def test_invalid_activation_function(self):
        config = MLPConfig(
            input_size=5,
            hidden_sizes=(4,),
            output_size=1,
            activation_function="notarealactivation",
            regularization=None,
            random_seed=42,
        )
        with pytest.raises(ValueError, match="Unknown activation function"):
            MLP(config)

    @pytest.mark.parametrize(
        "activation_enum, expected_type",
        [
            (ActivationFunctionEnum.RELU, nn.ReLU),
            (ActivationFunctionEnum.SIGMOID, nn.Sigmoid),
            (ActivationFunctionEnum.TANH, nn.Tanh),
        ],
    )
    def test_mlp_activation_function(self, activation_enum, expected_type):
        """
        Tests that the MLP model is correctly constructed with various activation functions.
        """
        config = MLPConfig(
            input_size=10,
            hidden_sizes=(8, 3),
            output_size=1,
            activation_function=activation_enum,
            regularization=None,
            random_seed=42,
        )
        model = MLP(config)
        assert isinstance(model.model[1], expected_type)
        assert isinstance(model.model[3], expected_type)

    def test_init_type_error(self):
        """
        Tests that initializing an MLP with an invalid config (dict) raises an AttributeError.
        """
        with pytest.raises(
            AttributeError, match="'dict' object has no attribute 'activation_function'"
        ):
            MLP(config={"input_size": 10})  # type: ignore


class TestModelTrainer:
    def test_trainer_initialization(self, trainer_setup):
        trainer = trainer_setup["trainer"]
        assert trainer.batch_size == 16
        # Trainer.config is TrainerConfig, check config_model is MLPConfig
        assert isinstance(trainer.config.config_model, MLPConfig)

    def test_train_and_history(self, trainer_setup):
        trainer = trainer_setup["trainer"]
        x = trainer_setup["x_tensor"]
        y = trainer_setup["y_tensor"]
        trainer.train(x, y)
        # Should have a history with train_loss and val_loss
        hist = trainer.history[0]
        assert "train_loss" in hist and "val_loss" in hist
        assert len(hist["train_loss"]) == trainer.epochs

    def test_test_method(self, trainer_setup):
        trainer = trainer_setup["trainer"]
        x = trainer_setup["x_tensor"]
        y = trainer_setup["y_tensor"]
        trainer.train(x, y)
        from sklearn.metrics import mean_squared_error

        test_loss, test_metrics = trainer.test(x, y, metrics=[mean_squared_error])
        assert isinstance(test_loss, float)
        assert "mean_squared_error" in test_metrics

    def test_save_and_load(self, tmp_path, trainer_setup):
        trainer = trainer_setup["trainer"]
        x = trainer_setup["x_tensor"]
        y = trainer_setup["y_tensor"]
        trainer.train(x, y)
        save_path = tmp_path / "model.pth"
        trainer.save(save_path)
        loaded_model = trainer.load(save_path)
        assert loaded_model is not None

    def test_labeled_subset(self):
        samples = torch.rand(10, 2)
        labels = torch.rand(10, 1)
        ds = LabeledSubset(samples, labels)
        assert len(ds) == 10
        x, y = ds[0]
        assert x.shape[0] == 2

    def test_labeled_subset_value_error(self):
        samples = torch.rand(10, 2)
        labels = torch.rand(9, 1)
        with pytest.raises(ValueError):
            LabeledSubset(samples, labels)

    def test_mlp_forward(self):
        config = MLPConfig(
            input_size=5,
            hidden_sizes=(4,),
            output_size=1,
            activation_function="relu",
            regularization=None,
            random_seed=42,
        )
        model = MLP(config)
        x = torch.rand(2, 5)
        out = model(x)
        assert out.shape == (2, 1)
