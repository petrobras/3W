import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pydantic

from torch.utils.data import DataLoader, TensorDataset

from ThreeWToolkit.models.mlp import (
    MLP,
    MLPConfig,
    ActivationFunctionEnum,
)
from ThreeWToolkit.trainer.trainer import ModelTrainer, TrainerConfig


@pytest.fixture
def trainer_setup():
    """
    Pytest fixture to set up a standard ModelTrainer and data for testing.
    """
    input_size = 8
    x = np.random.rand(50, input_size).astype(np.float32)
    # Create a binary label for demonstration, as int
    y = (np.random.rand(50) > 0.5).astype(int)
    x_df = pd.DataFrame(x, columns=[f"f{i}" for i in range(input_size)])
    y_series = pd.Series(y, name="label")
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
        shuffle_train=True,
    )
    trainer = ModelTrainer(trainer_config)
    return {
        "x_tensor": x_df,
        "y_tensor": y_series,
        "config": config,
        "trainer": trainer,
        "device": trainer.device,
    }


class TestMLP:
    def test_predict_multiclass(self):
        # Multiclass: output_size > 1
        input_size = 4
        num_classes = 3
        n_samples = 20
        x = np.random.rand(n_samples, input_size).astype(np.float32)
        y = np.random.randint(0, num_classes, size=n_samples)
        dataset = TensorDataset(torch.tensor(x), torch.tensor(y))
        loader = DataLoader(dataset, batch_size=5)
        config = MLPConfig(
            input_size=input_size,
            hidden_sizes=(6,),
            output_size=num_classes,
            activation_function="relu",
            regularization=None,
            random_seed=42,
        )
        model = MLP(config)
        preds = model.predict(loader, device="cpu")
        # Should be shape (n_samples,) and values in 0..num_classes-1
        assert preds.shape == (n_samples,)
        assert np.all((preds >= 0) & (preds < num_classes))

    def test_get_activation_function(self):
        config = MLPConfig(
            input_size=5,
            hidden_sizes=(4,),
            output_size=1,
            activation_function="relu",
            regularization=None,
            random_seed=42,
        )
        model = MLP(config)
        with pytest.raises(
            ValueError, match="Unknown activation function: notarealactivation"
        ):
            model._get_activation_function("notarealactivation")

    def test_invalid_activation_function(self):
        with pytest.raises(
            pydantic.ValidationError, match="activation_function must be one of"
        ):
            MLPConfig(
                input_size=5,
                hidden_sizes=(4,),
                output_size=1,
                activation_function="notarealactivation",
                regularization=None,
                random_seed=42,
            )

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

    def test_check_hidden_sizes(self):
        with pytest.raises(
            pydantic.ValidationError, match="hidden_sizes must be a tuple"
        ):
            MLPConfig(
                input_size=5,
                hidden_sizes=(-4,),
                output_size=1,
                activation_function=ActivationFunctionEnum.RELU.value,
                regularization=None,
                random_seed=42,
            )


class TestModelTrainer:
    def test_mlp_regression_else_branch(self, trainer_setup):
        # Use trainer_setup for regression (output_size=1, criterion=mse)
        trainer = trainer_setup["trainer"]
        x = trainer_setup["x_tensor"]
        y = trainer_setup["y_tensor"]
        trainer.train(x, y)
        hist = trainer.history[0]
        if hist is not None:
            assert "train_loss" in hist and "val_loss" in hist

    def test_mlp_multiclass_training(self):
        # Multiclass: output_size > 1, labels are class indices
        input_size = 6
        num_classes = 3
        n_samples = 40
        x = np.random.rand(n_samples, input_size).astype(np.float32)
        y = np.random.randint(0, num_classes, size=n_samples)
        x_df = pd.DataFrame(x, columns=[f"f{i}" for i in range(input_size)])
        y_series = pd.Series(y, name="label")
        config = MLPConfig(
            input_size=input_size,
            hidden_sizes=(12, 8),
            output_size=num_classes,
            activation_function="relu",
            regularization=None,
            random_seed=42,
        )
        trainer_config = TrainerConfig(
            batch_size=8,
            epochs=2,
            seed=42,
            learning_rate=1e-3,
            config_model=config,
            optimizer="adam",
            criterion="cross_entropy",
            device="cpu",
            shuffle_train=True,
        )
        trainer = ModelTrainer(trainer_config)
        trainer.train(x_df, y_series)
        # Should have a history with train_loss and val_loss
        hist = trainer.history[0]
        if hist is not None:
            assert "train_loss" in hist and "val_loss" in hist

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

    def test_save_and_load(self, tmp_path, trainer_setup):
        trainer = trainer_setup["trainer"]
        x = trainer_setup["x_tensor"]
        y = trainer_setup["y_tensor"]
        trainer.train(x, y)
        save_path = tmp_path / "model.pth"
        trainer.save(save_path)
        loaded_model = trainer.load(save_path)
        assert loaded_model is not None

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
