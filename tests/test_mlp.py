import pytest
import torch
import torch.nn as nn
import numpy as np
import pydantic
from unittest.mock import patch
from torch.utils.data import DataLoader, TensorDataset

from ThreeWToolkit.models.mlp import MLP, MLPConfig
from ThreeWToolkit.trainer.strategies.epoch_strategy import EpochTrainingStrategy
from ThreeWToolkit.assessment.strategies.torch_prediction_strategy import (
    TorchPredictionStrategy,
)


@pytest.fixture
def base_config():
    return MLPConfig(
        input_size=10,
        hidden_sizes=(8, 4),
        output_size=2,
        activation_function="relu",
        random_seed=42,
    )


@pytest.fixture
def dynamic_config():
    """Config with input_size=None for lazy build."""
    return MLPConfig(
        input_size=None,
        hidden_sizes=(8, 4),
        output_size=2,
        activation_function="relu",
        random_seed=42,
    )


@pytest.fixture
def base_model(base_config):
    return MLP(base_config)


class TestMLPConfig:
    def test_valid_config(self, base_config):
        assert base_config.input_size == 10
        assert base_config.output_size == 2

    def test_input_size_zero_raises(self):
        with pytest.raises(pydantic.ValidationError, match="input_size.*> 0"):
            MLPConfig(input_size=0, hidden_sizes=(4,), output_size=1)

    def test_input_size_negative_raises(self):
        with pytest.raises(pydantic.ValidationError, match="input_size.*> 0"):
            MLPConfig(input_size=-5, hidden_sizes=(4,), output_size=1)

    def test_input_size_none_is_valid(self, dynamic_config):
        assert dynamic_config.input_size is None

    def test_output_size_zero_raises(self):
        with pytest.raises(pydantic.ValidationError):
            MLPConfig(input_size=5, hidden_sizes=(4,), output_size=0)

    def test_hidden_sizes_negative_raises(self):
        with pytest.raises(
            pydantic.ValidationError,
            match="hidden_sizes must be a tuple of positive integers",
        ):
            MLPConfig(input_size=5, hidden_sizes=(-4,), output_size=1)

    def test_invalid_activation_raises(self):
        with pytest.raises(
            pydantic.ValidationError, match="activation_function must be one of"
        ):
            MLPConfig(
                input_size=5,
                hidden_sizes=(4,),
                output_size=1,
                activation_function="lelu",
            )

    def test_is_input_size_dynamic_true(self, dynamic_config):
        assert dynamic_config.is_input_size_dynamic() is True

    def test_is_input_size_dynamic_false(self, base_config):
        assert base_config.is_input_size_dynamic() is False

    def test_set_inferred_input_size(self, dynamic_config):
        dynamic_config.set_inferred_input_size(16)
        assert dynamic_config.input_size == 16

    def test_set_inferred_input_size_invalid_raises(self, dynamic_config):
        with pytest.raises(ValueError, match="Inferred input_size must be > 0"):
            dynamic_config.set_inferred_input_size(0)

    @pytest.mark.parametrize("regularization", [0.0, 0.01, 1.0])
    def test_valid_regularization(self, regularization):
        config = MLPConfig(
            input_size=5,
            hidden_sizes=(4,),
            output_size=1,
            regularization=regularization,
        )
        assert config.regularization == regularization

    def test_negative_regularization_raises(self):
        with pytest.raises(pydantic.ValidationError):
            MLPConfig(
                input_size=5, hidden_sizes=(4,), output_size=1, regularization=-0.1
            )


class TestMLPArchitecture:
    @pytest.mark.parametrize(
        "activation, expected_type",
        [
            ("relu", nn.ReLU),
            ("sigmoid", nn.Sigmoid),
            ("tanh", nn.Tanh),
        ],
    )
    def test_activation_functions(self, activation, expected_type):
        config = MLPConfig(
            input_size=10,
            hidden_sizes=(8, 4),
            output_size=1,
            activation_function=activation,
        )
        model = MLP(config)
        assert isinstance(model.model[1], expected_type)

    def test_unknown_activation_raises(self, base_model):
        with pytest.raises(ValueError, match="Unknown activation function"):
            base_model._get_activation_function("notreal")

    def test_forward_known_input_size(self, base_model):
        x = torch.randn(5, 10)
        out = base_model(x)
        assert out.shape == (5, 2)

    def test_forward_dynamic_input_size(self, dynamic_config):
        model = MLP(dynamic_config)
        assert not model._layers_built
        x = torch.randn(5, 12)
        out = model(x)
        assert out.shape == (5, 2)
        assert model._layers_built
        assert dynamic_config.input_size == 12

    def test_layers_structure(self, base_model):
        # hidden_sizes=(8,4), output=2 → Linear, Act, Linear, Act, Linear
        assert isinstance(base_model.model[0], nn.Linear)
        assert isinstance(base_model.model[-1], nn.Linear)
        assert base_model.model[-1].out_features == 2

    def test_output_layer_has_no_activation(self, base_model):
        assert isinstance(base_model.model[-1], nn.Linear)

    def test_set_inferred_input_size_except_branch(self):
        """Força o except simulando um modelo cujo __setattr__ rejeita a atribuição."""

        class FrozenConfig(MLPConfig):
            def __setattr__(self, name, value):
                if name == "input_size" and value is not None:
                    raise AttributeError("simulated frozen")
                super().__setattr__(name, value)

        frozen = FrozenConfig(input_size=None, hidden_sizes=(4,), output_size=1)
        frozen.set_inferred_input_size(16)
        assert frozen.input_size == 16


class TestMLPBehavior:
    def test_get_params_with_built_model(self, base_model):
        params = list(base_model.get_params())
        assert len(params) > 0
        assert all(isinstance(p, torch.Tensor) for p in params)

    def test_get_params_unbuilt_model_returns_dummy(self, dynamic_config):
        model = MLP(dynamic_config)
        params = list(model.get_params())
        assert len(params) == 1  # dummy param

    def test_get_training_strategy(self, base_model):
        assert base_model.get_training_strategy() is EpochTrainingStrategy

    def test_get_prediction_strategy(self, base_model):
        assert base_model.get_prediction_strategy() is TorchPredictionStrategy

    def test_save_delegates_to_recorder(self, base_model, tmp_path):
        path = tmp_path / "model.pth"
        with patch(
            "ThreeWToolkit.models.mlp.ModelRecorder.save_best_model"
        ) as mock_save:
            base_model.save(path)
            mock_save.assert_called_once_with(base_model, path)

    def test_load_delegates_to_recorder(self, base_model, tmp_path):
        path = tmp_path / "model.pth"
        with patch("ThreeWToolkit.models.mlp.ModelRecorder.load_model") as mock_load:
            result = base_model.load(path)
            mock_load.assert_called_once_with(path, model=base_model)
            assert result is base_model

    def test_predict_multiclass(self):
        """End-to-end prediction smoke test."""
        from ThreeWToolkit.core.enums import TaskTypeEnum

        n, input_size, num_classes = 20, 4, 3
        x = torch.rand(n, input_size)
        y = torch.randint(0, num_classes, (n,))
        loader = DataLoader(TensorDataset(x, y), batch_size=5)

        config = MLPConfig(
            input_size=input_size,
            hidden_sizes=(6,),
            output_size=num_classes,
            random_seed=42,
        )
        model = config.setup()
        strategy = model.get_prediction_strategy()()
        preds = strategy.predict(
            model, TaskTypeEnum.CLASSIFICATION, loader=loader, device="cpu"
        )

        assert preds.shape == (n,)
        assert np.all((preds >= 0) & (preds < num_classes))


# class TestModelTrainer:
#     def test_mlp_regression_else_branch(self, trainer_setup):
#         # Use trainer_setup for regression (output_size=1, criterion=mse)
#         trainer = trainer_setup["trainer"]
#         x = trainer_setup["x_tensor"]
#         y = trainer_setup["y_tensor"]
#         trainer.train(x, y)
#         hist = trainer.history[0]
#         if hist is not None:
#             assert "train_loss" in hist and "val_loss" in hist

#     def test_mlp_multiclass_training(self):
#         # Multiclass: output_size > 1, labels are class indices
#         input_size = 6
#         num_classes = 3
#         n_samples = 40
#         x = np.random.rand(n_samples, input_size).astype(np.float32)
#         y = np.random.randint(0, num_classes, size=n_samples)
#         x_df = pd.DataFrame(x, columns=[f"f{i}" for i in range(input_size)])
#         y_series = pd.Series(y, name="label")
#         config = MLPConfig(
#             input_size=input_size,
#             hidden_sizes=(12, 8),
#             output_size=num_classes,
#             activation_function="relu",
#             regularization=None,
#             random_seed=42,
#         )
#         trainer_config = TrainerConfig(
#             batch_size=8,
#             epochs=2,
#             seed=42,
#             learning_rate=1e-3,
#             config_model=config,
#             optimizer="adam",
#             criterion="cross_entropy",
#             device="cpu",
#             shuffle_train=True,
#         )
#         trainer = ModelTrainer(trainer_config)
#         trainer.train(x_df, y_series)
#         # Should have a history with train_loss and val_loss
#         hist = trainer.history[0]
#         if hist is not None:
#             assert "train_loss" in hist and "val_loss" in hist

#     def test_trainer_initialization(self, trainer_setup):
#         trainer = trainer_setup["trainer"]
#         assert trainer.batch_size == 16
#         # Trainer.config is TrainerConfig, check config_model is MLPConfig
#         assert isinstance(trainer.config.config_model, MLPConfig)

#     def test_train_and_history(self, trainer_setup):
#         trainer = trainer_setup["trainer"]
#         x = trainer_setup["x_tensor"]
#         y = trainer_setup["y_tensor"]
#         trainer.train(x, y)
#         # Should have a history with train_loss and val_loss
#         hist = trainer.history[0]
#         assert "train_loss" in hist and "val_loss" in hist
#         assert len(hist["train_loss"]) == trainer.epochs

#     def test_mlp_forward(self):
#         config = MLPConfig(
#             input_size=5,
#             hidden_sizes=(4,),
#             output_size=1,
#             activation_function="relu",
#             regularization=None,
#             random_seed=42,
#         )
#         model = MLP(config)
#         x = torch.rand(2, 5)
#         out = model(x)
#         assert out.shape == (2, 1)
