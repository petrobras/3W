import pytest
import torch
import torch.nn as nn
import numpy as np
import pydantic

from pathlib import Path
from unittest.mock import patch
from torch.utils.data import DataLoader, TensorDataset

from ThreeWToolkit.core.enums import TaskTypeEnum
from ThreeWToolkit.models.mlp import MLP, MLPConfig
from ThreeWToolkit.trainer.strategies.epoch_strategy import EpochTrainingStrategy
from ThreeWToolkit.assessment.strategies.torch_prediction_strategy import (
    TorchPredictionStrategy,
)


@pytest.fixture
def base_config():
    """Creates a valid MLPConfig instance for testing.

    Returns:
        MLPConfig: Configuration with fixed input size and two hidden layers.
    """
    return MLPConfig(
        input_size=10,
        hidden_sizes=(8, 4),
        output_size=2,
        activation_function="relu",
        random_seed=42,
    )


@pytest.fixture
def dynamic_config():
    """Creates an MLPConfig with dynamic input size.

    This configuration allows the model to infer the input size
    during the first forward pass.

    Returns:
        MLPConfig: Configuration with ``input_size=None``.
    """
    return MLPConfig(
        input_size=None,
        hidden_sizes=(8, 4),
        output_size=2,
        activation_function="relu",
        random_seed=42,
    )


@pytest.fixture
def base_model(base_config):
    """Creates an initialized MLP model using the base configuration.

    Args:
        base_config (MLPConfig): Valid configuration fixture.

    Returns:
        MLP: Instantiated MLP model.
    """
    return MLP(base_config)


class TestMLPConfig:
    """Unit tests for the MLPConfig validation and behavior."""

    def test_valid_config(self, base_config: MLPConfig):
        """Ensures that a valid configuration is correctly initialized."""
        assert base_config.input_size == 10
        assert base_config.output_size == 2

    def test_input_size_zero_raises(self):
        """Ensures that input_size equal to zero raises a validation error."""
        with pytest.raises(pydantic.ValidationError, match="input_size.*> 0"):
            MLPConfig(input_size=0, hidden_sizes=(4,), output_size=1)

    def test_input_size_negative_raises(self):
        """Ensures that negative input_size raises a validation error."""
        with pytest.raises(pydantic.ValidationError, match="input_size.*> 0"):
            MLPConfig(input_size=-5, hidden_sizes=(4,), output_size=1)

    def test_input_size_none_is_valid(self, dynamic_config: MLPConfig):
        """Ensures that input_size=None is accepted for dynamic input inference."""
        assert dynamic_config.input_size is None

    def test_output_size_zero_raises(self):
        """Ensures that output_size equal to zero raises a validation error."""
        with pytest.raises(pydantic.ValidationError):
            MLPConfig(input_size=5, hidden_sizes=(4,), output_size=0)

    def test_hidden_sizes_negative_raises(self):
        """Ensures that hidden layer sizes must be positive integers."""
        with pytest.raises(
            pydantic.ValidationError,
            match="hidden_sizes must be a tuple of positive integers",
        ):
            MLPConfig(input_size=5, hidden_sizes=(-4,), output_size=1)

    def test_invalid_activation_raises(self):
        """Ensures that an invalid activation function raises a validation error."""
        with pytest.raises(
            pydantic.ValidationError, match="activation_function must be one of"
        ):
            MLPConfig(
                input_size=5,
                hidden_sizes=(4,),
                output_size=1,
                activation_function="lelu",
            )

    def test_is_input_size_dynamic_true(self, dynamic_config: MLPConfig):
        """Ensures dynamic input detection returns True when input_size is None."""
        assert dynamic_config.is_input_size_dynamic() is True

    def test_is_input_size_dynamic_false(self, base_config: MLPConfig):
        """Ensures dynamic input detection returns False when input_size is fixed."""
        assert base_config.is_input_size_dynamic() is False

    def test_set_inferred_input_size(self, dynamic_config: MLPConfig):
        """Ensures inferred input size can be set correctly."""
        dynamic_config.set_inferred_input_size(16)
        assert dynamic_config.input_size == 16

    def test_set_inferred_input_size_invalid_raises(self, dynamic_config: MLPConfig):
        """Ensures that invalid inferred input sizes raise an error."""
        with pytest.raises(ValueError, match="Inferred input_size must be > 0"):
            dynamic_config.set_inferred_input_size(0)

    @pytest.mark.parametrize("regularization", [0.0, 0.01, 1.0])
    def test_valid_regularization(self, regularization):
        """Ensures valid regularization values are accepted."""
        config = MLPConfig(
            input_size=5,
            hidden_sizes=(4,),
            output_size=1,
            regularization=regularization,
        )
        assert config.regularization == regularization

    def test_negative_regularization_raises(self):
        """Ensures negative regularization values raise validation errors."""
        with pytest.raises(pydantic.ValidationError):
            MLPConfig(
                input_size=5, hidden_sizes=(4,), output_size=1, regularization=-0.1
            )


class TestMLPArchitecture:
    """Unit tests validating the architecture and layer construction of MLP."""

    @pytest.mark.parametrize(
        "activation, expected_type",
        [
            ("relu", nn.ReLU),
            ("sigmoid", nn.Sigmoid),
            ("tanh", nn.Tanh),
        ],
    )
    def test_activation_functions(
        self, activation: str, expected_type: type[nn.Module]
    ):
        """Ensures that the correct activation layer is used in the network."""
        config = MLPConfig(
            input_size=10,
            hidden_sizes=(8, 4),
            output_size=1,
            activation_function=activation,
        )
        model = MLP(config)
        assert isinstance(model._get_activation_function(activation), expected_type)

    def test_unknown_activation_raises(self, base_model: MLP):
        """Ensures requesting an unknown activation function raises an error."""
        with pytest.raises(ValueError, match="Unknown activation function"):
            base_model._get_activation_function("notreal")

    def test_forward_known_input_size(self, base_model: MLP):
        """Ensures forward pass works with a predefined input size."""
        x = torch.randn(5, 10)
        out = base_model(x)
        assert out.shape == (5, 2)

    def test_forward_dynamic_input_size(self, dynamic_config: MLPConfig):
        """Ensures layers are lazily built when using dynamic input size."""
        model = MLP(dynamic_config)
        assert not model._layers_built
        x = torch.randn(5, 12)
        out = model(x)
        assert out.shape == (5, 2)
        assert model._layers_built
        assert dynamic_config.input_size == 12

    def test_layers_structure(self, base_model: MLP):
        """Ensures the MLP layers follow the expected architecture pattern."""
        # hidden_sizes=(8,4), output=2 → Linear, Act, Linear, Act, Linear
        assert base_model.model
        assert isinstance(base_model.model[0], nn.Linear)
        assert isinstance(base_model.model[-1], nn.Linear)
        assert base_model.model[-1].out_features == 2

    def test_output_layer_has_no_activation(self, base_model: MLP):
        """Ensures the output layer is linear without activation."""
        assert base_model.model
        assert isinstance(base_model.model[-1], nn.Linear)

    def test_set_inferred_input_size_except_branch(self):
        """Ensures inferred input assignment works even when attribute setting fails."""

        class FrozenConfig(MLPConfig):
            def __setattr__(self, name, value):
                if name == "input_size" and value is not None:
                    raise AttributeError("simulated frozen")
                super().__setattr__(name, value)

        frozen = FrozenConfig(input_size=None, hidden_sizes=(4,), output_size=1)
        frozen.set_inferred_input_size(16)
        assert frozen.input_size == 16


class TestMLPBehavior:
    """Unit tests validating runtime behavior and integrations of the MLP model."""

    def test_get_params_with_built_model(self, base_model: MLP):
        """Ensures parameters are returned correctly when layers are built."""
        params = list(base_model.get_params())
        assert len(params) > 0
        assert all(isinstance(p, torch.Tensor) for p in params)

    def test_get_params_unbuilt_model_returns_dummy(self, dynamic_config: MLPConfig):
        """Ensures a dummy parameter is returned when the model is not yet built."""
        model = MLP(dynamic_config)
        params = list(model.get_params())
        assert len(params) == 1  # dummy param

    def test_get_training_strategy(self, base_model: MLP):
        """Ensures the model returns the correct training strategy."""
        assert base_model.get_training_strategy() is EpochTrainingStrategy

    def test_get_prediction_strategy(self, base_model: MLP):
        """Ensures the model returns the correct prediction strategy."""
        assert base_model.get_prediction_strategy() is TorchPredictionStrategy

    def test_save_delegates_to_recorder(self, base_model: MLP, tmp_path: Path):
        """Ensures the save operation delegates to ModelRecorder."""
        path = tmp_path / "model.pth"
        with patch(
            "ThreeWToolkit.models.mlp.ModelRecorder.save_best_model"
        ) as mock_save:
            base_model.save(path)
            mock_save.assert_called_once_with(base_model, path)

    def test_load_delegates_to_recorder(self, base_model: MLP, tmp_path: Path):
        """Ensures the load operation delegates to ModelRecorder."""
        path = tmp_path / "model.pth"
        with patch("ThreeWToolkit.models.mlp.ModelRecorder.load_model") as mock_load:
            result = base_model.load(path)
            mock_load.assert_called_once_with(path, model=base_model)
            assert result is base_model

    def test_predict_multiclass(self):
        """Runs a full prediction pipeline for a multiclass classification task."""
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
