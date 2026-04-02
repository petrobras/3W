"""Tests for MLP model and MLPConfig."""

from pydantic import ValidationError
from pathlib import Path
import pytest
import torch
from torch import nn
from ThreeWToolkit.constants import CHECKPOINT_DIR
from ThreeWToolkit.models.mlp import MLP, MLPConfig
from ThreeWToolkit.core.enums import ModelTypeEnum


class TestMLPConfig:
    """Tests for MLPConfig validation and behavior."""

    def test_valid_config(self):
        """Config with all valid parameters should work."""
        config = MLPConfig(
            hidden_sizes=(64, 32),
            output_size=3,
            input_size=100,
            activation_function=nn.ReLU(),
        )
        assert config.hidden_sizes == list((64, 32))
        assert config.output_size == 3
        assert config.input_size == 100

    def test_config_without_input_size(self):
        """Config without input_size should work (dynamic inference)."""
        config = MLPConfig(hidden_sizes=(64,), output_size=2)
        assert config.input_size is None
        assert config.is_input_size_dynamic

    def test_config_with_input_size(self):
        """Config with input_size should not be dynamic."""
        config = MLPConfig(hidden_sizes=(64,), output_size=2, input_size=50)
        assert config.input_size == 50
        assert not config.is_input_size_dynamic

    @pytest.mark.parametrize("activation", [nn.ReLU(), nn.Sigmoid(), nn.Tanh()])
    def test_valid_activation_functions(self, activation):
        """All valid activation functions should be accepted."""
        config = MLPConfig(
            hidden_sizes=(32,), output_size=2, activation_function=activation
        )
        assert config.activation_function == activation

    def test_invalid_activation_function(self):
        """Invalid activation function should raise ValueError."""
        with pytest.raises(ValidationError):
            MLPConfig(hidden_sizes=(32,), output_size=2, activation_function=None) # type: ignore

    def test_invalid_input_size_zero(self):
        """input_size of 0 should raise ValueError."""
        with pytest.raises(ValidationError):
            MLPConfig(hidden_sizes=(32,), output_size=2, input_size=0)

    def test_invalid_input_size_negative(self):
        """Negative input_size should raise ValueError."""
        with pytest.raises(ValidationError):
            MLPConfig(hidden_sizes=(32,), output_size=2, input_size=-5)

    def test_invalid_output_size_zero(self):
        """output_size of 0 should raise ValidationError."""
        with pytest.raises(ValidationError):
            MLPConfig(hidden_sizes=(32,), output_size=0)

    def test_empty_hidden_sizes(self):
        """Empty hidden_sizes tuple should raise ValueError."""
        with pytest.raises(ValidationError):
            MLPConfig(hidden_sizes=(), output_size=2)

    def test_invalid_hidden_sizes_negative(self):
        """Negative values in hidden_sizes should raise ValueError."""
        with pytest.raises(ValidationError):
            MLPConfig(hidden_sizes=(64, -32), output_size=2)

    def test_invalid_hidden_sizes_zero(self):
        """Zero value in hidden_sizes should raise ValueError."""
        with pytest.raises(ValidationError):
            MLPConfig(hidden_sizes=(64, 0), output_size=2)

    def test_set_inferred_input_size(self):
        """set_inferred_input_size should update input_size."""
        config = MLPConfig(hidden_sizes=(32,), output_size=2)
        assert config.input_size is None
        config.set_inferred_input_size(100)
        assert config.input_size == 100

    def test_set_inferred_input_size_invalid(self):
        """set_inferred_input_size with invalid value should raise."""
        config = MLPConfig(hidden_sizes=(32,), output_size=2)
        with pytest.raises(ValueError, match="Inferred input_size must be > 0"):
            config.set_inferred_input_size(0)

    def test_model_type_default(self):
        """Default model_type should be MLP."""
        config = MLPConfig(hidden_sizes=(32,), output_size=2)
        assert config.model_type is MLP

    def test_target_returns_mlp_class(self):
        """_target should return MLP class."""
        config = MLPConfig(hidden_sizes=(32,), output_size=2)
        assert config._target == MLP


class TestMLP:
    """Tests for MLP model."""

    @pytest.fixture
    def basic_config(self):
        """Provide a basic MLPConfig for testing."""
        return MLPConfig(hidden_sizes=(64, 32), output_size=3, input_size=100)

    @pytest.fixture
    def dynamic_config(self):
        """Provide a dynamic input MLPConfig for testing."""
        return MLPConfig(hidden_sizes=(64, 32), output_size=3)

    def test_model_creation_with_input_size(self, basic_config):
        """Model with input_size should build layers immediately."""
        model = MLP(basic_config)
        assert model.model is not None

    def test_model_creation_without_input_size(self, dynamic_config):
        """Model without input_size should fail building """

        with pytest.raises(ValueError):
            model = MLP(dynamic_config)
            assert model.model is None

    def test_forward_pass_with_input_size(self, basic_config):
        """Forward pass should work with pre-built layers."""
        model = MLP(basic_config)
        x = torch.randn(8, 100)
        output = model(x)
        assert output.shape == (8, 3)

    @pytest.mark.parametrize("batch_size", [1, 16, 64])
    def test_forward_various_batch_sizes(self, basic_config, batch_size):
        """Model should handle various batch sizes."""
        model = MLP(basic_config)
        x = torch.randn(batch_size, 100)
        output = model(x)
        assert output.shape == (batch_size, 3)

    @pytest.mark.parametrize("activation", [nn.ReLU(), nn.Sigmoid(), nn.Tanh()])
    def test_activation_functions(self, activation):
        """Model should use correct activation function."""
        config = MLPConfig(
            hidden_sizes=(32,),
            output_size=2,
            input_size=10,
            activation_function=activation,
        )
        model = MLP(config)
        assert model.activation_func is not None
        x = torch.randn(4, 10)
        output = model(x)
        assert output.shape == (4, 2)

    def test_get_params_with_built_model(self, basic_config):
        """get_params should return model parameters when built."""
        model = MLP(basic_config)
        params = list(model.get_params())
        assert len(params) > 0
        assert all(isinstance(p, torch.nn.Parameter) for p in params)

    def test_save_and_load(self, basic_config):
        """Model should save and load correctly."""

        model = MLP(basic_config)
        x = torch.randn(4, 100)
        original_output = model(x)

        filename = "test_mlp_save_load.pth"
        model.save(filename)

        new_model = MLP.load(filename)

        loaded_output = new_model(x)
        torch.testing.assert_close(original_output, loaded_output)

        # Cleanup
        Path(filename).unlink(missing_ok=True)

    def test_layer_structure(self, basic_config):
        """Layers should match config specification."""
        model = MLP(basic_config)
        layers = list(model.modules())
        # Linear layers + activation layers
        # Expected: Linear(100, 64), ReLU, Linear(64, 32), ReLU, Linear(32, 3)
        # Note: ReLU may be shared in Sequential, so count Linear layers
        linear_layers = [layer for layer in layers if isinstance(layer, torch.nn.Linear)]
        assert len(linear_layers) == 3
        assert linear_layers[0].in_features == 100
        assert linear_layers[0].out_features == 64
        assert linear_layers[1].in_features == 64
        assert linear_layers[1].out_features == 32
        assert linear_layers[2].in_features == 32
        assert linear_layers[2].out_features == 3

    def test_multiple_hidden_layers(self):
        """Model should support multiple hidden layers."""
        config = MLPConfig(
            hidden_sizes=(128, 64, 32, 16), output_size=5, input_size=256
        )
        model = MLP(config)
        x = torch.randn(4, 256)
        output = model(x)
        assert output.shape == (4, 5)

    def test_single_hidden_layer(self):
        """Model should work with single hidden layer."""
        config = MLPConfig(hidden_sizes=(32,), output_size=2, input_size=10)
        model = MLP(config)
        x = torch.randn(4, 10)
        output = model(x)
        assert output.shape == (4, 2)


class TestMLPIntegration:
    """Integration tests for MLP with realistic scenarios."""

    def test_binary_classification(self):
        """MLP should work for binary classification."""
        config = MLPConfig(hidden_sizes=(64, 32), output_size=2, input_size=50)
        model = MLP(config)
        x = torch.randn(32, 50)
        output = model(x)
        probs = torch.softmax(output, dim=1)
        assert probs.shape == (32, 2)
        assert torch.allclose(probs.sum(dim=1), torch.ones(32), atol=1e-6)

    def test_multiclass_classification(self):
        """MLP should work for multi-class classification."""
        config = MLPConfig(hidden_sizes=(128, 64), output_size=10, input_size=100)
        model = MLP(config)
        x = torch.randn(16, 100)
        output = model(x)
        probs = torch.softmax(output, dim=1)
        assert probs.shape == (16, 10)
        assert torch.allclose(probs.sum(dim=1), torch.ones(16), atol=1e-6)

    def test_gradient_flow(self):
        """Gradients should flow through the model."""
        config = MLPConfig(hidden_sizes=(32,), output_size=2, input_size=10)
        model = MLP(config)
        x = torch.randn(4, 10, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
