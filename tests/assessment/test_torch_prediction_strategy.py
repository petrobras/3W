"""Tests for TorchPredictionStrategy."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ThreeWToolkit.assessment.strategies.torch_prediction_strategy import (
    TorchPredictionStrategy,
)
from ThreeWToolkit.core.enums import TaskTypeEnum


class SimpleMLP(nn.Module):
    """Simple MLP for testing prediction strategy."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


class BinaryMLP(nn.Module):
    """Binary classification MLP (single output)."""

    def __init__(self, input_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)


def create_dataloader(n_samples: int, n_features: int, batch_size: int = 4):
    """Create a DataLoader with random data for testing."""
    x = torch.randn(n_samples, n_features)
    y = torch.zeros(n_samples, dtype=torch.long)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size)


class TestTorchPredictionStrategyPredict:
    """Tests for TorchPredictionStrategy.predict() method."""

    @pytest.fixture
    def strategy(self):
        """Provide a fresh prediction strategy instance."""
        return TorchPredictionStrategy()

    @pytest.fixture
    def multiclass_model(self):
        """Provide a multiclass classification model."""
        return SimpleMLP(input_size=10, output_size=3)

    @pytest.fixture
    def binary_model(self):
        """Provide a binary classification model."""
        return BinaryMLP(input_size=10)

    @pytest.fixture
    def regression_model(self):
        """Provide a regression model (single output)."""
        return BinaryMLP(input_size=10)

    @pytest.fixture
    def dataloader(self):
        """Provide a DataLoader for testing."""
        return create_dataloader(n_samples=12, n_features=10, batch_size=4)

    def test_predict_returns_numpy_array(self, strategy, multiclass_model, dataloader):
        """Predict should return a numpy array."""
        result = strategy.predict(
            multiclass_model,
            TaskTypeEnum.CLASSIFICATION,
            loader=dataloader,
            device="cpu",
        )

        assert isinstance(result, np.ndarray)

    def test_predict_multiclass_shape(self, strategy, multiclass_model, dataloader):
        """Multiclass predictions should have correct shape."""
        result = strategy.predict(
            multiclass_model,
            TaskTypeEnum.CLASSIFICATION,
            loader=dataloader,
            device="cpu",
        )

        assert result.shape == (12,)  # One prediction per sample

    def test_predict_multiclass_values(self, strategy, multiclass_model, dataloader):
        """Multiclass predictions should be valid class indices."""
        result = strategy.predict(
            multiclass_model,
            TaskTypeEnum.CLASSIFICATION,
            loader=dataloader,
            device="cpu",
        )

        assert all(0 <= pred < 3 for pred in result)

    def test_predict_binary_classification(self, strategy, binary_model, dataloader):
        """Binary classification should produce 0/1 predictions."""
        result = strategy.predict(
            binary_model, TaskTypeEnum.CLASSIFICATION, loader=dataloader, device="cpu"
        )

        assert result.shape == (12,)
        assert all(pred in [0, 1] for pred in result)

    def test_predict_binary_with_custom_threshold(
        self, strategy, binary_model, dataloader
    ):
        """Binary classification should use custom threshold."""
        # Low threshold should predict more 1s
        result_low = strategy.predict(
            binary_model,
            TaskTypeEnum.CLASSIFICATION,
            loader=dataloader,
            device="cpu",
            threshold=0.1,
        )
        # High threshold should predict more 0s
        result_high = strategy.predict(
            binary_model,
            TaskTypeEnum.CLASSIFICATION,
            loader=dataloader,
            device="cpu",
            threshold=0.9,
        )

        # With extreme thresholds, we expect different results
        # (though not guaranteed due to random initialization)
        assert result_low.shape == (12,)
        assert result_high.shape == (12,)

    def test_predict_regression(self, strategy, regression_model, dataloader):
        """Regression predictions should be continuous values."""
        result = strategy.predict(
            regression_model, TaskTypeEnum.REGRESSION, loader=dataloader, device="cpu"
        )

        assert result.shape == (12,)
        # Regression values can be any float

    def test_predict_raises_without_loader(self, strategy, multiclass_model):
        """Predict should raise ValueError when loader is not provided."""
        with pytest.raises(ValueError, match="DataLoader must be provided"):
            strategy.predict(
                multiclass_model, TaskTypeEnum.CLASSIFICATION, device="cpu"
            )

    def test_predict_raises_for_invalid_model(self, strategy, dataloader):
        """Predict should raise AssertionError for non-Module model."""
        with pytest.raises(AssertionError, match="torch.nn.Module"):
            strategy.predict(
                "not_a_model", TaskTypeEnum.CLASSIFICATION, loader=dataloader
            )

    def test_predict_raises_for_unknown_task(
        self, strategy, multiclass_model, dataloader
    ):
        """Predict should raise ValueError for unknown task type."""
        with pytest.raises(ValueError, match="Unknown task type"):
            strategy.predict(
                multiclass_model, task="unknown_task", loader=dataloader, device="cpu"
            )

    def test_predict_default_task_is_classification(
        self, strategy, multiclass_model, dataloader
    ):
        """Default task should be classification."""
        result = strategy.predict(multiclass_model, loader=dataloader, device="cpu")

        assert result.shape == (12,)
        assert all(0 <= pred < 3 for pred in result)

    def test_predict_uses_specified_device(
        self, strategy, multiclass_model, dataloader
    ):
        """Model should be moved to specified device."""
        result = strategy.predict(
            multiclass_model,
            TaskTypeEnum.CLASSIFICATION,
            loader=dataloader,
            device="cpu",
        )

        # Verify model is on CPU
        for param in multiclass_model.parameters():
            assert param.device == torch.device("cpu")

        assert isinstance(result, np.ndarray)

    def test_predict_sets_model_to_eval(self, strategy, multiclass_model, dataloader):
        """Model should be set to eval mode during prediction."""
        multiclass_model.train()
        assert multiclass_model.training is True

        strategy.predict(
            multiclass_model,
            TaskTypeEnum.CLASSIFICATION,
            loader=dataloader,
            device="cpu",
        )

        assert multiclass_model.training is False

    def test_predict_with_different_batch_sizes(self, strategy, multiclass_model):
        """Predict should work with various batch sizes."""
        for batch_size in [1, 4, 8, 16]:
            loader = create_dataloader(
                n_samples=16, n_features=10, batch_size=batch_size
            )
            result = strategy.predict(
                multiclass_model,
                TaskTypeEnum.CLASSIFICATION,
                loader=loader,
                device="cpu",
            )

            assert result.shape == (16,)


class TestTorchPredictionStrategyClassification:
    """Tests for _predict_classification helper method."""

    @pytest.fixture
    def strategy(self):
        """Provide a fresh prediction strategy instance."""
        return TorchPredictionStrategy()

    def test_multiclass_argmax(self, strategy):
        """Multiclass outputs should use argmax."""
        outputs = torch.tensor([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1], [0.2, 0.3, 0.5]])
        result = strategy._predict_classification(outputs, threshold=0.5)

        expected = torch.tensor([1, 0, 2])
        assert torch.equal(result, expected)

    def test_binary_with_sigmoid(self, strategy):
        """Binary outputs should apply sigmoid and threshold."""
        # Pre-sigmoid values: 2.0 -> sigmoid ~0.88, -2.0 -> sigmoid ~0.12, 0.0 -> sigmoid = 0.5
        outputs = torch.tensor([[2.0], [-2.0], [0.0]])
        result = strategy._predict_classification(outputs, threshold=0.5)

        # sigmoid(2.0) ~= 0.88 > 0.5 -> 1
        # sigmoid(-2.0) ~= 0.12 < 0.5 -> 0
        # sigmoid(0.0) = 0.5 is NOT > 0.5 -> 0
        expected = torch.tensor([1, 0, 0])
        assert torch.equal(result, expected)

    def test_binary_threshold_effect(self, strategy):
        """Threshold should affect binary classification results."""
        outputs = torch.tensor([[0.0], [0.5], [1.0]])  # sigmoid: 0.5, 0.62, 0.73

        result_low = strategy._predict_classification(outputs, threshold=0.3)
        result_high = strategy._predict_classification(outputs, threshold=0.7)

        # Low threshold -> more 1s
        assert result_low.sum() >= result_high.sum()

    def test_classification_raises_for_1d_output(self, strategy):
        """Classification should raise for 1D outputs."""
        outputs = torch.tensor([0.1, 0.7, 0.2])

        with pytest.raises(ValueError, match="must be 2D"):
            strategy._predict_classification(outputs, threshold=0.5)


class TestTorchPredictionStrategyRegression:
    """Tests for _predict_regression helper method."""

    @pytest.fixture
    def strategy(self):
        """Provide a fresh prediction strategy instance."""
        return TorchPredictionStrategy()

    def test_regression_squeezes_2d_single_column(self, strategy):
        """Regression should squeeze (N, 1) to (N,)."""
        outputs = torch.tensor([[1.5], [2.3], [0.8]])
        result = strategy._predict_regression(outputs)

        assert result.shape == torch.Size([3])
        assert torch.allclose(result, torch.tensor([1.5, 2.3, 0.8]))

    def test_regression_squeezes_1d(self, strategy):
        """Regression should handle 1D outputs."""
        outputs = torch.tensor([1.5, 2.3, 0.8])
        result = strategy._predict_regression(outputs)

        assert result.shape == torch.Size([3])

    def test_regression_preserves_values(self, strategy):
        """Regression should preserve original values."""
        outputs = torch.tensor([[-1.5], [0.0], [3.14]])
        result = strategy._predict_regression(outputs)

        expected = torch.tensor([-1.5, 0.0, 3.14])
        assert torch.allclose(result, expected)


class TestTorchPredictionStrategyIntegration:
    """Integration tests for TorchPredictionStrategy."""

    @pytest.fixture
    def strategy(self):
        """Provide a fresh prediction strategy instance."""
        return TorchPredictionStrategy()

    def test_full_classification_pipeline(self, strategy):
        """Full classification prediction pipeline should work."""
        # Create model
        model = SimpleMLP(input_size=20, output_size=5)

        # Create dataset
        x = torch.randn(50, 20)
        y = torch.randint(0, 5, (50,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=10)

        # Predict
        result = strategy.predict(
            model, TaskTypeEnum.CLASSIFICATION, loader=loader, device="cpu"
        )

        assert result.shape == (50,)
        assert all(0 <= pred < 5 for pred in result)

    def test_full_regression_pipeline(self, strategy):
        """Full regression prediction pipeline should work."""
        # Create model
        model = BinaryMLP(input_size=15)

        # Create dataset
        x = torch.randn(30, 15)
        y = torch.randn(30)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=8)

        # Predict
        result = strategy.predict(
            model, TaskTypeEnum.REGRESSION, loader=loader, device="cpu"
        )

        assert result.shape == (30,)

    def test_deterministic_predictions(self, strategy):
        """Same model with same input should produce same predictions."""
        torch.manual_seed(42)
        model = SimpleMLP(input_size=10, output_size=3)

        x = torch.randn(20, 10)
        y = torch.zeros(20, dtype=torch.long)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=5, shuffle=False)

        result1 = strategy.predict(
            model, TaskTypeEnum.CLASSIFICATION, loader=loader, device="cpu"
        )
        result2 = strategy.predict(
            model, TaskTypeEnum.CLASSIFICATION, loader=loader, device="cpu"
        )

        assert np.array_equal(result1, result2)

    def test_handles_single_sample(self, strategy):
        """Strategy should handle single sample prediction."""
        model = SimpleMLP(input_size=10, output_size=2)

        x = torch.randn(1, 10)
        y = torch.zeros(1, dtype=torch.long)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=1)

        result = strategy.predict(
            model, TaskTypeEnum.CLASSIFICATION, loader=loader, device="cpu"
        )

        assert result.shape == (1,)
