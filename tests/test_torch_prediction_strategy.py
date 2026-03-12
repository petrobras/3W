import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ThreeWToolkit.assessment.strategies.torch_prediction_strategy import (
    TorchPredictionStrategy,
)
from ThreeWToolkit.core.enums import TaskTypeEnum


def simple_model(input_size=4, output_size=1):
    """Creates a minimal PyTorch model for prediction tests.

    Args:
        input_size (int): Number of input features.
        output_size (int): Number of output units.

    Returns:
        torch.nn.Module: A simple linear model.
    """
    return nn.Sequential(nn.Linear(input_size, output_size))


def make_loader(n=8, input_size=4, batch_size=4):
    """Creates a DataLoader with random input data.

    Args:
        n (int): Number of samples.
        input_size (int): Number of input features.
        batch_size (int): Batch size used in the DataLoader.

    Returns:
        torch.utils.data.DataLoader: DataLoader containing random inputs.
    """
    x = torch.rand(n, input_size)
    y = torch.zeros(n)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


class TestTorchPredictionStrategy:
    """Unit tests for the TorchPredictionStrategy class.

    This test suite validates prediction behavior for classification and
    regression tasks, input validation, and internal prediction utilities.
    """

    @pytest.fixture
    def strategy(self):
        """Creates a TorchPredictionStrategy instance for testing.

        Returns:
            TorchPredictionStrategy: Strategy instance used in tests.
        """
        return TorchPredictionStrategy()

    # --- requires_dataloader ---

    def test_requires_dataloader(self, strategy):
        """Ensures the strategy declares that a DataLoader is required."""
        assert strategy.requires_dataloader() is True

    # --- predict: validations ---

    def test_predict_raises_if_model_is_none(self, strategy):
        """Ensures predict raises an error when the model is None."""
        with pytest.raises(AssertionError):
            strategy.predict(None, TaskTypeEnum.CLASSIFICATION, loader=make_loader())

    def test_predict_raises_if_no_loader(self, strategy):
        """Ensures predict raises an error when no DataLoader is provided."""
        with pytest.raises(ValueError, match="DataLoader must be provided"):
            strategy.predict(simple_model(), TaskTypeEnum.CLASSIFICATION)

    def test_predict_raises_for_unknown_task(self, strategy):
        """Ensures predict raises an error for unsupported task types."""
        with pytest.raises(ValueError, match="Unknown task type"):
            strategy.predict(simple_model(), task=None, loader=make_loader())

    # --- predict: classification ---

    def test_predict_binary_classification(self, strategy):
        """Ensures binary classification predictions return valid labels."""
        model = simple_model(output_size=1)
        preds = strategy.predict(
            model, TaskTypeEnum.CLASSIFICATION, loader=make_loader()
        )
        assert preds.shape == (8,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_binary_classification_custom_threshold(self, strategy):
        """Ensures custom threshold is applied in binary classification."""

        class ConstantModel(nn.Module):
            def forward(self, x):
                return torch.ones((x.shape[0], 1))

        model = ConstantModel()

        preds = strategy.predict(
            model,
            TaskTypeEnum.CLASSIFICATION,
            loader=make_loader(),
            threshold=0.7,
        )

        assert np.all(preds == 1)

    def test_predict_multiclass_classification(self, strategy):
        """Ensures multiclass classification predictions return valid class indices."""
        model = simple_model(input_size=4, output_size=3)
        preds = strategy.predict(
            model, TaskTypeEnum.CLASSIFICATION, loader=make_loader()
        )
        assert preds.shape == (8,)
        assert np.all((preds >= 0) & (preds < 3))

    # --- predict: regression ---

    def test_predict_regression_shape(self, strategy):
        """Ensures regression predictions return a 1D numeric array."""
        model = simple_model(output_size=1)
        preds = strategy.predict(model, TaskTypeEnum.REGRESSION, loader=make_loader())
        assert preds.shape == (8,)
        assert preds.dtype in (np.float32, np.float64)

    # --- _predict_classification ---

    def test_predict_classification_raises_for_1d_output(self, strategy):
        """Ensures classification prediction requires 2D model outputs."""
        with pytest.raises(ValueError, match="2D"):
            strategy._predict_classification(torch.rand(8), threshold=0.5)

    def test_predict_classification_binary_output(self, strategy):
        """Ensures binary classification outputs are correctly thresholded."""
        outputs = torch.zeros(4, 1)  # sigmoid(0) = 0.5, threshold=0.5 → 0
        preds = strategy._predict_classification(outputs, threshold=0.5)
        assert preds.shape == (4,)

    def test_predict_classification_multiclass_output(self, strategy):
        """Ensures multiclass predictions select the class with highest score."""
        outputs = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])
        preds = strategy._predict_classification(outputs, threshold=0.5)
        assert preds.tolist() == [1, 0]

    # --- _predict_regression ---

    def test_predict_regression_2d_single_output(self, strategy):
        """Ensures regression outputs with shape (N,1) are flattened."""
        outputs = torch.rand(4, 1)
        preds = strategy._predict_regression(outputs)
        assert preds.shape == (4,)

    def test_predict_regression_1d_output(self, strategy):
        """Ensures regression outputs with shape (N,) are returned unchanged."""
        outputs = torch.rand(4)
        preds = strategy._predict_regression(outputs)
        assert preds.shape == (4,)
