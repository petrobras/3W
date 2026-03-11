import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from unittest.mock import MagicMock
from ThreeWToolkit.trainer.strategies.epoch_strategy import EpochTrainingStrategy


def simple_model(input_size=4, output_size=1):
    """Creates a simple feedforward neural network for testing purposes.

    The model consists of a single hidden layer with ReLU activation and is
    intended for lightweight training tests.

    Args:
        input_size (int, optional): Number of input features. Defaults to 4.
        output_size (int, optional): Number of output units. Defaults to 1.

    Returns:
        torch.nn.Sequential: A simple neural network model.
    """
    return nn.Sequential(nn.Linear(input_size, 8), nn.ReLU(), nn.Linear(8, output_size))


def make_tensors(n=20, input_size=4):
    """Generates random tensor datasets for testing.

    The inputs are sampled from a uniform distribution and the targets are
    binary class labels.

    Args:
        n (int, optional): Number of samples to generate. Defaults to 20.
        input_size (int, optional): Number of features per sample. Defaults to 4.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - x: Input tensor of shape (n, input_size)
            - y: Target tensor of shape (n,)
    """
    x = torch.rand(n, input_size)
    y = torch.randint(0, 2, (n,))
    return x, y


def make_dataframes(n=20, input_size=4):
    """Generates random pandas datasets for testing.

    This utility mirrors ``make_tensors`` but returns pandas structures
    instead of PyTorch tensors, allowing tests to validate dataframe
    compatibility.

    Args:
        n (int, optional): Number of samples to generate. Defaults to 20.
        input_size (int, optional): Number of features per sample. Defaults to 4.

    Returns:
        tuple[pandas.DataFrame, pandas.Series]: A tuple containing:
            - x: Feature matrix as a DataFrame with shape (n, input_size)
            - y: Target labels as a Series with length n
    """
    x = pd.DataFrame(np.random.rand(n, input_size).astype(np.float32))
    y = pd.Series(np.random.randint(0, 2, n))
    return x, y


def default_kwargs(model, epochs=2):
    """Creates a default set of training keyword arguments for tests.

    This helper function builds a minimal configuration dictionary
    containing optimizer, loss criterion, device, and batch size
    required by the training strategy.

    Args:
        model (torch.nn.Module): Model whose parameters will be optimized.
        epochs (int, optional): Number of training epochs. Defaults to 2.

    Returns:
        dict: Dictionary containing default training parameters including:
            - epochs
            - optimizer
            - criterion
            - device
            - batch_size
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    return dict(
        epochs=epochs,
        optimizer=optimizer,
        criterion=criterion,
        device="cpu",
        batch_size=8,
    )


class TestEpochTrainingStrategy:
    """
    Unit tests for the EpochTrainingStrategy training workflow.

    This test suite validates:
    - Required dependencies (optimizer and criterion)
    - Training loop behavior
    - Loss computation logic
    - DataLoader creation
    - Error handling for invalid inputs
    """

    @pytest.fixture
    def strategy(self):
        """Creates a fresh instance of EpochTrainingStrategy for each test."""
        return EpochTrainingStrategy()

    def test_requires_optimizer(self, strategy):
        """Verify that the training strategy declares optimizer as required."""
        assert strategy.requires_optimizer is True

    def test_requires_criterion(self, strategy):
        """Verify that the training strategy declares criterion as required."""
        assert strategy.requires_criterion is True

    def test_train_raises_if_model_is_none(self, strategy):
        """
        Ensure that training fails when the model argument is None.

        The training strategy requires a valid PyTorch model instance.
        """
        x, y = make_tensors()
        with pytest.raises(AssertionError):
            strategy.train(None, x, y, **default_kwargs(simple_model()))

    def test_train_raises_if_no_optimizer(self, strategy):
        """
        Ensure that training raises an error if no optimizer is provided.
        """
        model = simple_model()
        x, y = make_tensors()
        with pytest.raises(ValueError, match="Optimizer must be provided"):
            strategy.train(
                model,
                x,
                y,
                epochs=1,
                criterion=nn.MSELoss(),
                device="cpu",
                batch_size=8,
            )

    def test_train_raises_if_no_criterion(self, strategy):
        """
        Ensure that training raises an error if the loss criterion is missing.
        """
        model = simple_model()
        x, y = make_tensors()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with pytest.raises(ValueError, match="Criterion"):
            strategy.train(
                model, x, y, epochs=1, optimizer=optimizer, device="cpu", batch_size=8
            )

    def test_train_returns_history_without_validation(self, strategy):
        """
        Verify that training returns a history dictionary when no validation set is provided.
        """
        model = simple_model()
        x, y = make_tensors()
        history = strategy.train(model, x, y, **default_kwargs(model))

        assert "train_loss" in history
        assert "model" in history
        assert "val_loss" not in history
        assert len(history["train_loss"]) == 2

    def test_train_returns_history_with_validation(self, strategy):
        """
        Verify that validation loss is included in the history when validation data is provided.
        """
        model = simple_model()
        x_train, y_train = make_tensors(30)
        x_val, y_val = make_tensors(10)
        kwargs = default_kwargs(model)

        history = strategy.train(model, x_train, y_train, x_val, y_val, **kwargs)

        assert "val_loss" in history
        assert len(history["val_loss"]) == len(history["train_loss"])

    def test_train_loss_is_positive_float(self, strategy):
        """
        Ensure that all training losses returned are non-negative floats.
        """
        model = simple_model()
        x, y = make_tensors()
        history = strategy.train(model, x, y, **default_kwargs(model))
        assert all(isinstance(v, float) and v >= 0 for v in history["train_loss"])

    def test_train_accepts_dataframes(self, strategy):
        """
        Verify that the training strategy accepts pandas DataFrame and Series as inputs.
        """
        model = simple_model()
        x, y = make_dataframes()
        history = strategy.train(model, x, y, **default_kwargs(model))
        assert len(history["train_loss"]) == 2

    def test_train_model_in_history_is_nn_module(self, strategy):
        """
        Ensure that the returned history includes the trained PyTorch model instance.
        """
        model = simple_model()
        x, y = make_tensors()
        history = strategy.train(model, x, y, **default_kwargs(model))
        assert isinstance(history["model"], nn.Module)

    def test_compute_loss_single_output(self, strategy):
        """
        Verify loss computation for single-output regression models.
        """
        outputs = torch.randn(8, 1)
        targets = torch.rand(8)
        loss = strategy._compute_loss(outputs, targets, nn.MSELoss())
        assert loss.item() >= 0

    def test_compute_loss_multiclass(self, strategy):
        """
        Verify loss computation for multi-class classification tasks.
        """
        outputs = torch.randn(8, 3)
        targets = torch.randint(0, 3, (8,))
        loss = strategy._compute_loss(outputs, targets, nn.CrossEntropyLoss())
        assert loss.item() >= 0

    def test_compute_loss_bce_with_logits(self, strategy):
        """
        Verify loss computation when using BCEWithLogitsLoss.
        """
        outputs = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8,))
        loss = strategy._compute_loss(outputs, targets, nn.BCEWithLogitsLoss())
        assert loss.item() >= 0

    def test_compute_loss_1d_output_gets_unsqueezed(self, strategy):
        """
        Ensure that 1D outputs are automatically reshaped when required.
        """
        outputs = torch.randn(8)
        targets = torch.rand(8)
        loss = strategy._compute_loss(outputs, targets, nn.MSELoss())
        assert loss.item() >= 0

    def test_create_dataloader_from_tensors(self, strategy):
        """
        Verify that a DataLoader can be created from torch tensors.
        """
        x, y = make_tensors()
        loader = strategy._create_dataloader(x, y, batch_size=5)
        assert isinstance(loader, DataLoader)
        xb, _ = next(iter(loader))
        assert xb.shape[1] == 4

    def test_create_dataloader_from_dataframes(self, strategy):
        """
        Verify that pandas inputs are converted correctly to tensors in the DataLoader.
        """
        x, y = make_dataframes()
        loader = strategy._create_dataloader(x, y, batch_size=5)
        xb, _ = next(iter(loader))
        assert xb.dtype == torch.float32

    def test_train_epoch_raises_if_model_none(self, strategy):
        """
        Ensure that _train_epoch raises an error if the model is not initialized.
        """
        x, y = make_tensors()
        loader = strategy._create_dataloader(x, y, batch_size=8)
        with pytest.raises(ValueError, match="Model must be initialized"):
            strategy._train_epoch(None, loader, nn.MSELoss(), MagicMock(), "cpu")

    def test_calculate_val_loss_raises_if_model_none(self, strategy):
        """
        Ensure that validation loss calculation fails when the model is None.
        """
        x, y = make_tensors()
        loader = strategy._create_dataloader(x, y, batch_size=8)
        with pytest.raises(ValueError, match="Model must be initialized"):
            strategy._calculate_val_loss(None, loader, nn.MSELoss(), "cpu")
