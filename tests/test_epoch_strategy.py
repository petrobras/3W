import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from unittest.mock import MagicMock
from ThreeWToolkit.trainer.strategies.epoch_strategy import EpochTrainingStrategy


def simple_model(input_size=4, output_size=1):
    return nn.Sequential(nn.Linear(input_size, 8), nn.ReLU(), nn.Linear(8, output_size))


def make_tensors(n=20, input_size=4):
    x = torch.rand(n, input_size)
    y = torch.randint(0, 2, (n,))
    return x, y


def make_dataframes(n=20, input_size=4):
    x = pd.DataFrame(np.random.rand(n, input_size).astype(np.float32))
    y = pd.Series(np.random.randint(0, 2, n))
    return x, y


def default_kwargs(model, epochs=2):
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

    @pytest.fixture
    def strategy(self):
        return EpochTrainingStrategy()

    # --- properties ---

    def test_requires_optimizer(self, strategy):
        assert strategy.requires_optimizer is True

    def test_requires_criterion(self, strategy):
        assert strategy.requires_criterion is True

    # --- train: validations ---

    def test_train_raises_if_model_is_none(self, strategy):
        x, y = make_tensors()
        with pytest.raises(AssertionError):
            strategy.train(None, x, y, **default_kwargs(simple_model()))

    def test_train_raises_if_no_optimizer(self, strategy):
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
        model = simple_model()
        x, y = make_tensors()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with pytest.raises(ValueError, match="Criterion"):
            strategy.train(
                model, x, y, epochs=1, optimizer=optimizer, device="cpu", batch_size=8
            )

    # --- train: happy paths ---

    def test_train_returns_history_without_validation(self, strategy):
        model = simple_model()
        x, y = make_tensors()
        history = strategy.train(model, x, y, **default_kwargs(model))

        assert "train_loss" in history
        assert "model" in history
        assert "val_loss" not in history
        assert len(history["train_loss"]) == 2

    def test_train_returns_history_with_validation(self, strategy):
        model = simple_model()
        x_train, y_train = make_tensors(30)
        x_val, y_val = make_tensors(10)
        kwargs = default_kwargs(model)

        history = strategy.train(model, x_train, y_train, x_val, y_val, **kwargs)

        assert "val_loss" in history
        assert len(history["val_loss"]) == len(history["train_loss"])

    def test_train_loss_is_positive_float(self, strategy):
        model = simple_model()
        x, y = make_tensors()
        history = strategy.train(model, x, y, **default_kwargs(model))
        assert all(isinstance(v, float) and v >= 0 for v in history["train_loss"])

    def test_train_accepts_dataframes(self, strategy):
        model = simple_model()
        x, y = make_dataframes()
        history = strategy.train(model, x, y, **default_kwargs(model))
        assert len(history["train_loss"]) == 2

    def test_train_model_in_history_is_nn_module(self, strategy):
        model = simple_model()
        x, y = make_tensors()
        history = strategy.train(model, x, y, **default_kwargs(model))
        assert isinstance(history["model"], nn.Module)

    # --- _compute_loss ---

    def test_compute_loss_single_output(self, strategy):
        outputs = torch.randn(8, 1)
        targets = torch.rand(8)
        loss = strategy._compute_loss(outputs, targets, nn.MSELoss())
        assert loss.item() >= 0

    def test_compute_loss_multiclass(self, strategy):
        outputs = torch.randn(8, 3)
        targets = torch.randint(0, 3, (8,))
        loss = strategy._compute_loss(outputs, targets, nn.CrossEntropyLoss())
        assert loss.item() >= 0

    def test_compute_loss_bce_with_logits(self, strategy):
        outputs = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8,))
        loss = strategy._compute_loss(outputs, targets, nn.BCEWithLogitsLoss())
        assert loss.item() >= 0

    def test_compute_loss_1d_output_gets_unsqueezed(self, strategy):
        outputs = torch.randn(8)  # 1D — deve ser unsqueezed internamente
        targets = torch.rand(8)
        loss = strategy._compute_loss(outputs, targets, nn.MSELoss())
        assert loss.item() >= 0

    # --- _create_dataloader ---

    def test_create_dataloader_from_tensors(self, strategy):
        x, y = make_tensors()
        loader = strategy._create_dataloader(x, y, batch_size=5)
        assert isinstance(loader, DataLoader)
        xb, yb = next(iter(loader))
        assert xb.shape[1] == 4

    def test_create_dataloader_from_dataframes(self, strategy):
        x, y = make_dataframes()
        loader = strategy._create_dataloader(x, y, batch_size=5)
        xb, yb = next(iter(loader))
        assert xb.dtype == torch.float32

    # --- _train_epoch / _calculate_val_loss: model=None guard ---

    def test_train_epoch_raises_if_model_none(self, strategy):
        x, y = make_tensors()
        loader = strategy._create_dataloader(x, y, batch_size=8)
        with pytest.raises(ValueError, match="Model must be initialized"):
            strategy._train_epoch(None, loader, nn.MSELoss(), MagicMock(), "cpu")

    def test_calculate_val_loss_raises_if_model_none(self, strategy):
        x, y = make_tensors()
        loader = strategy._create_dataloader(x, y, batch_size=8)
        with pytest.raises(ValueError, match="Model must be initialized"):
            strategy._calculate_val_loss(None, loader, nn.MSELoss(), "cpu")
