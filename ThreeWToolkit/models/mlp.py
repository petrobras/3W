import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from ..core.base_models import ModelsConfig, BaseModels
from ..core.enums import (
    ModelTypeEnum,
    ActivationFunctionEnum,
)
from typing import Iterable, Any, TypeAlias, Union, Callable
from tqdm import tqdm
from pydantic import Field, field_validator

ParamsT: TypeAlias = Union[
    Iterable[torch.Tensor], Iterable[dict[str, Any]], Iterable[tuple[str, torch.Tensor]]
]


class MLPConfig(ModelsConfig):
    """
    Configuration for the Multi-Layer Perceptron (MLP) model.

    Description:
        Defines the hyperparameters and architecture for an MLP model, including input/output sizes, hidden layers, activation function, and regularization.

    Args:
        model_type (ModelTypeEnum): The type of model (default: MLP).
        input_size (int): Number of input features (must be > 0).
        hidden_sizes (tuple[int, ...]): Sizes of hidden layers (at least one, all > 0).
        output_size (int): Number of output features (must be > 0).
        activation_function (str): Activation function to use ("relu", "sigmoid", or "tanh").
        regularization (float | None): Regularization parameter (>= 0 or None).
        random_seed (int | None): Random seed for reproducibility.

    Example:
        >>> from ThreeWToolkit.models.mlp import MLPConfig, ActivationFunctionEnum
        >>> config = MLPConfig(
        ...     input_size=10,
        ...     hidden_sizes=(64, 32),
        ...     output_size=1,
        ...     activation_function=ActivationFunctionEnum.RELU.value,
        ...     regularization=None,
        ...     random_seed=42
        ... )
    """

    model_type: ModelTypeEnum = Field(
        default=ModelTypeEnum.MLP, description="Type of model (MLP)."
    )
    input_size: int = Field(
        ..., gt=0, description="Number of input features (must be > 0)."
    )
    hidden_sizes: tuple[int, ...] = Field(
        ..., min_length=1, description="Sizes of hidden layers (at least one)."
    )
    output_size: int = Field(
        ..., gt=0, description="Number of output features (must be > 0)."
    )
    activation_function: str = Field(
        default="relu", description="Activation function to use."
    )
    regularization: float | None = Field(
        default=None, ge=0, description="Regularization parameter (>=0 or None)."
    )

    @field_validator("activation_function")
    @classmethod
    def check_activation_function(cls, v):
        valid = {
            ActivationFunctionEnum.RELU.value,
            ActivationFunctionEnum.SIGMOID.value,
            ActivationFunctionEnum.TANH.value,
        }
        if v not in valid:
            raise ValueError(f"activation_function must be one of {valid}")
        return v

    @field_validator("hidden_sizes")
    @classmethod
    def check_hidden_sizes(cls, v):
        if not v or not all(isinstance(h, int) and h > 0 for h in v):
            raise ValueError("hidden_sizes must be a tuple of positive integers")
        return v


class LabeledSubset(torch.utils.data.Dataset):
    """
    Custom dataset for labeled data, compatible with PyTorch DataLoader.

    Args:
        samples (np.ndarray | torch.Tensor): Input samples.
        labels (np.ndarray | torch.Tensor): Corresponding labels.

    Example:
        >>> dataset = LabeledSubset(np.random.rand(100, 10), np.random.rand(100, 1))
        >>> loader = DataLoader(dataset, batch_size=32)
    """

    def __init__(
        self,
        samples: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
    ):
        """
        Initialize the LabeledSubset dataset.

        Args:
            samples (np.ndarray | torch.Tensor): Input samples.
            labels (np.ndarray | torch.Tensor): Corresponding labels.
        """
        if len(samples) != len(labels):
            raise ValueError("Samples and labels must have the same length.")

        self.samples = samples
        self.labels = labels

    def __getitem__(self, idx):
        """
        Get a sample and its label by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (sample, label)
        """
        x_i = self.samples[idx]
        y_i = self.labels[idx]
        return x_i, y_i

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.samples)


class MLP(BaseModels, nn.Module):
    """
    Multi-Layer Perceptron (MLP) model for regression or classification tasks.

    Args:
        config (MLPConfig): Configuration for the MLP model.

    Example:
        >>> config = MLPConfig(input_size=10, hidden_sizes=(64, 32), output_size=1,
        ...                    activation_function=ActivationFunctionEnum.RELU,
        ...                    regularization=None)
        >>> model = MLP(config)
    """

    def __init__(self, config: MLPConfig) -> None:
        nn.Module.__init__(self)
        BaseModels.__init__(self, config)

        layers: list[nn.Module] = []
        self.activation_func = self._get_activation_function(config.activation_function)

        in_size = config.input_size
        for h in config.hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(self.activation_func)
            in_size = h

        layers.append(nn.Linear(in_size, config.output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

    def _get_activation_function(self, activation: str) -> nn.Module:
        """Return activation function module from enum value."""
        if activation == ActivationFunctionEnum.RELU.value:
            return nn.ReLU()
        elif activation == ActivationFunctionEnum.SIGMOID.value:
            return nn.Sigmoid()
        elif activation == ActivationFunctionEnum.TANH.value:
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def get_params(self) -> ParamsT:
        """Return model parameters for optimization."""
        return self.model.parameters()

    def _run_evaluation_epoch(
        self,
        loader: DataLoader,
        criterion: Callable,
        device: str,
        metrics: list[Callable] | None,
    ) -> tuple[float, dict]:
        """Run one evaluation epoch on validation/test set."""
        self.model.eval()
        running_loss = 0.0
        all_preds: list[Any] = []
        all_labels: list[Any] = []

        with torch.no_grad():
            for x_values, y_values in loader:
                x_values, y_values = x_values.to(device), y_values.to(device)
                out = self.model(x_values)

                # Task detection
                if out.shape[1] > 1:  # multiclass
                    preds = torch.argmax(out, dim=1)
                    y_values = y_values.long()
                    loss = criterion(out, y_values)
                else:  # binary or regression
                    if isinstance(criterion, nn.BCEWithLogitsLoss):
                        preds = (torch.sigmoid(out) > 0.5).long().squeeze(1)
                        y_values = y_values.float()
                        loss = criterion(out.squeeze(1), y_values)
                    else:  # regression
                        preds = out.squeeze(1)
                        y_values = y_values.float()
                        loss = criterion(out.squeeze(1), y_values)

                running_loss += loss.item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_values.cpu().numpy())

        avg_loss = running_loss / len(loader)

        if metrics is None or len(metrics) == 0:
            from sklearn.metrics import explained_variance_score

            metrics = [explained_variance_score]

        result_metrics = self.evaluate(all_preds, all_labels, metrics)
        return avg_loss, result_metrics

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: Callable,
        optimizer: torch.optim.Optimizer,
        device: str,
    ) -> float:
        """Run one training epoch."""
        model.train()
        epoch_train_loss = 0.0

        for x_values, y_values in train_loader:
            x_values, y_values = x_values.to(device), y_values.to(device)

            optimizer.zero_grad()
            outputs = model(x_values)

            # Task detection
            if outputs.shape[1] > 1:  # multiclass
                y_values = y_values.long()
                loss = criterion(outputs, y_values)
            else:  # binary or regression
                if isinstance(criterion, nn.BCEWithLogitsLoss):
                    y_values = y_values.float()
                    loss = criterion(outputs.squeeze(1), y_values)
                else:
                    y_values = y_values.float()
                    loss = criterion(outputs.squeeze(1), y_values)

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        return epoch_train_loss / len(train_loader)

    def fit(
        self,
        train_loader: DataLoader,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        val_loader: DataLoader | None = None,
        metrics: list[Callable] | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> dict[str, list[Any]]:
        """
        Train the MLP model.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader | None): DataLoader for validation data.
            epochs (int): Number of training epochs.
            optimizer (torch.optim.Optimizer): Optimizer.
            criterion (Callable): Loss function.

        Returns:
            dict[str, list[Any]]: Dictionary containing training/validation losses and metrics.
        """
        self.model.to(device)
        best_model: dict[str, Any] = {
            "epoch": -1,
            "model": None,
            "val_loss": float("inf"),
        }
        loss_dict: dict[str, list[Any]] = {
            "train_loss": [],
            "val_loss": [],
            "metrics": [],
        }

        with tqdm(
            range(epochs), desc="Training", unit="epoch", leave=False
        ) as progress_bar:
            for epoch_idx in progress_bar:
                progress_bar.set_description(f"Epoch {epoch_idx + 1}/{epochs}")

                avg_epoch_train_loss = self._train_epoch(
                    model=self.model,
                    train_loader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                )
                loss_dict["train_loss"].append(avg_epoch_train_loss)

                if val_loader is not None:
                    avg_val_loss, result_metrics = self._run_evaluation_epoch(
                        loader=val_loader,
                        criterion=criterion,
                        device=device,
                        metrics=metrics,
                    )
                    loss_dict["metrics"].append(result_metrics)
                    loss_dict["val_loss"].append(avg_val_loss)
                else:
                    avg_val_loss = float("nan")
                    loss_dict["val_loss"].append(avg_val_loss)

                progress_bar.set_postfix(
                    {
                        "train_loss": f"{avg_epoch_train_loss:.4f}",
                        "val_loss": f"{avg_val_loss:.4f}",
                    }
                )

                # Save best model
                if (
                    best_model["val_loss"] is not None
                    and isinstance(avg_val_loss, float)
                    and avg_val_loss < float(best_model["val_loss"])
                ):
                    best_model["epoch"] = epoch_idx
                    best_model["model"] = self.model.state_dict()
                    best_model["val_loss"] = avg_val_loss

        return loss_dict

    def evaluate(
        self,
        x: list | torch.Tensor,
        y: list | torch.Tensor,
        metrics: list[Callable],
    ) -> dict:
        """Compute metrics for predictions and targets."""
        return {metric.__name__: metric(y, x) for metric in metrics}

    def test(
        self,
        test_loader: DataLoader,
        criterion: Callable,
        metrics: list[Callable],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> tuple[float, dict]:
        """Evaluate the model on a test set."""
        return self._run_evaluation_epoch(
            test_loader, criterion=criterion, device=device, metrics=metrics
        )

    def predict(
        self,
        loader: DataLoader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> np.ndarray:
        """Return predictions for given data loader."""
        self.model.eval()
        y_pred: list[Any] = []

        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(device).float()
                outputs = self.model.forward(X_batch)

                if outputs.shape[1] > 1:  # multiclass
                    _, preds = torch.max(outputs, 1)
                elif outputs.shape[1] == 1:  # binary
                    preds = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)
                else:  # regression
                    preds = outputs.squeeze(1)

                y_pred.extend(preds.cpu().numpy())

        return np.array(y_pred)
