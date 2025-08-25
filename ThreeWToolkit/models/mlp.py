import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from ..metrics import explained_variance_score
from ..core.base_models import ModelsConfig, BaseModels
from ..core.enums import (
    ModelTypeEnum,
    ActivationFunctionEnum,
)
from typing import Iterable, Any, TypeAlias, Union, Callable
from tqdm import tqdm

ParamsT: TypeAlias = Union[
    Iterable[torch.Tensor], Iterable[dict[str, Any]], Iterable[tuple[str, torch.Tensor]]
]


class MLPConfig(ModelsConfig):
    """
    Configuration class for the MLP model.

    Args:
        model_type (ModelTypeEnum): The type of model (default: MLP).
        input_size (int): Number of input features.
        hidden_sizes (tuple[int, ...]): Sizes of hidden layers.
        output_size (int): Number of output features.
        activation_function (str): "relu", "sigmoid", "tanh", Activation function to use, default to "relu".
        regularization (float | None): Regularization parameter.

    Example:
        >>> config = MLPConfig(input_size=10, hidden_sizes=(64, 32), output_size=1, activation_function=ActivationFunctionEnum.RELU, regularization=None)
    """

    model_type: ModelTypeEnum = ModelTypeEnum.MLP
    input_size: int
    hidden_sizes: tuple[int, ...]
    output_size: int
    activation_function: str = "relu"
    regularization: float | None


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
        >>> config = MLPConfig(input_size=10, hidden_sizes=(64, 32), output_size=1, activation_function=ActivationFunctionEnum.RELU, regularization=None)
        >>> model = MLP(config)
    """

    def __init__(self, config: MLPConfig):
        """
        Initialize the MLP model.

        Args:
            config (MLPConfig): Configuration for the MLP model.
        """
        nn.Module.__init__(self)
        BaseModels.__init__(self, config)
        layers = []
        self.activation_func = self._get_activation_function(config.activation_function)
        in_size = config.input_size
        for h in config.hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(self.activation_func)
            in_size = h
        layers.append(nn.Linear(in_size, config.output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor.

        Example:
            >>> output = model(torch.randn(32, 10))
        """
        return self.model(x)

    def _get_activation_function(self, activation: str):
        """
        Get the activation function based on the enum.

        Args:
            activation (str): Activation function enum.

        Returns:
            nn.Module: Activation function module.
        """
        if activation == ActivationFunctionEnum.RELU.value:
            return nn.ReLU()
        elif activation == ActivationFunctionEnum.SIGMOID.value:
            return nn.Sigmoid()
        elif activation == ActivationFunctionEnum.TANH.value:
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def get_params(self) -> ParamsT:
        """
        Get the model's parameters for optimization.

        Returns:
            Iterator over model parameters.

        Example:
            >>> params = model.get_params()
        """
        return self.model.parameters()

    def _run_evaluation_epoch(
        self,
        loader: DataLoader,
        criterion: Callable,
        device: str,
        metrics: list[Callable] | None,
    ) -> tuple[float, dict]:
        """
        Run a full evaluation epoch over a DataLoader.

        Args:
            loader (DataLoader): DataLoader for evaluation.
            criterion (Callable): Loss function.

        Returns:
            tuple: (average loss, metrics dictionary)
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        total_eval_samples = 0
        with torch.no_grad():
            for x_values, y_values in loader:
                x_values, y_values = (
                    x_values.to(device).float(),
                    y_values.to(device).float(),
                )

                out = self.model(x_values)
                loss = criterion(out, y_values)
                running_loss += loss.item() * x_values.size(0)

                preds = out.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_values.cpu().numpy())
                total_eval_samples += x_values.size(0)

        avg_loss = running_loss / total_eval_samples

        if metrics is None or len(metrics) == 0:
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
        model.train()
        epoch_train_loss = 0.0
        total_train_samples = 0
        for x_values, y_values in train_loader:
            x_values, y_values = (
                x_values.to(device).float(),
                y_values.to(device).float(),
            )
            # Forward pass
            optimizer.zero_grad()
            outputs = model(x_values)

            # Compute the loss
            loss = criterion(outputs, y_values)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            # Compute the loss for the batch
            samples_in_batch = x_values.size(0)
            epoch_train_loss += loss.item() * samples_in_batch
            total_train_samples += samples_in_batch

        # Compute the average loss for the epoch
        avg_epoch_train_loss = epoch_train_loss / total_train_samples
        return avg_epoch_train_loss

    def fit(
        self,
        train_loader: DataLoader,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        val_loader: DataLoader | None,
        metrics: list[Callable] | None,
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

        Example:
            >>> model.fit(train_loader, val_loader, epochs=10, optimizer=optim.Adam(), criterion=nn.MSELoss())
        """
        self.model.train()
        best_model = {
            "epoch": -1,
            "model": None,
            "val_loss": float("inf"),
        }
        loss_dict = {"train_loss": [], "val_loss": [], "metrics": []}

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

                # Show metrics in tqdm bar
                progress_bar.set_postfix(
                    {
                        "train_loss": f"{avg_epoch_train_loss:.4f}",
                        "val_loss": f"{avg_val_loss:.4f}",
                    }
                )

                # Update the best model if the current epoch has the best validation loss
                if avg_val_loss < best_model["val_loss"]:
                    best_model["epoch"] = epoch_idx
                    best_model["model"] = self.model.eval()
                    best_model["val_loss"] = avg_val_loss
        return loss_dict

    def evaluate(
        self,
        x: list | torch.Tensor,
        y: list | torch.Tensor,
        metrics: list[Callable],
    ) -> dict:
        """
        Compute metrics for predictions and targets.

        Args:
            x (list | torch.Tensor): Predictions.
            y (list | torch.Tensor): Targets.
            metrics (list[Callable]): List of metric functions.

        Returns:
            dict: Dictionary of metric results.
        """
        return {metric.__name__: metric(y, x) for metric in metrics}

    def test(
        self,
        test_loader: DataLoader,
        criterion: Callable,
        metrics: list[Callable],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> tuple[float, dict]:
        """
        Evaluate the model on a test set.

        Args:
            test_loader (DataLoader): DataLoader for test data.
            criterion (Callable): Loss function.
            metrics (list[Callable]): List of metric functions.

        Returns:
            tuple: (average loss, metrics dictionary)
        """
        return self._run_evaluation_epoch(
            test_loader, criterion=criterion, device=device, metrics=metrics
        )

    def predict(
        self,
        loader: DataLoader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> np.ndarray:
        self.model.eval()
        y_pred = []
        with torch.no_grad():
            for X_batch in loader:
                X_batch = X_batch.to(device).float()
                outputs = self.model.forward(X_batch)
                _, preds = torch.max(outputs, 1)
                y_pred.extend(preds.cpu().numpy())
        return np.array(y_pred)
