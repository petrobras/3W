import torch
import torch.nn as nn

from typing import Any, Callable
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from ...core.base_training_strategies import TrainingStrategy


class EpochTrainingStrategy(TrainingStrategy):
    """Epoch-based training strategy for neural network models.

    This strategy implements a complete training loop, including forward and
    backward passes, loss computation, optimization, optional validation,
    and progress tracking.
    """

    def train(
        self,
        model: Any,
        x_train: Any,
        y_train: Any,
        x_val: Any = None,
        y_val: Any = None,
        **kwargs,
    ) -> dict[str, list[Any]]:
        """Train a model using an epoch-based training loop.

        Args:
            model: Neural network model to be trained.
            x_train: Training input features.
            y_train: Training target labels.
            x_val: Validation input features (optional).
            y_val: Validation target labels (optional).
            **kwargs: Additional training parameters:
                epochs (int): Number of training epochs.
                optimizer (torch.optim.Optimizer): Optimizer instance.
                criterion (Callable): Loss function.
                device (str): Device to use ("cpu" or "cuda").
                batch_size (int): Number of samples per batch.
                shuffle (bool): Whether to shuffle training data.

        Returns:
            dict[str, list[Any]]: Dictionary containing the training history:
                - "train_loss": Average training loss per epoch.
                - "val_loss": Average validation loss per epoch (if validation is used).

        Raises:
            AssertionError: If the model is not a valid torch.nn.Module.
            ValueError: If optimizer or criterion is not provided.
        """
        assert model is not None and isinstance(model, nn.Module), (
            "Model must be a valid torch.nn.Module before training."
        )

        epochs = kwargs.get("epochs", 10)
        optimizer = kwargs.get("optimizer")
        criterion = kwargs.get("criterion")
        shuffle = kwargs.get("shuffle", True)
        batch_size = kwargs.get("batch_size", 8)
        device = kwargs.get("device", "cpu")

        if optimizer is None:
            raise ValueError("Optimizer must be provided.")
        if criterion is None:
            raise ValueError("Criterion (loss function) must be provided.")

        model = model.to(device)

        train_loader = self._create_dataloader(
            x_train, y_train, batch_size=batch_size, shuffle=shuffle
        )

        val_loader = None
        if x_val is not None and y_val is not None:
            val_loader = self._create_dataloader(
                x_val, y_val, batch_size=batch_size, shuffle=False
            )

        loss_dict: dict[str, list[Any]] = {"train_loss": []}
        if val_loader is not None:
            loss_dict["val_loss"] = []

        pbar = tqdm(
            range(epochs),
            desc="[Pipeline] Training",
            unit="epoch",
            colour="#00b4d8",
        )

        for _ in pbar:
            avg_train_loss = self._train_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )
            loss_dict["train_loss"].append(avg_train_loss)

            if val_loader is not None:
                val_loss = self._calculate_val_loss(
                    model=model,
                    val_loader=val_loader,
                    criterion=criterion,
                    device=device,
                )
                loss_dict["val_loss"].append(val_loss)

                pbar.set_postfix(
                    train_loss=f"{avg_train_loss:.4f}", val_loss=f"{val_loss:.4f}"
                )
            else:
                pbar.set_postfix(train_loss=f"{avg_train_loss:.4f}")

        return loss_dict

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: Callable,
        optimizer: torch.optim.Optimizer,
        device: str,
    ) -> float:
        """Run a single training epoch.

        Args:
            model: Neural network model.
            train_loader: DataLoader with training data.
            criterion: Loss function.
            optimizer: Optimizer used to update model parameters.
            device: Device used for computation.

        Returns:
            float: Average training loss for the epoch.

        Raises:
            ValueError: If the model is not initialized.
        """
        if model is None:
            raise ValueError("Model must be initialized before training.")

        model.train()
        running_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = self._compute_loss(outputs, y_batch, criterion)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        return running_loss / len(train_loader)

    def _calculate_val_loss(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: Callable,
        device: str,
    ) -> float:
        """Compute the average validation loss.

        Args:
            model: Neural network model.
            val_loader: DataLoader with validation data.
            criterion: Loss function.
            device: Device used for computation.

        Returns:
            float: Average validation loss.

        Raises:
            ValueError: If the model is not initialized.
        """
        if model is None:
            raise ValueError("Model must be initialized before validation.")

        model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(x_batch)
                loss = self._compute_loss(outputs, y_batch, criterion)
                running_loss += loss.item()

        return running_loss / len(val_loader)

    def _compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: Callable,
    ) -> torch.Tensor:
        """Compute the loss according to the task type.

        The task type is inferred automatically based on the output shape
        and the loss function.

        Args:
            outputs: Model predictions.
            targets: Ground-truth labels.
            criterion: Loss function.

        Returns:
            torch.Tensor: Computed loss.
        """
        if outputs.ndim == 1:
            outputs = outputs.unsqueeze(1)

        if outputs.shape[1] > 1:
            targets = targets.long()
            return criterion(outputs, targets)

        if isinstance(criterion, nn.BCEWithLogitsLoss):
            targets = targets.float()
            return criterion(outputs.squeeze(1), targets)

        targets = targets.float()
        return criterion(outputs.squeeze(1), targets)

    def _create_dataloader(
        self,
        X: Any,
        y: Any,
        batch_size: int,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create a DataLoader from input features and labels.

        Args:
            X: Input features (NumPy array or torch.Tensor).
            y: Target labels (NumPy array or torch.Tensor).
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle the dataset.

        Returns:
            DataLoader: PyTorch DataLoader instance.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X.values, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y.values)

        dataset = TensorDataset(X, y)
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True
        )

    def requires_optimizer(self) -> bool:
        """Indicate whether this strategy requires an optimizer.

        Returns:
            bool: Always True for neural network training strategies.
        """
        return True

    def requires_criterion(self) -> bool:
        """Indicate whether this strategy requires a loss function.

        Returns:
            bool: Always True for neural network training strategies.
        """
        return True
