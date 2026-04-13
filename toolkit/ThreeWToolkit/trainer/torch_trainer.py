import logging
from typing import Literal, Any

from pydantic import Field, PrivateAttr, field_validator, ValidationInfo
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from ..core.base_dataset import BaseDataset
from ..core.base_trainer import (
    BaseTrainer,
    BaseTrainerConfig,
    PredictionResult,
    TrainingHistory,
)
from ..models.torch_models import TorchModelsConfig, TorchModels

logger = logging.getLogger(__name__)


class TorchTrainerConfig(BaseTrainerConfig):
    """Configuration for PyTorch trainer."""

    config_model: TorchModelsConfig = Field(
        ..., description="Torch model configuration"
    )

    batch_size: int = Field(default=32, gt=0, description="Batch size")
    epochs: int = Field(default=50, gt=0, description="Training epochs")

    optimizer: type[optim.Optimizer] = Field(
        default=optim.Adam, description="Optimizer"
    )
    learning_rate: float = Field(default=1e-3, gt=0, description="Learning rate")
    optimizer_args: dict[str, Any] = Field(
        default_factory=dict, description="Additional optimizer hyperparameters"
    )

    criterion: type[nn.Module] = Field(
        default=nn.CrossEntropyLoss, description="Loss function"
    )

    device: Literal["cuda", "cpu"] = Field(
        default="cuda" if torch.cuda.is_available() else "cpu", description="Device"
    )

    deterministic: bool = Field(
        default=False,
        description="Use deterministic algorithms for cuda (may impact performance)",
    )
    _target: type["TorchTrainer"] = PrivateAttr(default_factory=lambda: TorchTrainer)

    @field_validator("optimizer_args")
    @classmethod
    def check_optimizer_args(
        cls, optimizer_args: dict[str, Any], info: ValidationInfo
    ) -> dict[str, Any]:
        """Validate that optimizer_args are compatible with the chosen optimizer."""
        optimizer_class = info.data["optimizer"]
        try:
            optimizer_class(info.data["learning_rate"], **optimizer_args)
        except Exception as e:
            raise ValueError(
                f"Invalid optimizer_args for {optimizer_class.__name__}: {e}"
            )
        return optimizer_args


class TorchTrainer(BaseTrainer):
    """PyTorch trainer for neural network models."""

    model: TorchModels

    def __init__(self, config: TorchTrainerConfig):
        """Initialize trainer with config and set random seeds.
        Args:
            config: TorchTrainerConfig instance with training parameters
        """
        super().__init__(config)
        self.config: TorchTrainerConfig = config
        self._class_weights: torch.Tensor | None = None

        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = self.config.deterministic
        torch.backends.cudnn.benchmark = self.config.deterministic

        logger.info(
            "TorchTrainer initialized | device=%s | epochs=%d | batch_size=%d",
            config.device,
            config.epochs,
            config.batch_size,
        )

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        return self.config.optimizer(
            self.model.parameters(),
            lr=self.config.learning_rate,  # type: ignore
            **self.config.optimizer_args,
        )

    def _create_criterion(self, weights: torch.Tensor | None = None) -> nn.Module:
        """Create loss function, optionally with class weights.
        Args:
            weights: Optional tensor of class weights for imbalanced classification
        Returns:
            Instantiated loss function
        """
        # Only CrossEntropyLoss supports class weights
        if weights is not None and self.config.criterion is nn.CrossEntropyLoss:
            return self.config.criterion(weight=weights)
        return self.config.criterion()

    def _set_random_seeds(self, seed: int) -> None:
        """Override to set random seeds for reproducibility across PyTorch and CUDA.
        Also calls the base implementation to set seeds for other libraries if needed.
        Args:
            seed: Integer seed value to set for all random number generators
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return super()._set_random_seeds(seed)

    def _prepare_data(self, dataset: BaseDataset, shuffle: bool = True) -> DataLoader:
        """Convert dataset to PyTorch DataLoader.
        This method currently concatenates all signals into a single tensor.
        Args:
            dataset: BaseDataset instance containing signals and labels
            shuffle: Whether to shuffle the data (default: True)
        Returns:
            DataLoader instance for the dataset
        """
        logger.info("Converting dataset to DataLoader (size=%d)", len(dataset))

        signals_list = []
        labels_list = []

        for idx in range(len(dataset)):
            event = dataset[idx]
            signal = event.signal.values

            if signal.ndim == 2:
                signal_tensor = torch.tensor(signal, dtype=torch.float32)
            else:
                signal_tensor = torch.tensor(signal.reshape(1, -1), dtype=torch.float32)
            signals_list.append(signal_tensor)

            if event.label is not None:
                label = event.label.values
                label_tensor = torch.tensor(label, dtype=torch.long).flatten()
                labels_list.append(label_tensor)

        X = torch.cat(signals_list, dim=0)

        # Flatten if 3D (after windowing)
        if X.ndim == 3:
            batch_size, time_steps, features = X.shape
            X = X.reshape(batch_size, time_steps * features)

        y = (
            torch.cat(labels_list)
            if labels_list
            else torch.zeros(X.shape[0], dtype=torch.long)
        )

        tensor_dataset = TensorDataset(X, y)
        dataloader = DataLoader(
            tensor_dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            pin_memory=(self.config.device == "cuda"),
        )

        logger.info("Created DataLoader | batches=%d", len(dataloader))
        return dataloader

    def _execute_training(
        self, train_data: DataLoader, val_data: DataLoader | None
    ) -> TrainingHistory:
        """Execute epoch-based training loop.
        Args:
            train_data: DataLoader for training data
            val_data: Optional DataLoader for validation data
        Returns:
            TrainingHistory object containing training and validation loss history
        """

        train_loss: list[float] = []
        val_loss: list[float] | None = [] if val_data else None

        pbar = tqdm(
            range(self.config.epochs),
            desc="Training",
            unit="epoch",
            colour="#00b4d8",
            dynamic_ncols=True,
        )

        for epoch in pbar:
            epoch_train_loss = self._train_epoch(train_data)
            train_loss.append(epoch_train_loss)

            if val_data is not None and val_loss is not None:
                epoch_val_loss = self._validate_epoch(val_data)
                val_loss.append(epoch_val_loss)
                pbar.set_postfix(
                    loss=f"{epoch_train_loss:.4f}", val_loss=f"{epoch_val_loss:.4f}"
                )
            else:
                pbar.set_postfix(loss=f"{epoch_train_loss:.4f}")

            pbar.refresh()

        return TrainingHistory(train_loss=train_loss, val_loss=val_loss)

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Run single training epoch."""
        self.model.train()
        running_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.config.device, non_blocking=True)
            y_batch = y_batch.to(self.config.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(x_batch)
            loss = self._compute_loss(outputs, y_batch)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(train_loader)

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Compute validation loss.
        Args:
            val_loader: DataLoader for validation data
        Returns:
            Average validation loss for the epoch
        """
        running_loss = 0.0

        with torch.inference_mode():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(self.config.device)
                y_batch = y_batch.to(self.config.device)

                outputs = self.model(x_batch)
                loss = self._compute_loss(outputs, y_batch)
                running_loss += loss.item()

        return running_loss / len(val_loader)

    def predict(self, dataset: BaseDataset) -> PredictionResult:
        """Predict labels for a dataset.
        Args:
            dataset: BaseDataset instance to predict on
        Returns:
            PredictionResult containing predicted and true labels
        """
        dataloader = self._prepare_data(dataset, shuffle=False)
        self.model.eval()

        predictions = []
        true_labels = []

        with torch.inference_mode():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.config.device)
                outputs = self.model(x_batch)

                if outputs.shape[1] > 1:  # multi-class classification
                    outputs = torch.argmax(outputs, dim=1)

                predictions.append(outputs.cpu())
                true_labels.append(y_batch.cpu())

        _true_labels = torch.cat(true_labels, dim=0).numpy()
        _predictions = torch.cat(predictions, dim=0).numpy()

        return PredictionResult(y_pred=_predictions, y_true=_true_labels)

    def _infer_input_size(self, train_data: DataLoader) -> int:
        """Infer the input size from the training data."""
        if len(train_data) > 0:
            return int(train_data.dataset.tensors[0].shape[1])  # type: ignore
        raise ValueError("Training data is empty, cannot infer input size.")

    def _initialize_training_state(
        self, train_data, train_dataset: BaseDataset
    ) -> None:
        """Build model and intialize optimizer and criterion with class weights.
        This method is called after data preparation, so the input size can be inferred

        Args:
            train_data: DataLoader for training data (used to infer input size if needed)
            train_dataset: Original BaseDataset for training (used to compute class weights if enabled)
        """

        # Auto-detect input size if not set
        if self.config.config_model.is_input_size_dynamic:
            input_size = self._infer_input_size(train_data)
            self.config.config_model.input_size = input_size
            logger.info("Initializing model with input_size=%d", input_size)

        # Instantiate model after input size is known
        self.model = self.config.config_model.build().to(self.config.device)  # type: ignore
        self.optimizer = self._create_optimizer()

        # Compute class weights if enabled
        if self.config.use_class_weights:
            class_weight_dict = self._compute_class_weights(train_dataset)
            num_classes = max(class_weight_dict.keys()) + 1
            weights = torch.zeros(num_classes, dtype=torch.float32)
            for cls, weight in class_weight_dict.items():
                weights[cls] = weight
            self._class_weights = weights.to(self.config.device)
            logger.info("Using class weights: %s", class_weight_dict)

        # Create criterion with weights
        self.criterion = self._create_criterion(self._class_weights)

    def _compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss based on task type.
        Args:
            outputs: Model outputs (logits)
            targets: True labels
        Returns:
            Computed loss tensor
        """
        if outputs.ndim == 1:
            outputs = outputs.unsqueeze(1)

        if outputs.shape[1] > 1:
            targets = targets.long()
            return self.criterion(outputs, targets)

        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
            targets = targets.float()
            return self.criterion(outputs.squeeze(1), targets)

        targets = targets.float()
        return self.criterion(outputs.squeeze(1), targets)
