"""TorchTrainer for training PyTorch models with datasets."""

import logging

import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import Field, field_validator, PrivateAttr
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from ..core.base_trainer import BaseTrainer, BaseTrainerConfig, TrainingHistory
from ..core.base_dataset import BaseDataset
from ..core.base_models import BaseTorchModels
from ..core.enums import OptimizersEnum, CriterionEnum
from ..models.mlp import MLPConfig

logger = logging.getLogger(__name__)


class TorchTrainerConfig(BaseTrainerConfig):
    """Configuration for PyTorch trainer."""

    config_model: MLPConfig = Field(..., description="MLP model configuration")
    batch_size: int = Field(default=32, gt=0, description="Batch size")
    epochs: int = Field(default=50, gt=0, description="Training epochs")
    learning_rate: float = Field(default=1e-3, gt=0, description="Learning rate")
    optimizer: str = Field(default="adam", description="Optimizer")
    criterion: str = Field(default="cross_entropy", description="Loss function")
    device: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu", description="Device"
    )
    shuffle: bool = Field(default=False, description="Shuffle training data")
    deterministic: bool = Field(
        default=False,
        description="Use deterministic algorithms for cuda (may impact performance)",
    )

    _target: type["TorchTrainer"] = PrivateAttr(default_factory=lambda: TorchTrainer)

    @field_validator("optimizer")
    @classmethod
    def check_optimizer(cls, value: str) -> str:
        valid = {o.value for o in OptimizersEnum}
        if value not in valid:
            raise ValueError(f"optimizer must be one of {valid}, got '{value}'")
        return value

    @field_validator("criterion")
    @classmethod
    def check_criterion(cls, value: str) -> str:
        valid = {c.value for c in CriterionEnum}
        if value not in valid:
            raise ValueError(f"criterion must be one of {valid}, got '{value}'")
        return value

    @field_validator("device")
    @classmethod
    def check_device(cls, value: str) -> str:
        if value not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if value == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but not available")
        return value


class TorchTrainer(BaseTrainer):
    """PyTorch trainer for neural network models."""

    model: BaseTorchModels

    def __init__(self, config: TorchTrainerConfig):
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

    def _create_optimizer(self) -> torch.optim.Optimizer:
        optimizer_map = {
            OptimizersEnum.ADAM.value: optim.Adam,
            OptimizersEnum.SGD.value: optim.SGD,
            OptimizersEnum.RMSPROP.value: optim.RMSprop,
            OptimizersEnum.ADAMW.value: optim.AdamW,
        }

        optimizer_class = optimizer_map[self.config.optimizer.lower()]
        return optimizer_class(self.model.parameters(), lr=self.config.learning_rate)

    def _create_criterion(self, weights: torch.Tensor | None = None) -> nn.Module:
        """Create loss function, optionally with class weights."""
        criterion_map = {
            CriterionEnum.CROSS_ENTROPY.value: nn.CrossEntropyLoss,
            CriterionEnum.MSE.value: nn.MSELoss,
            CriterionEnum.MAE.value: nn.L1Loss,
        }

        criterion_class = criterion_map[self.config.criterion.lower()]

        # Only CrossEntropyLoss supports class weights
        if weights is not None and criterion_class == nn.CrossEntropyLoss:
            return criterion_class(weight=weights)
        return criterion_class()

    def _set_random_seeds(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return super()._set_random_seeds(seed)

    def _prepare_data_for_training(self, dataset: BaseDataset) -> DataLoader:
        """Convert dataset to PyTorch DataLoader."""
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

        # Auto-detect input size if not set
        if self.config.config_model.input_size is None:
            inferred_size = X.shape[1]
            logger.info("Auto-detected input_size=%d", inferred_size)
            self.config.config_model.input_size = inferred_size

        # Instantiate model after input size is known
        self.model: BaseTorchModels = self.config.config_model.build().to(self.config.device) # type: ignore
        self.optimizer = self._create_optimizer()

        y = (
            torch.cat(labels_list)
            if labels_list
            else torch.zeros(X.shape[0], dtype=torch.long)
        )

        # Compute class weights if enabled
        if self.config.use_class_weights and labels_list:
            class_weight_dict = self._compute_class_weights(dataset)
            num_classes = max(class_weight_dict.keys()) + 1
            weights = torch.zeros(num_classes, dtype=torch.float32)
            for cls, weight in class_weight_dict.items():
                weights[cls] = weight
            self._class_weights = weights.to(self.config.device)
            logger.info("Using class weights: %s", class_weight_dict)

        # Create criterion with weights
        self.criterion = self._create_criterion(self._class_weights)

        tensor_dataset = TensorDataset(X, y)
        dataloader = DataLoader(
            tensor_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            pin_memory=(self.config.device == "cuda"),
        )

        logger.info("Created DataLoader | batches=%d", len(dataloader))
        return dataloader

    def _execute_training(
        self, train_data: DataLoader, val_data: DataLoader | None
    ) -> TrainingHistory:
        """Execute epoch-based training loop."""

        train_loss: list[float] = []
        val_loss: list[float] | None =  [] if val_data else None

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
        """Compute validation loss."""
        running_loss = 0.0

        with torch.inference_mode():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(self.config.device)
                y_batch = y_batch.to(self.config.device)

                outputs = self.model(x_batch)
                loss = self._compute_loss(outputs, y_batch)
                running_loss += loss.item()

        return running_loss / len(val_loader)

    def _compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss based on task type."""
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
