from abc import abstractmethod
import logging
from pathlib import Path
from pydantic import Field, PrivateAttr

import torch
from torch import nn


from ..core.base_models import BaseModels, ParamsT, ModelsConfig

logger = logging.getLogger(__name__)


class TorchModelsConfig(ModelsConfig):
    """Base configuration for PyTorch models. Use with TorchTrainer for training."""

    output_size: int = Field(..., gt=0, description="Output size (number of classes).")

    input_size: int | None = Field(
        default=None,
        gt=0,
        description="Input size (auto-detected if None), may be ignored if using LazyModules.",
    )

    @property
    def is_input_size_dynamic(self) -> bool:
        """Whether the input size is dynamic (i.e. not specified in the config). If True, the trainer will infer the
        input size from the training data.
        Returns:
            bool: True if input size is dynamic, False otherwise.
        """
        return self.input_size is None

    def set_inferred_input_size(self, input_size: int) -> None:
        """Set the inferred input size after it has been determined from the training data. This is used when
        input_size is not specified in the config and needs to be inferred from the training data.
        Args:
            input_size (int): The inferred input size.
        """
        if input_size <= 0:
            raise ValueError("Inferred input_size must be > 0")
        self.input_size = input_size

    _target: type = PrivateAttr(default_factory=lambda: TorchModels)


class TorchModels(BaseModels, nn.Module):
    """Base class for PyTorch models."""

    def save(self, filename: str | Path) -> Path:
        """Save model to disk.
        Args:
            filename: Path to save the model. Must have .pt or .pth extension.\
                    If no extension is provided, .pt will be used by default.
        Returns:
            Path to the saved model.
        """
        path = Path(filename)  # ensure path
        if path.suffix and path.suffix not in {".pt", ".pth"}:
            raise ValueError("Torch models must be saved with .pt or .pth extension")
        elif not path.suffix:
            path = path.with_suffix(".pt")
            logger.warning(
                "No file extension provided. Saving torch model with .pt extension: %s",
                path,
            )
        torch.save(self, path)
        return path

    @classmethod
    def load(cls, filename: str | Path, device="cpu") -> "TorchModels":
        """Load model from disk.
        Args:
            filename: Path to the saved model. Must have .pt or .pth extension.
            device: Device to load the model on (e.g. "cpu" or "cuda"). Default is "cpu".
        Returns:
            Loaded TorchModels instance.
        """
        path = Path(filename)
        obj = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(obj, cls):
            raise ValueError(f"Loaded object is not a TorchModel instance: {type(obj)}")
        return obj

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        Args:
            x: Input tensor of shape (batch_size, input_size).
        Returns:
            Output tensor of shape (batch_size, output_size).
        """
        pass

    @abstractmethod
    def get_params(self) -> ParamsT:
        """Return model parameters.

        Returns:
            Iterable of model parameters (Tensors or parameter tuples).
        """
        pass
