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

    input_size: int | None = Field(
        default=None, gt=0, description="Input size (auto-detected if None)."
    )
    output_size: int = Field(..., gt=0, description="Output size (number of classes).")

    @property
    def is_input_size_dynamic(self) -> bool:
        return self.input_size is None

    def set_inferred_input_size(self, input_size: int) -> None:
        if input_size <= 0:
            raise ValueError("Inferred input_size must be > 0")
        self.input_size = input_size

    _target: type = PrivateAttr(default_factory=lambda: TorchModels)


class TorchModels(BaseModels, nn.Module):
    """Base class for PyTorch models."""

    def save(self, filename: str | Path) -> Path:
        """Save model to disk."""
        path = Path(filename) # ensure path
        if path.suffix and path.suffix not in {".pt", ".pth"}:
            raise ValueError(
                "Sklearn models must be saved with .pkl or .pickle extension"
            )
        elif not path.suffix:
            path = path.with_suffix(".pkl")
            logger.warning(
                "No file extension provided. Saving sklearn model with .pkl extension: %s", path
            )
        torch.save(self, path)
        return path


    @classmethod
    def load(cls, filename: str | Path, device="cpu") -> "TorchModels":
        """Load model from disk."""
        path = Path(filename)
        obj = torch.load(path, map_location=device)
        if not isinstance(obj, cls):
            raise ValueError(f"Loaded object is not a MLP instance: {type(obj)}")
        return obj

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        pass

    @abstractmethod
    def get_params(self) -> ParamsT:
        """Return model parameters."""
        pass

