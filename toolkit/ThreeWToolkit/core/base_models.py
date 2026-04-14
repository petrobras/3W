from typing import Iterable, TypeAlias
from abc import ABC, abstractmethod

from pathlib import Path
from pydantic import BaseModel

import torch

from .base_instantiable import Instantiable

ParamsT: TypeAlias = (
    Iterable[torch.Tensor]
    | Iterable[dict[str, torch.Tensor | int | float | str | bool]]
    | Iterable[tuple[str, torch.Tensor]]
)


class ModelsConfig(BaseModel, Instantiable):
    """Base configuration class for all models."""

    _target: type["BaseModels"]


class BaseModels(ABC):
    """
    Abstract base class for all models.

    Defines the core interface that all models must implement,
    separating model architecture from training logic.
    """

    @property
    def model_name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def save(self, filename: str | Path) -> Path:
        """Save model to disk.

        Args:
            filename: File path where model should be saved.

        Returns:
            Path to the saved model file.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, filename: str | Path) -> "BaseModels":
        """Load model from disk.

        Args:
            filename: File path from which to load model.

        Returns:
            Loaded model instance.
        """
        pass
