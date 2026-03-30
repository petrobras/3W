from pathlib import Path
from typing import Mapping, TypeVar
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, field_validator
import torch
from torch import nn
from .enums import ModelTypeEnum
from .base_instantiable import Instantiable
from typing import Iterable, TypeAlias

ParamsT: TypeAlias = (
    Iterable[torch.Tensor]
    | Iterable[dict[str, torch.Tensor | int | float | str | bool]]
    | Iterable[tuple[str, torch.Tensor]]
)


class ModelsConfig(BaseModel, Instantiable):
    """Base configuration class for all models."""

    model_type: ModelTypeEnum = Field(..., description="Type of model to use.")
    random_seed: int | None = Field(
        default=42, description="Random seed for reproducibility."
    )
    target_: type["BaseModels"]

    @field_validator("model_type")
    @classmethod
    def check_model_type(cls: type["ModelsConfig"], value: ModelTypeEnum):
        """Validate that model_type is supported."""
        allowed = {
            ModelTypeEnum.MLP,
            ModelTypeEnum.LOGISTIC_REGRESSION,
            ModelTypeEnum.RANDOM_FOREST,
            ModelTypeEnum.DECISION_TREE,
            ModelTypeEnum.GRADIENT_BOOSTING,
            ModelTypeEnum.KNN,
            ModelTypeEnum.NAIVE_BAYES,
            ModelTypeEnum.SVM,
        }

        if value not in allowed:
            raise NotImplementedError(f"model_type {value} not implemented yet.")
        return value


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
    def save(self, filename: str | Path) -> None:
        """Save model to disk.

        Args:
            filename: File path where model should be saved.
        """
        pass

    @abstractmethod
    def load(self, filename: str | Path):
        """Load model from disk.

        Args:
            filename: File path from which to load model.

        Returns:
            Loaded model instance.
        """
        pass


class BaseTorchModels(BaseModels, nn.Module):
    """Base class for PyTorch models."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        pass

    @abstractmethod
    def get_params(self) -> ParamsT:
        """Return model parameters."""
        pass


class BaseSkLearnModels(BaseModels):
    """Base class for scikit-learn models."""

    @abstractmethod
    def get_params(self) -> Mapping[str, object]:
        """Return model parameters."""
        pass
