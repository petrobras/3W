from typing import Type
from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import BaseModel, Field, field_validator

from .enums import ModelTypeEnum
from .base_training_strategies import TrainingStrategy
from .base_prediction_strategies import PredictionStrategy


class ModelsConfig(BaseModel):
    """
    Base configuration class for all models.

    Defines common configuration attributes shared across
    all model implementations and provides a factory method
    to instantiate concrete models.
    """

    model_type: ModelTypeEnum = Field(..., description="Type of model to use.")
    random_seed: int | None = Field(42, description="Random seed for reproducibility.")
    target_: type["BaseModels"]

    @field_validator("model_type")
    @classmethod
    def check_model_type(cls, v, info):
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

        if v not in allowed:
            raise NotImplementedError(f"model_type {v} not implemented yet.")

        return v

    def setup(self, **kwargs) -> "BaseModels":
        """Instantiate the model specified in target_.

        This method acts as a factory that builds the concrete model
        associated with this configuration object.

        Args:
            **kwargs: Additional keyword arguments passed to the model constructor.

        Returns:
            BaseModels: Instantiated model object.

        Raises:
            ValueError: If target_ is not defined.
            RuntimeError: If model instantiation fails.
        """

        if self.target_ is None:
            raise ValueError(
                f"{self.__class__.__name__} must set target_ attribute. "
                f"Example: target_: Type = 'MLP'"
            )

        try:
            return self.target_(self, **kwargs)
        except Exception as e:
            raise RuntimeError("Failed to instantiate model") from e


class BaseModels(ABC):
    """
    Abstract base class for all models.

    Defines the core interface that all models must implement,
    separating model architecture from training logic.
    """

    @property
    def model_name(self) -> str:
        return self.__class__.__name__

    def __init__(self, config: ModelsConfig):
        """
        Initialize model with configuration.

        Args:
            config: Model configuration object.
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x):
        """Forward pass through the model.

        Args:
            x: Input data.

        Returns:
            Model output.
        """
        pass

    @abstractmethod
    def save(self, path: Path):
        """Save model to disk.

        Args:
            path: File path where model should be saved.
        """
        pass

    @abstractmethod
    def load(self, path: Path):
        """Load model from disk.

        Args:
            path: File path from which to load model.

        Returns:
            Loaded model instance.
        """
        pass

    @abstractmethod
    def get_training_strategy(self) -> Type[TrainingStrategy]:
        """Return the training strategy class associated with this model.

        Returns:
            Type[TrainingStrategy]: Training strategy class.
        """
        pass

    @abstractmethod
    def get_prediction_strategy(self) -> Type[PredictionStrategy]:
        """Return the prediction strategy class associated with this model.

        Returns:
            Type[PredictionStrategy]: Prediction strategy class.
        """
        pass
