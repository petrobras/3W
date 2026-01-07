from typing import Type
from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import BaseModel, Field, PrivateAttr, field_validator
from importlib import import_module

from .enums import ModelTypeEnum
from .base_training_strategies import TrainingStrategy
from .base_prediction_strategies import PredictionStrategy


class ModelsConfig(BaseModel):
    model_type: ModelTypeEnum = Field(..., description="Type of model to use.")
    random_seed: int | None = Field(42, description="Random seed for reproducibility.")
    _target: str = PrivateAttr()

    @field_validator("model_type")
    @classmethod
    def check_model_type(cls, v, info):
        if v not in {
            ModelTypeEnum.MLP,
            ModelTypeEnum.LOGISTIC_REGRESSION,
            ModelTypeEnum.RANDOM_FOREST,
            ModelTypeEnum.DECISION_TREE,
            ModelTypeEnum.GRADIENT_BOOSTING,
            ModelTypeEnum.KNN,
            ModelTypeEnum.NAIVE_BAYES,
            ModelTypeEnum.SVM,
        }:
            raise NotImplementedError("model_type not implemented yet.")
        elif v is None:
            raise ValueError("model_type is required.")

        return v

    def setup(self, **kwargs) -> "BaseModels":
        """Instantiate the model specified in _target.

        Args:
            **kwargs: Additional arguments passed to model constructor.

        Returns:
            Instantiated model instance.

        Raises:
            ValueError: If _target is not set.

        Example:
            >>> config = MLPConfig(hidden_sizes=(64, 32), output_size=10)
            >>> model = config.setup(device='cuda')
        """
        _MODELS_NAMESPACE = "ThreeWToolkit.models"

        if self._target is None:
            raise ValueError(
                f"{self.__class__.__name__} must set _target attribute. "
                f"Example: _target: Type = 'MLP'"
            )

        models_pkg = import_module(_MODELS_NAMESPACE)

        try:
            model_cls: Type = getattr(models_pkg, self._target)
        except AttributeError:
            raise ValueError(
                f"Model '{self._target}' not found in {_MODELS_NAMESPACE}.__init__.py"
            )

        return model_cls(self, **kwargs)


class BaseModels(ABC):
    """
    Abstract base class for all models.

    Defines the core interface that all models must implement,
    separating model architecture from training logic.
    """

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
