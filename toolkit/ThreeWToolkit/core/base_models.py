from dataclasses import dataclass
from typing import Any, Type
import numpy as np

from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import BaseModel, Field, field_validator

from ..core.enums import ModelTypeEnum
from ..core.base_training_strategies import TrainingStrategy


class ModelsConfig(BaseModel):
    model_type: ModelTypeEnum = Field(..., description="Type of model to use.")
    random_seed: int | None = Field(42, description="Random seed for reproducibility.")
    _target: Type | None = Field(default=None, exclude=True)

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
    
    def setup(self, **kwargs) -> Any:
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
        if self._target is None:
            raise ValueError(
                f"{self.__class__.__name__} must set _target attribute. "
                f"Example: _target: Type = MLP"
            )
        return self._target(self, **kwargs)


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
    def predict(self) -> np.ndarray:
        """Generate predictions for the given data.

        Returns:
            Array of predictions.
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
    def get_training_strategy(self) -> TrainingStrategy:
        """Return the appropriate training strategy for this model.

        Returns:
            TrainingStrategy instance.
        """
        pass

@dataclass
class InstantiateConfig:
    """Base config class with instantiation capability.
    
    This follows the Hydra pattern where configs know how to
    instantiate their corresponding objects via the _target attribute.
    
    Attributes:
        _target: The class to instantiate.
    
    Example:
        @dataclass
        class MLPConfig(InstantiateConfig):
            _target: Type = MLP
            hidden_sizes: tuple = (64, 32)
            output_size: int = 10
        
        config = MLPConfig()
        model = config.setup()  # Creates MLP instance
    """
    
    _target: Type
    
    def setup(self, **kwargs) -> Any:
        """Instantiate the object specified in _target.
        
        Args:
            **kwargs: Additional keyword arguments passed to the constructor.
        
        Returns:
            Instantiated object.
        
        Example:
            config = MLPConfig(hidden_sizes=(128, 64))
            model = config.setup(device='cuda')
        """
        return self._target(self, **kwargs)