from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator

from ..core.enums import ModelTypeEnum


class ModelsConfig(BaseModel):
    model_type: ModelTypeEnum = Field(..., description="Type of model to use.")
    random_seed: Optional[int] = Field(
        42, description="Random seed for reproducibility."
    )

    @field_validator("model_type")
    @classmethod
    def check_model_type(cls, v, info):
        # --- THIS IS THE FIX ---
        # We check the direct value `v` instead of accessing it through `info`.
        if v not in {
            ModelTypeEnum.MLP,
            ModelTypeEnum.LGBM,
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


class BaseModels(ABC):
    def __init__(self, config: ModelsConfig):
        """
        Base model class constructor.

        Args:
            config (ModelsConfig): Configuration object with model parameters.
        """
        self.config = config

    @abstractmethod
    def train(self, X: Any, y: Any = None) -> None:
        """
        Train the model on the given data.

        Args:
            X (Any): Input features.
            y (Any, optional): Target values (if supervised).
        """

        pass

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """
        Make predictions using the trained model.

        Args:
            X (Any): Input features to predict.

        Returns:
            Any: Predicted outputs.
        """
        pass

    @abstractmethod
    def evaluate(self, X: Any, y: Any) -> Dict[str, float]:
        """
        Evaluate the model performance.

        Args:
            X (Any): Input features.
            y (Any): Ground truth target values.

        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics.
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters.

        Returns:
            Dict[str, Any]: Dictionary of model parameters.
        """
        pass

    @abstractmethod
    def set_params(self, **params: Any) -> None:
        """
        Set model parameters.

        Args:
            **params (Any): Key-value pairs of parameters to update.
        """
        pass
