import numpy as np

from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator
from torch import nn

from .enums import TaskTypeEnum


class PredictionStrategy(ABC):
    """Abstract prediction strategy.

    Defines the interface for different prediction approaches,
    allowing multiple backends or inference pipelines.
    """

    @abstractmethod
    def predict(
        self,
        model: nn.Module | BaseEstimator,
        task: TaskTypeEnum | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Generate predictions using a trained model.

        Args:
            model: Trained model used for inference.
            task: Task type associated with the prediction
                (e.g., classification or regression). If None,
                the strategy may infer the task automatically.
            **kwargs: Additional parameters required by the
                prediction strategy (e.g., input data, device,
                batch size).

        Returns:
            np.ndarray: Array containing the model predictions.
        """
        pass

    @property
    def requires_dataloader(self) -> bool:
        """Check if the prediction strategy requires a DataLoader.

        Returns:
            bool: True if a DataLoader is required, False otherwise.
        """
        return False
