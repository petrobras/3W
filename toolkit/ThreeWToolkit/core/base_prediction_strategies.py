import numpy as np

from abc import ABC, abstractmethod
from typing import Any
from .enums import TaskType


class PredictionStrategy(ABC):
    """Abstract prediction strategy.

    Defines the interface for different prediction approaches,
    allowing multiple backends or inference pipelines.
    """

    @abstractmethod
    def predict(
        self,
        model: Any,
        task: TaskType | None = None,
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

    def requires_dataloader(self) -> bool:
        """Check if the prediction strategy requires a DataLoader.

        Returns:
            bool: True if a DataLoader is required, False otherwise.
        """
        return False
