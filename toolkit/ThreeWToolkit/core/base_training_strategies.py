from typing import Any
from abc import ABC, abstractmethod


class TrainingStrategy(ABC):
    """Abstract training strategy.

    Defines the interface for different training approaches.
    """

    @abstractmethod
    def train(
        self,
        model: Any,
        x_train: Any,
        y_train: Any,
        x_val: Any = None,
        y_val: Any = None,
        **kwargs,
    ) -> Any:
        """Train the model.

        Args:
            model: Model to train.
            x_train: Training features.
            y_train: Training labels.
            x_val: Validation features (optional).
            y_val: Validation labels (optional).
            **kwargs: Additional training parameters.

        Returns:
            Dictionary containing training history.
        """
        pass

    def requires_optimizer(self) -> bool:
        """Check if model requires an optimizer.

        Returns:
            True if model needs optimizer (neural networks), False otherwise (sklearn).
        """
        return False

    def requires_criterion(self) -> bool:
        """Check if model requires a loss function.

        Returns:
            True if model needs criterion (neural networks), False otherwise (sklearn).
        """
        return False
