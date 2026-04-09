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
    ) -> dict[str, Any]:
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

    @property
    def requires_optimizer(self) -> bool:
        """True if strategy requires optimizer (eg neural networks)."""
        return False

    @property
    def requires_criterion(self) -> bool:
        """True if strategy requires loss function."""
        return False
