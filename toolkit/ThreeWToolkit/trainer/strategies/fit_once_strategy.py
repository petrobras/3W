from typing import Any

from ...core.base_training_strategies import TrainingStrategy


class FitOnceStrategy(TrainingStrategy):
    """Training strategy for scikit-learn models.

    Implements simple fit-based training for sklearn models.
    """

    def train(
        self,
        model: Any,
        x_train: Any,
        y_train: Any,
        x_val: Any = None,
        y_val: Any = None,
        **kwargs,
    ) -> Any:
        """Train sklearn model.

        Args:
            model: Sklearn model to train.
            x_train: Training features.
            y: Training labels.
            **kwargs: Additional arguments passed to fit().

        Returns:
            None (sklearn models don't return history).
        """
        return model.model_class.fit(x_train, y_train)
