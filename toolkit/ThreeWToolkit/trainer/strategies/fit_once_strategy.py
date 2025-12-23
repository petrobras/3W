from typing import Any

from ...core.base_training_strategies import TrainingStrategy


class FitOnceStrategy(TrainingStrategy):
    """Training strategy for scikit-learn models.

    Implements simple fit-based training for sklearn models.
    """

    def train(
        self,
        model: Any,
        X: Any,
        y: Any,
        **kwargs,
    ) -> None:
        """Train sklearn model.

        Args:
            model: Sklearn model to train.
            X: Training features.
            y: Training labels.
            **kwargs: Additional arguments passed to fit().

        Returns:
            None (sklearn models don't return history).
        """
        return model.fit(X, y, **kwargs)
