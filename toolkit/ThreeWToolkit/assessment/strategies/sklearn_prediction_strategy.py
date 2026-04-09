import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator

from ...core.base_prediction_strategies import PredictionStrategy
from ...core.enums import TaskTypeEnum


class SklearnPredictionStrategy(PredictionStrategy):
    """Prediction strategy for scikit-learn models.

    Supports both classification and regression tasks using the standard
    scikit-learn `predict` API.
    """

    def predict(
        self, model: BaseEstimator, task: TaskTypeEnum | None = None, **kwargs
    ) -> np.ndarray:
        """Generate predictions using a scikit-learn model.

        Args:
            model (BaseEstimator): Trained scikit-learn model implementing `predict`.
            task (TaskTypeEnum | None): Task type indicating how predictions
                should be interpreted. Defaults to TaskTypeEnum.CLASSIFICATION.
            **kwargs: Additional keyword arguments:
                x (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Array containing model predictions.

        Raises:
            ValueError: If input data is missing or the model is invalid.
        """
        x = kwargs.get("x")

        if x is None:
            raise ValueError("Input data must be provided via 'x'.")

        if not hasattr(model.model_class, "predict"):
            raise ValueError("Model must implement a 'predict' method.")

        x = self._ensure_dataframe(x, kwargs)
        y_pred = model.model_class.predict(x)

        # Explicit casting to numpy array for consistency
        return np.asarray(y_pred)

    def predict_proba(self, model: BaseEstimator, **kwargs) -> np.ndarray:
        """Generate class probability estimates using a scikit-learn model.

        This method is only applicable to classification models that implement
        the `predict_proba` interface. The output follows the standard
        scikit-learn convention.

        Args:
            model (BaseEstimator): Trained scikit-learn classification model implementing
                `predict_proba`.
            **kwargs: Additional keyword arguments:
                x (np.ndarray): Input feature matrix of shape (N, D).

        Returns:
            np.ndarray: Array of class probabilities with shape:
                - (N, 2) for binary classification
                - (N, C) for multiclass classification

        Raises:
            ValueError: If input data is not provided via `x`.
            NotImplementedError: If the model does not support `predict_proba`.
        """
        x = kwargs.get("x")

        if x is None:
            raise ValueError("Input data must be provided via 'x'.")

        if not hasattr(model.model_class, "predict_proba"):
            raise NotImplementedError(
                "This scikit-learn model does not support predict_proba."
            )

        x = self._ensure_dataframe(x, kwargs)
        return np.asarray(model.model_class.predict_proba(x))

    def _ensure_dataframe(
        self, x: pd.DataFrame | pd.Series | np.ndarray, kwargs: dict
    ) -> pd.DataFrame:
        """
        Ensures input is in the correct format (DataFrame if possible).
        Preserves DataFrame structure to avoid feature name warnings.

        Args:
            x: Input data (can be DataFrame, Series, or array-like)

        Returns:
            The input data, converted to DataFrame if it was originally a DataFrame
        """

        if isinstance(x, pd.DataFrame):
            return x
        elif isinstance(x, pd.Series):
            return x.to_frame() if x.ndim == 1 else x
        elif isinstance(x, np.ndarray):
            feature_names = kwargs.get("feature_names")
            # If we stored feature names during fit, recreate DataFrame
            if feature_names is not None:
                return pd.DataFrame(x, columns=feature_names)
            return x
        return x
