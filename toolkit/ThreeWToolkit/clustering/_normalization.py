import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ..core.base_clustering import TimeSeriesScalingConfig


class TimeSeriesScaler(BaseEstimator, TransformerMixin):
    """Z-normalizes each time series instance independently.

    This standardizes time series data by removing the mean and scaling to unit
    variance. Because this operates per-instance (rather than globally over the
    entire dataset), it ensures that clustering algorithms group instances based
    on shape similarity rather than amplitude magnitude.
    """

    def __init__(self, config: TimeSeriesScalingConfig):
        self.config = config

    def fit(
        self, X: list[np.ndarray], y: np.ndarray | None = None
    ) -> "TimeSeriesScaler":
        """Fit the scaler (no-op since scaling is per-instance)."""
        return self

    def transform(self, X: list[np.ndarray]) -> list[np.ndarray]:
        """Apply independent Z-normalization to each instance.

        Args:
            X (list[np.ndarray]): List of time series arrays.

        Returns:
            list[np.ndarray]: List of scaled time series arrays.
        """
        return [self._scale_series(series) for series in X]

    def _scale_series(self, series: np.ndarray) -> np.ndarray:
        if series.size == 0:
            return series

        scaled_series = series.astype(np.float64, copy=True)

        if self.config.with_mean:
            mean = np.mean(scaled_series)
            scaled_series -= mean

        if self.config.with_std:
            std = np.std(scaled_series)
            if std != 0:
                scaled_series /= std

        return scaled_series
