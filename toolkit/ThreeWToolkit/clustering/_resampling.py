import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ..core.base_clustering import ResamplingConfig


class TimeSeriesResampler(BaseEstimator, TransformerMixin):
    """Downsamples time series data to reduce length.

    This transformer shortens variable-length time series arrays to speed up
    downstream operations like Dynamic Time Warping (DTW), which has O(N^2)
    complexity.
    """

    def __init__(self, config: ResamplingConfig):
        self.config = config

    def fit(
        self, X: list[np.ndarray], y: np.ndarray | None = None
    ) -> "TimeSeriesResampler":
        """Fit the resampler (no-op for stateless resampling)."""
        return self

    def transform(self, X: list[np.ndarray]) -> list[np.ndarray]:
        """Apply the resampling to all instances.

        Args:
            X (list[np.ndarray]): List of time series arrays.

        Returns:
            list[np.ndarray]: List of downsampled time series arrays.
        """
        return [self._resample_series(series) for series in X]

    def _resample_series(self, series: np.ndarray) -> np.ndarray:
        if self.config.step_size <= 1:
            return series

        if self.config.step_method == "slice":
            return series[:: self.config.step_size]

        elif self.config.step_method == "mean":
            length = len(series)
            trunc_len = (length // self.config.step_size) * self.config.step_size
            if trunc_len == 0:
                return np.array([series.mean()]) if length > 0 else np.zeros(1)

            truncated = series[:trunc_len]
            return truncated.reshape(-1, self.config.step_size).mean(axis=1)

        else:
            raise ValueError(f"Unknown step_method: {self.config.step_method}")
