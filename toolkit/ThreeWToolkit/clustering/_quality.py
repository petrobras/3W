import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..core.base_clustering import InstanceQualityConfig


class InstanceQualityFilter(BaseEstimator, TransformerMixin):
    """Filters out low-quality instances from a collection of time series.

    This transformer analyzes each time series instance for unrecoverable
    flaws (high ratio of NaN gaps, long frozen sensor periods). It discards
    bad instances, repairs slightly corrupted ones (via NaN interpolation),
    and keeps track of which original instances survived the filter.
    """

    def __init__(self, config: InstanceQualityConfig):
        self.config = config
        self.kept_indices_: list[int] = []

    def fit(
        self, X: list[np.ndarray], y: np.ndarray | None = None
    ) -> "InstanceQualityFilter":
        """Fit the filter (no-op for stateless filtering)."""
        return self

    def transform(self, X: list[np.ndarray]) -> list[np.ndarray]:
        """Filter and repair instances.

        Args:
            X (list[np.ndarray]): A list of variable-length time series arrays.

        Returns:
            list[np.ndarray]: A list of cleaned and repaired time series arrays.
                Instances Failing quality thresholds are dropped.
        """
        self.kept_indices_ = []
        cleaned_dataset = []

        for index, series in enumerate(X):
            if self._is_discardable(series):
                continue

            repaired_series = self._repair_series_if_needed(series)
            cleaned_dataset.append(repaired_series)
            self.kept_indices_.append(index)

        return cleaned_dataset

    def _is_discardable(self, series: np.ndarray) -> bool:
        if series.size == 0:
            return True

        if self._calculate_nan_ratio(series) > self.config.max_nan_ratio:
            return True

        if self._calculate_frozen_ratio(series) > self.config.max_frozen_ratio:
            return True

        return False

    def _repair_series_if_needed(self, series: np.ndarray) -> np.ndarray:
        working_series = series.copy()

        if self._contains_nan(working_series):
            working_series = self._interpolate_nan_values(working_series)

        return working_series

    def _calculate_nan_ratio(self, series: np.ndarray) -> float:
        return np.isnan(series).mean()

    def _calculate_frozen_ratio(self, series: np.ndarray) -> float:
        differences = np.diff(series)
        is_static_step = np.abs(differences) <= self.config.frozen_threshold
        # Mean of boolean array gives ratio of True values
        return is_static_step.mean()

    def _contains_nan(self, series: np.ndarray) -> bool:
        return np.isnan(series).any()

    def _interpolate_nan_values(self, series: np.ndarray) -> np.ndarray:
        series_frame = pd.DataFrame(series)
        interpolated_frame = series_frame.interpolate(
            method="linear", limit_direction="both"
        )
        return interpolated_frame.to_numpy().flatten()
