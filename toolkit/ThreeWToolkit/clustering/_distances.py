import numpy as np
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances

from dtaidistance import dtw, dtw_barycenter

from ..core.base_clustering import DistanceMatrixConfig
from ..core.enums import DistanceMetricEnum


class DistanceComputer(BaseEstimator, TransformerMixin):
    """Computes pairwise distance matrices for time series clustering.

    Supports Dynamic Time Warping (DTW), DTW Barycenter Average (DBA), and
    Euclidean distance. Required for hierarchical and divisive clustering logic.

    Requires the optional `dtaidistance` package for DTW metrics.
    """

    def __init__(self, config: DistanceMatrixConfig):
        self.config = config

    def fit(
        self, X: list[np.ndarray], y: np.ndarray | None = None
    ) -> "DistanceComputer":
        """Fit the computer (no-op for distance computation)."""
        return self

    def transform(self, X: list[np.ndarray]) -> np.ndarray:
        """Compute the pairwise distance matrix.

        Args:
            X (list[np.ndarray]): List of time series arrays.

        Returns:
            np.ndarray: A 2D square distance matrix, or a 1D condensed
                distance vector if `config.return_condensed` is True.
        """
        if self.config.metric == DistanceMetricEnum.DTW:
            return self._compute_dtw_matrix(X)

        if self.config.metric == DistanceMetricEnum.DTW_BARYCENTER:
            return self._compute_dtw_matrix_via_barycenter(X)

        if self.config.metric == DistanceMetricEnum.EUCLIDEAN:
            return self._compute_euclidean_matrix(X)

        raise ValueError(f"Unsupported metric: {self.config.metric}")

    def _compute_dtw_matrix(self, series_list: list[np.ndarray]) -> np.ndarray:
        series_float_type = [series.astype(np.float64) for series in series_list]

        condensed_matrix = dtw.distance_matrix_fast(
            series_float_type,
            compact=True,
            parallel=self._should_use_parallelism(),
        )

        if self.config.return_condensed:
            return condensed_matrix

        return squareform(condensed_matrix)

    def _compute_dtw_matrix_via_barycenter(
        self, series_list: list[np.ndarray]
    ) -> np.ndarray:
        """Computes the DTW Barycenter Average (DBA)."""
        series_float_type = [series.astype(np.float64) for series in series_list]

        _ = dtw_barycenter.dba_loop(series_float_type, c=None, use_c=True)

        condensed_matrix = dtw.distance_matrix_fast(
            series_float_type,
            compact=True,
            parallel=self._should_use_parallelism(),
        )

        if self.config.return_condensed:
            return condensed_matrix

        return squareform(condensed_matrix)

    def _compute_euclidean_matrix(self, series_list: list[np.ndarray]) -> np.ndarray:
        matrix_form = np.vstack(series_list)

        distance_matrix = euclidean_distances(matrix_form)

        if self.config.return_condensed:
            return squareform(distance_matrix, checks=False)

        return distance_matrix

    def _should_use_parallelism(self) -> bool:
        return self.config.n_jobs != 1
