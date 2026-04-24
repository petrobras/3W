import numpy as np
from sklearn.base import BaseEstimator

from ..core.base_clustering import DivisiveClusteringConfig


class DivisiveRanker(BaseEstimator):
    """Rank timeseries instances from most outlier to most central.

    Uses a divisive top-down approach operating on a distance matrix.
    Recursively eliminates the instance with the highest sum of distances
    to all remaining instances in the active set. The elimination order
    produces a ranking from extreme outliers to the dense core.
    """

    def __init__(self, config: DivisiveClusteringConfig):
        self.config = config
        self.ranking_: list[int] = []
        self.elimination_distances_: list[float] = []
        self._is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "DivisiveRanker":
        """Fit the divisive model to produce instance rankings.

        Args:
            X (np.ndarray): Square pairwise distance matrix.
            y (np.ndarray | None): Ignored.

        Returns:
            DivisiveRanker: The fitted instance.
        """
        if len(X.shape) != 2 or X.shape[0] != X.shape[1]:
            raise ValueError("DivisiveRanker requires a 2D square distance matrix.")

        working_distance_matrix = X.copy()
        number_of_samples = working_distance_matrix.shape[0]
        remaining_original_indices = list(range(number_of_samples))

        self.ranking_ = []
        self.elimination_distances_ = []

        while len(remaining_original_indices) > 1:
            most_distant_local_index, max_distance_sum = (
                self._find_outlier_in_current_set(working_distance_matrix)
            )

            self._record_elimination(
                remaining_original_indices, most_distant_local_index, max_distance_sum
            )

            working_distance_matrix = self._remove_instance_from_matrix(
                working_distance_matrix, most_distant_local_index
            )
            remaining_original_indices.pop(most_distant_local_index)

        self._record_final_survivor(remaining_original_indices)
        self._is_fitted = True
        return self

    def get_ranked_indices(self) -> list[int]:
        """Get the integer indices sorted from worst outlier to tightest centroid.

        Returns:
            list[int]: Ranked list of original instance indices.
        """
        self._validate_is_fitted()
        return self.ranking_

    def _find_outlier_in_current_set(
        self, distance_matrix: np.ndarray
    ) -> tuple[int, float]:
        distance_sums_per_instance = np.sum(distance_matrix, axis=1)
        outlier_index = int(np.argmax(distance_sums_per_instance))
        max_distance = float(distance_sums_per_instance[outlier_index])
        return outlier_index, max_distance

    def _record_elimination(
        self,
        current_indices: list[int],
        local_index_to_remove: int,
        distance_value: float,
    ) -> None:
        original_index = current_indices[local_index_to_remove]
        self.ranking_.append(original_index)
        self.elimination_distances_.append(distance_value)

    def _remove_instance_from_matrix(
        self, matrix: np.ndarray, index_to_remove: int
    ) -> np.ndarray:
        matrix_minus_row = np.delete(matrix, index_to_remove, axis=0)
        matrix_minus_row_and_col = np.delete(matrix_minus_row, index_to_remove, axis=1)
        return matrix_minus_row_and_col

    def _record_final_survivor(self, remaining_indices: list[int]) -> None:
        final_survivor_index = remaining_indices[0]
        self.ranking_.append(final_survivor_index)
        self.elimination_distances_.append(0.0)

    def _validate_is_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("The model must be fitted before retrieving ranking.")
