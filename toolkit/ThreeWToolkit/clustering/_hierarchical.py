import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, ClusterMixin

from ..core.base_clustering import HierarchicalClusteringConfig


class HierarchicalClusterer(BaseEstimator, ClusterMixin):
    """Agglomerative hierarchical clustering for time series.

    Builds a hierarchical linkage tree from a precomputed distance matrix.
    Supports querying the cluster assignments at arbitrary normalized distance
    thresholds to explore cluster stability.
    """

    def __init__(self, config: HierarchicalClusteringConfig):
        self.config = config
        self.linkage_matrix_: np.ndarray | None = None
        self.distance_matrix_normalized_: np.ndarray | None = None
        self._is_fitted: bool = False

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> "HierarchicalClusterer":
        """Fit the hierarchical tree using the distance matrix.

        Args:
            X (np.ndarray): Square pairwise distance matrix or condensed 1D vector.
            y (np.ndarray | None): Ignored.

        Returns:
            HierarchicalClusterer: The fitted instance.
        """
        self.distance_matrix_normalized_ = self._normalize_distance_matrix(X)
        self.linkage_matrix_ = self._compute_linkage_matrix(
            self.distance_matrix_normalized_
        )
        self._is_fitted = True
        return self

    def get_clusters_at_threshold(self, normalized_threshold: float) -> np.ndarray:
        """Get flat cluster assignments using a specific distance threshold.

        Args:
            normalized_threshold (float): Cutoff distance (0.0 to 1.0).

        Returns:
            np.ndarray: Array of cluster integer labels.
        """
        self._validate_is_fitted()
        return fcluster(
            self.linkage_matrix_, t=normalized_threshold, criterion="distance"
        )

    def find_main_cluster_indices(self, normalized_threshold: float) -> list[int]:
        """Find the instances belonging to the cluster containing the first instance.

        Args:
            normalized_threshold (float): Cutoff distance (0.0 to 1.0).

        Returns:
            list[int]: List of integer indices belonging to the main cluster.
        """
        cluster_labels = self.get_clusters_at_threshold(normalized_threshold)
        reference_instance_label = cluster_labels[0]

        main_cluster_indices = np.where(cluster_labels == reference_instance_label)[0]
        return main_cluster_indices.tolist()

    def _normalize_distance_matrix(self, distance_matrix: np.ndarray) -> np.ndarray:
        max_distance = np.max(distance_matrix)
        if max_distance == 0:
            return distance_matrix
        return distance_matrix / max_distance

    def _compute_linkage_matrix(self, distance_matrix: np.ndarray) -> np.ndarray:
        if len(distance_matrix.shape) == 2:
            condensed_distance_array = squareform(distance_matrix, checks=False)
        else:
            condensed_distance_array = distance_matrix

        return linkage(
            condensed_distance_array, method=self.config.linkage_method.value
        )

    def _validate_is_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("The model must be fitted before retrieving clusters.")
