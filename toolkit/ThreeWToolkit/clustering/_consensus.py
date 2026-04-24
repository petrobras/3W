import numpy as np
from sklearn.base import BaseEstimator

from ..core.base_clustering import MultivariateConsensusConfig


class MultivariateConsensus(BaseEstimator):
    """Intersects per-variable cluster memberships across thresholds.

    Finds instances that are consistently considered "normal" or "in-cluster"
    across multiple sensor variables simultaneously. This sweeps through a
    range of normalized distance thresholds and acts as a strict veto system:
    an instance only survives at threshold T if it belongs to the main cluster
    for every single variable at that threshold T.
    """

    def __init__(self, config: MultivariateConsensusConfig):
        self.config = config

        self.common_counts_: dict[float, int] = {}
        self.selection_mask_: np.ndarray = np.array([])
        self.thresholds_analyzed_: list[float] = []
        self._is_fitted: bool = False

    def fit(
        self,
        models: dict[str, BaseEstimator],
        y: np.ndarray | None = None,
        valid_indices_map: dict[str, list[int]] | None = None,
    ) -> "MultivariateConsensus":
        """Fit the consensus model across all trained univariate models.

        Args:
            models (dict[str, BaseEstimator]): Dictionary mapping variable names
                to fitted clustering models (e.g., HierarchicalClusterer).
            y (np.ndarray | None): Ignored.
            valid_indices_map (dict[str, list[int]] | None): Mapping of variable
                names to the lists of original instance indices that survived
                pre-processing (e.g., from InstanceQualityFilter).

        Returns:
            MultivariateConsensus: The fitted consensus instance.
        """
        self.thresholds_analyzed_ = self._generate_threshold_range()

        num_samples = self._determine_total_sample_count(models, valid_indices_map)
        num_thresholds = len(self.thresholds_analyzed_)

        self.selection_mask_ = np.zeros((num_thresholds, num_samples), dtype=int)
        self.common_counts_ = {}

        for row_idx, threshold in enumerate(self.thresholds_analyzed_):
            survivors_at_threshold = self._compute_intersection_at_threshold(
                models, threshold, valid_indices_map
            )

            self._update_state(row_idx, threshold, survivors_at_threshold)

        self._is_fitted = True
        return self

    def get_selected_indices_at_threshold(self, threshold: float) -> list[int]:
        """Get the global indices of instances surviving consensus at given threshold.

        Args:
            threshold (float): Target threshold (0.0 to 1.0).

        Returns:
            list[int]: List of global integer indices.
        """
        self._validate_is_fitted()

        threshold_idx = self._find_nearest_threshold_index(threshold)

        binary_row = self.selection_mask_[threshold_idx, :]
        selected_indices = np.where(binary_row == 1)[0]

        return selected_indices.tolist()

    def _generate_threshold_range(self) -> list[float]:
        num_steps = (
            int(
                (self.config.max_threshold - self.config.min_threshold)
                / self.config.threshold_step
            )
            + 1
        )
        return np.linspace(
            self.config.min_threshold, self.config.max_threshold, num_steps
        ).tolist()

    def _determine_total_sample_count(
        self,
        models: dict[str, BaseEstimator],
        valid_indices_map: dict[str, list[int]] | None,
    ) -> int:
        if valid_indices_map:
            return self._get_max_index_from_map(valid_indices_map)

        return self._infer_sample_count_from_model(next(iter(models.values())))

    def _get_max_index_from_map(self, valid_indices_map: dict[str, list[int]]) -> int:
        all_indices = [idx for indices in valid_indices_map.values() for idx in indices]
        if not all_indices:
            return 0
        return max(all_indices) + 1

    def _infer_sample_count_from_model(self, model: BaseEstimator) -> int:
        if hasattr(model, "ranking_"):
            return len(model.ranking_)

        if hasattr(model, "linkage_matrix_"):
            return len(model.linkage_matrix_) + 1

        raise ValueError(
            "Could not infer sample count from models. Unknown model type."
        )

    def _compute_intersection_at_threshold(
        self,
        models: dict[str, BaseEstimator],
        threshold: float,
        valid_indices_map: dict[str, list[int]] | None,
    ) -> list[int]:
        current_intersection: set[int] | None = None

        for var_name, model in models.items():
            local_survivors = self._extract_survivors(model, threshold)
            global_survivors = self._map_to_global_indices(
                local_survivors, var_name, valid_indices_map
            )

            if current_intersection is None:
                current_intersection = set(global_survivors)
            else:
                current_intersection = current_intersection.intersection(
                    set(global_survivors)
                )

            if current_intersection is not None and len(current_intersection) == 0:
                break

        return list(current_intersection) if current_intersection else []

    def _map_to_global_indices(
        self,
        local_indices: list[int],
        var_name: str,
        valid_indices_map: dict[str, list[int]] | None,
    ) -> list[int]:
        if not valid_indices_map or var_name not in valid_indices_map:
            return local_indices

        mapping = valid_indices_map[var_name]
        return [mapping[i] for i in local_indices if i < len(mapping)]

    def _extract_survivors(self, model: BaseEstimator, threshold: float) -> list[int]:
        # Hierarchical Clusterer
        if hasattr(model, "find_main_cluster_indices"):
            return model.find_main_cluster_indices(threshold)

        # Divisive Ranker
        if hasattr(model, "ranking_") and hasattr(model, "elimination_distances_"):
            return self._filter_divisive_survivors(model, threshold)

        raise TypeError(f"Unsupported model type: {type(model)}")

    def _filter_divisive_survivors(self, model, threshold: float) -> list[int]:
        rankings = np.array(model.ranking_)
        distances = np.array(model.elimination_distances_)

        # Normalize distances to 0-1 range for threshold comparison
        max_dist = np.max(distances)
        if max_dist > 0:
            normalized_distances = distances / max_dist
        else:
            normalized_distances = distances

        valid_mask = normalized_distances <= threshold
        return rankings[valid_mask].tolist()

    def _update_state(
        self, row_idx: int, threshold: float, survivors: list[int]
    ) -> None:
        self.common_counts_[threshold] = len(survivors)
        if survivors:
            self.selection_mask_[row_idx, survivors] = 1

    def _find_nearest_threshold_index(self, threshold: float) -> int:
        array_thresholds = np.array(self.thresholds_analyzed_)
        idx = int((np.abs(array_thresholds - threshold)).argmin())
        return idx

    def _validate_is_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Intersection model is not fitted. Run fit() first.")
