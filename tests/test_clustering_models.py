import pytest
import numpy as np
from pydantic import ValidationError

from ThreeWToolkit.core.base_clustering import (
    HierarchicalClusteringConfig,
    DivisiveClusteringConfig,
    MultivariateConsensusConfig,
)
from ThreeWToolkit.core.enums import LinkageMethodEnum
from ThreeWToolkit.clustering import (
    HierarchicalClusterer,
    DivisiveRanker,
    MultivariateConsensus,
)


class TestHierarchicalClusterer:
    """Test suite for HierarchicalClusterer estimator."""

    @pytest.fixture
    def distance_matrix(self):
        """Two tight pairs: (0,1) and (2,3)."""
        return np.array(
            [
                [0.0, 1.0, 5.0, 6.0],
                [1.0, 0.0, 4.0, 5.0],
                [5.0, 4.0, 0.0, 2.0],
                [6.0, 5.0, 2.0, 0.0],
            ]
        )

    @pytest.fixture
    def config(self):
        return HierarchicalClusteringConfig(
            linkage_method=LinkageMethodEnum.AVERAGE,
            default_threshold=0.5,
        )

    @pytest.fixture
    def fitted_clusterer(self, config, distance_matrix):
        clusterer = HierarchicalClusterer(config)
        clusterer.fit(distance_matrix)
        return clusterer

    def test_fit_returns_self(self, config, distance_matrix):
        clusterer = HierarchicalClusterer(config)
        result = clusterer.fit(distance_matrix)
        assert result is clusterer

    def test_linkage_matrix_stored(self, fitted_clusterer):
        assert fitted_clusterer.linkage_matrix_ is not None
        assert fitted_clusterer.linkage_matrix_.ndim == 2
        assert fitted_clusterer.linkage_matrix_.shape[1] == 4

    def test_distance_matrix_normalized(self, fitted_clusterer):
        assert fitted_clusterer.distance_matrix_normalized_ is not None
        assert fitted_clusterer.distance_matrix_normalized_.max() <= 1.0

    def test_clusters_at_low_threshold(self, fitted_clusterer):
        """At a very low threshold, each instance is its own cluster."""
        labels = fitted_clusterer.get_clusters_at_threshold(0.01)
        assert len(labels) == 4
        assert len(set(labels)) == 4

    def test_clusters_at_high_threshold(self, fitted_clusterer):
        """At a high threshold, all instances merge into one cluster."""
        labels = fitted_clusterer.get_clusters_at_threshold(1.0)
        assert len(labels) == 4
        assert len(set(labels)) == 1

    def test_find_main_cluster_indices(self, fitted_clusterer):
        indices = fitted_clusterer.find_main_cluster_indices(1.0)
        assert sorted(indices) == [0, 1, 2, 3]

    def test_raises_before_fit(self, config):
        clusterer = HierarchicalClusterer(config)
        with pytest.raises(RuntimeError, match="must be fitted"):
            clusterer.get_clusters_at_threshold(0.5)

    def test_zero_distance_matrix(self, config):
        """All-zero distance matrix should not crash."""
        zeros = np.zeros((3, 3))
        clusterer = HierarchicalClusterer(config)
        clusterer.fit(zeros)
        labels = clusterer.get_clusters_at_threshold(0.5)
        assert len(labels) == 3


class TestDivisiveRanker:
    """Test suite for DivisiveRanker estimator."""

    @pytest.fixture
    def distance_matrix(self):
        """Instance 2 is the outlier (highest total distance)."""
        return np.array(
            [
                [0.0, 1.0, 10.0],
                [1.0, 0.0, 9.0],
                [10.0, 9.0, 0.0],
            ]
        )

    @pytest.fixture
    def config(self):
        return DivisiveClusteringConfig()

    @pytest.fixture
    def fitted_ranker(self, config, distance_matrix):
        ranker = DivisiveRanker(config)
        ranker.fit(distance_matrix)
        return ranker

    def test_fit_returns_self(self, config, distance_matrix):
        ranker = DivisiveRanker(config)
        result = ranker.fit(distance_matrix)
        assert result is ranker

    def test_ranking_length_matches_instances(self, fitted_ranker):
        ranking = fitted_ranker.get_ranked_indices()
        assert len(ranking) == 3

    def test_ranking_contains_all_indices(self, fitted_ranker):
        ranking = fitted_ranker.get_ranked_indices()
        assert sorted(ranking) == [0, 1, 2]

    def test_outlier_eliminated_first(self, fitted_ranker):
        """Instance 2 has the largest distance sum and should be eliminated first."""
        ranking = fitted_ranker.get_ranked_indices()
        assert ranking[0] == 2

    def test_elimination_distances_length(self, fitted_ranker):
        assert len(fitted_ranker.elimination_distances_) == 3

    def test_last_elimination_distance_is_zero(self, fitted_ranker):
        assert fitted_ranker.elimination_distances_[-1] == 0.0

    def test_raises_before_fit(self, config):
        ranker = DivisiveRanker(config)
        with pytest.raises(RuntimeError, match="must be fitted"):
            ranker.get_ranked_indices()

    def test_raises_on_non_square_matrix(self, config):
        ranker = DivisiveRanker(config)
        with pytest.raises(ValueError, match="2D square"):
            ranker.fit(np.array([[1.0, 2.0]]))


class TestMultivariateConsensus:
    """Test suite for MultivariateConsensus estimator."""

    @pytest.fixture
    def config(self):
        return MultivariateConsensusConfig(
            min_threshold=0.1,
            max_threshold=1.0,
            threshold_step=0.1,
        )

    @pytest.fixture
    def fitted_hierarchical_models(self):
        """Build two fitted HierarchicalClusterers for two variables."""
        cfg = HierarchicalClusteringConfig(
            linkage_method=LinkageMethodEnum.AVERAGE,
            default_threshold=0.5,
        )
        # Variable A: instances 0,1 are close; instance 2 is far
        dm_a = np.array(
            [
                [0.0, 1.0, 10.0],
                [1.0, 0.0, 9.0],
                [10.0, 9.0, 0.0],
            ]
        )
        model_a = HierarchicalClusterer(cfg)
        model_a.fit(dm_a)

        # Variable B: similar structure
        dm_b = np.array(
            [
                [0.0, 2.0, 8.0],
                [2.0, 0.0, 7.0],
                [8.0, 7.0, 0.0],
            ]
        )
        model_b = HierarchicalClusterer(cfg)
        model_b.fit(dm_b)

        return {"var_A": model_a, "var_B": model_b}

    def test_fit_returns_self(self, config, fitted_hierarchical_models):
        consensus = MultivariateConsensus(config)
        result = consensus.fit(fitted_hierarchical_models)
        assert result is consensus

    def test_selection_mask_shape(self, config, fitted_hierarchical_models):
        consensus = MultivariateConsensus(config)
        consensus.fit(fitted_hierarchical_models)

        n_thresholds = len(consensus.thresholds_analyzed_)
        assert consensus.selection_mask_.shape[0] == n_thresholds
        assert consensus.selection_mask_.shape[1] > 0

    def test_common_counts_populated(self, config, fitted_hierarchical_models):
        consensus = MultivariateConsensus(config)
        consensus.fit(fitted_hierarchical_models)

        assert len(consensus.common_counts_) == len(consensus.thresholds_analyzed_)

    def test_high_threshold_selects_all(self, config, fitted_hierarchical_models):
        consensus = MultivariateConsensus(config)
        consensus.fit(fitted_hierarchical_models)

        selected = consensus.get_selected_indices_at_threshold(1.0)
        assert len(selected) == 3

    def test_low_threshold_selects_fewer(self, config, fitted_hierarchical_models):
        consensus = MultivariateConsensus(config)
        consensus.fit(fitted_hierarchical_models)

        selected_low = consensus.get_selected_indices_at_threshold(0.1)
        selected_high = consensus.get_selected_indices_at_threshold(1.0)
        assert len(selected_low) <= len(selected_high)

    def test_raises_before_fit(self, config):
        consensus = MultivariateConsensus(config)
        with pytest.raises(RuntimeError, match="not fitted"):
            consensus.get_selected_indices_at_threshold(0.5)

    def test_with_valid_indices_map(self, config, fitted_hierarchical_models):
        valid_indices = {
            "var_A": [0, 1, 2],
            "var_B": [0, 1, 2],
        }
        consensus = MultivariateConsensus(config)
        consensus.fit(
            fitted_hierarchical_models,
            valid_indices_map=valid_indices,
        )

        assert consensus._is_fitted


class TestMultivariateConsensusConfig:
    """Test suite for MultivariateConsensusConfig validation."""

    def test_default_values(self):
        config = MultivariateConsensusConfig()
        assert config.min_threshold == 0.1
        assert config.max_threshold == 1.0
        assert config.threshold_step == 0.1

    def test_max_must_be_greater_than_min(self):
        with pytest.raises(ValidationError, match="max_threshold"):
            MultivariateConsensusConfig(
                min_threshold=0.5,
                max_threshold=0.3,
            )

    def test_equal_min_max_rejected(self):
        with pytest.raises(ValidationError, match="max_threshold"):
            MultivariateConsensusConfig(
                min_threshold=0.5,
                max_threshold=0.5,
            )
