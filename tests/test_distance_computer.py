import pytest
import numpy as np
from numpy.testing import assert_allclose
from unittest.mock import patch

from ThreeWToolkit.core.base_clustering import DistanceMatrixConfig
from ThreeWToolkit.core.enums import DistanceMetricEnum
from ThreeWToolkit.clustering import DistanceComputer


class TestDistanceComputerEuclidean:
    """Test suite for DistanceComputer with Euclidean metric."""

    @pytest.fixture
    def euclidean_config(self):
        return DistanceMatrixConfig(
            metric=DistanceMetricEnum.EUCLIDEAN,
            return_condensed=False,
        )

    @pytest.fixture
    def equal_length_series(self):
        return [
            np.array([0.0, 0.0]),
            np.array([3.0, 4.0]),
            np.array([6.0, 8.0]),
        ]

    def test_returns_square_matrix(self, euclidean_config, equal_length_series):
        computer = DistanceComputer(euclidean_config)
        result = computer.fit_transform(equal_length_series)

        assert result.shape == (3, 3)

    def test_diagonal_is_zero(self, euclidean_config, equal_length_series):
        computer = DistanceComputer(euclidean_config)
        result = computer.fit_transform(equal_length_series)

        assert_allclose(np.diag(result), 0.0, atol=1e-10)

    def test_matrix_is_symmetric(self, euclidean_config, equal_length_series):
        computer = DistanceComputer(euclidean_config)
        result = computer.fit_transform(equal_length_series)

        assert_allclose(result, result.T, atol=1e-10)

    def test_known_euclidean_distance(self, euclidean_config):
        computer = DistanceComputer(euclidean_config)
        X = [np.array([0.0, 0.0]), np.array([3.0, 4.0])]
        result = computer.fit_transform(X)

        assert_allclose(result[0, 1], 5.0, atol=1e-10)

    def test_condensed_output(self):
        config = DistanceMatrixConfig(
            metric=DistanceMetricEnum.EUCLIDEAN,
            return_condensed=True,
        )
        computer = DistanceComputer(config)
        X = [np.array([0.0, 0.0]), np.array([3.0, 4.0]), np.array([6.0, 8.0])]
        result = computer.fit_transform(X)

        # Condensed form for 3 items has 3 entries: C(3,2)
        assert result.ndim == 1
        assert len(result) == 3


class TestDistanceComputerDTW:
    """Test suite for DistanceComputer with DTW metric (mocked)."""

    @pytest.fixture
    def dtw_config(self):
        return DistanceMatrixConfig(
            metric=DistanceMetricEnum.DTW,
            return_condensed=False,
            n_jobs=1,
        )

    @patch("ThreeWToolkit.clustering._distances.dtw")
    def test_dtw_calls_distance_matrix_fast(self, mock_dtw, dtw_config):
        mock_dtw.distance_matrix_fast.return_value = np.array([1.0, 2.0, 3.0])

        computer = DistanceComputer(dtw_config)
        X = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
        computer.fit_transform(X)

        mock_dtw.distance_matrix_fast.assert_called_once()

    @patch("ThreeWToolkit.clustering._distances.dtw")
    def test_dtw_parallelism_disabled_for_single_job(self, mock_dtw, dtw_config):
        mock_dtw.distance_matrix_fast.return_value = np.array([1.0])

        computer = DistanceComputer(dtw_config)
        X = [np.array([1.0]), np.array([2.0])]
        computer.fit_transform(X)

        call_kwargs = mock_dtw.distance_matrix_fast.call_args
        assert call_kwargs[1]["parallel"] is False

    @patch("ThreeWToolkit.clustering._distances.dtw")
    def test_dtw_parallelism_enabled_for_multi_job(self, mock_dtw):
        config = DistanceMatrixConfig(
            metric=DistanceMetricEnum.DTW,
            return_condensed=True,
            n_jobs=-1,
        )
        mock_dtw.distance_matrix_fast.return_value = np.array([1.0])

        computer = DistanceComputer(config)
        X = [np.array([1.0]), np.array([2.0])]
        computer.fit_transform(X)

        call_kwargs = mock_dtw.distance_matrix_fast.call_args
        assert call_kwargs[1]["parallel"] is True

    def test_fit_returns_self(self, dtw_config):
        computer = DistanceComputer(dtw_config)
        X = [np.array([1.0, 2.0])]
        assert computer.fit(X) is computer
