import pytest
import numpy as np
from numpy.testing import assert_allclose

from ThreeWToolkit.core.base_clustering import ResamplingConfig, TimeSeriesScalingConfig
from ThreeWToolkit.clustering import TimeSeriesResampler, TimeSeriesScaler


class TestTimeSeriesResampler:
    """Test suite for TimeSeriesResampler transformer."""

    @pytest.fixture
    def slice_config(self):
        return ResamplingConfig(step_size=5, step_method="slice")

    @pytest.fixture
    def mean_config(self):
        return ResamplingConfig(step_size=5, step_method="mean")

    def test_fit_returns_self(self, slice_config):
        resampler = TimeSeriesResampler(slice_config)
        X = [np.arange(100, dtype=float)]
        assert resampler.fit(X) is resampler

    def test_slice_method_picks_every_nth(self, slice_config):
        resampler = TimeSeriesResampler(slice_config)
        series = np.arange(20, dtype=float)
        result = resampler.fit_transform([series])

        assert len(result) == 1
        assert_allclose(result[0], np.array([0, 5, 10, 15], dtype=float))

    def test_mean_method_averages_blocks(self, mean_config):
        resampler = TimeSeriesResampler(mean_config)
        series = np.arange(10, dtype=float)  # [0..9], 2 blocks of 5
        result = resampler.fit_transform([series])

        assert len(result) == 1
        expected = np.array([2.0, 7.0])  # mean(0..4)=2, mean(5..9)=7
        assert_allclose(result[0], expected)

    def test_mean_truncates_remainder(self, mean_config):
        resampler = TimeSeriesResampler(mean_config)
        series = np.arange(13, dtype=float)  # 2 full blocks (10), remainder 3 dropped
        result = resampler.fit_transform([series])

        assert len(result[0]) == 2

    def test_step_size_one_returns_original(self):
        config = ResamplingConfig(step_size=1, step_method="slice")
        resampler = TimeSeriesResampler(config)
        series = np.array([1.0, 2.0, 3.0])
        result = resampler.fit_transform([series])
        assert_allclose(result[0], series)

    def test_mean_short_series_returns_mean(self):
        config = ResamplingConfig(step_size=100, step_method="mean")
        resampler = TimeSeriesResampler(config)
        series = np.array([2.0, 4.0, 6.0])
        result = resampler.fit_transform([series])

        assert len(result[0]) == 1
        assert_allclose(result[0][0], 4.0)

    def test_handles_multiple_instances(self, slice_config):
        resampler = TimeSeriesResampler(slice_config)
        X = [np.arange(20, dtype=float), np.arange(15, dtype=float)]
        result = resampler.fit_transform(X)
        assert len(result) == 2


class TestTimeSeriesScaler:
    """Test suite for TimeSeriesScaler (Z-normalization)."""

    @pytest.fixture
    def default_config(self):
        return TimeSeriesScalingConfig(with_mean=True, with_std=True)

    def test_fit_returns_self(self, default_config):
        scaler = TimeSeriesScaler(default_config)
        X = [np.array([1.0, 2.0, 3.0])]
        assert scaler.fit(X) is scaler

    def test_z_normalization(self, default_config):
        scaler = TimeSeriesScaler(default_config)
        series = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = scaler.fit_transform([series])

        assert_allclose(np.mean(result[0]), 0.0, atol=1e-10)
        assert_allclose(np.std(result[0]), 1.0, atol=1e-10)

    def test_mean_only(self):
        config = TimeSeriesScalingConfig(with_mean=True, with_std=False)
        scaler = TimeSeriesScaler(config)
        series = np.array([10.0, 20.0, 30.0])
        result = scaler.fit_transform([series])

        assert_allclose(np.mean(result[0]), 0.0, atol=1e-10)
        assert np.std(result[0]) != 1.0  # std not normalized

    def test_std_only(self):
        config = TimeSeriesScalingConfig(with_mean=False, with_std=True)
        scaler = TimeSeriesScaler(config)
        series = np.array([10.0, 20.0, 30.0])
        result = scaler.fit_transform([series])

        assert np.mean(result[0]) != 0.0  # mean not centered
        assert_allclose(np.std(result[0]), 1.0, atol=1e-10)

    def test_no_scaling(self):
        config = TimeSeriesScalingConfig(with_mean=False, with_std=False)
        scaler = TimeSeriesScaler(config)
        series = np.array([10.0, 20.0, 30.0])
        result = scaler.fit_transform([series])
        assert_allclose(result[0], series)

    def test_constant_series_std_zero(self, default_config):
        """Constant series should not divide by zero."""
        scaler = TimeSeriesScaler(default_config)
        series = np.ones(10) * 5.0
        result = scaler.fit_transform([series])
        # After centering, all zeros; std=0 so no division
        assert_allclose(result[0], np.zeros(10), atol=1e-10)

    def test_empty_series_passthrough(self, default_config):
        scaler = TimeSeriesScaler(default_config)
        series = np.array([])
        result = scaler.fit_transform([series])
        assert result[0].size == 0

    def test_handles_multiple_instances(self, default_config):
        scaler = TimeSeriesScaler(default_config)
        X = [np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0, 40.0])]
        result = scaler.fit_transform(X)

        assert len(result) == 2
        for scaled in result:
            assert_allclose(np.mean(scaled), 0.0, atol=1e-10)
