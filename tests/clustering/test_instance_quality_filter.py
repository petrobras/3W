import pytest
import numpy as np
from numpy.testing import assert_allclose

from ThreeWToolkit.core.base_clustering import InstanceQualityConfig
from ThreeWToolkit.clustering import InstanceQualityFilter


class TestInstanceQualityFilter:
    """Test suite for InstanceQualityFilter transformer."""

    @pytest.fixture
    def config(self):
        return InstanceQualityConfig(
            frozen_threshold=0.0,
            max_nan_ratio=0.10,
            max_frozen_ratio=0.10,
        )

    @pytest.fixture
    def quality_filter(self, config):
        return InstanceQualityFilter(config)

    def test_fit_returns_self(self, quality_filter):
        X = [np.array([1.0, 2.0, 3.0])]
        result = quality_filter.fit(X)
        assert result is quality_filter

    def test_keeps_clean_instances(self, quality_filter):
        X = [np.array([1.0, 2.0, 3.0, 4.0, 5.0])]
        result = quality_filter.fit_transform(X)
        assert len(result) == 1
        assert_allclose(result[0], X[0])

    def test_discards_high_nan_ratio(self, quality_filter):
        """Instance with >10% NaN should be discarded."""
        series = np.array([1.0, np.nan, np.nan, np.nan, 5.0])  # 60% NaN
        result = quality_filter.fit_transform([series])
        assert len(result) == 0
        assert quality_filter.kept_indices_ == []

    def test_repairs_low_nan_ratio(self, quality_filter):
        """Instance with <=10% NaN should be repaired via interpolation."""
        # 1 NaN out of 20 = 5% NaN ratio
        series = np.arange(20, dtype=float)
        series[5] = np.nan
        result = quality_filter.fit_transform([series])

        assert len(result) == 1
        assert not np.isnan(result[0]).any()

    def test_interpolation_fills_middle_nan(self, quality_filter):
        series = np.array(
            [
                0.0,
                1.0,
                np.nan,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                16.0,
                17.0,
                18.0,
                19.0,
            ]
        )
        result = quality_filter.fit_transform([series])

        assert_allclose(result[0][2], 2.0, atol=1e-10)

    def test_discards_high_frozen_ratio(self, quality_filter):
        """Instance where all differences are zero (100% frozen) should be discarded."""
        series = np.ones(20)  # completely flat
        result = quality_filter.fit_transform([series])
        assert len(result) == 0

    def test_discards_empty_series(self, quality_filter):
        series = np.array([])
        result = quality_filter.fit_transform([series])
        assert len(result) == 0

    def test_kept_indices_tracking(self, quality_filter):
        """Verify kept_indices_ correctly tracks which instances survived."""
        good = np.arange(20, dtype=float)
        bad_nan = np.full(20, np.nan)
        good2 = np.arange(20, dtype=float) * 2

        result = quality_filter.fit_transform([good, bad_nan, good2])
        assert len(result) == 2
        assert quality_filter.kept_indices_ == [0, 2]

    def test_multiple_transforms_reset_state(self, quality_filter):
        """Second transform should reset kept_indices_."""
        X1 = [np.arange(20, dtype=float)]
        quality_filter.fit_transform(X1)
        assert quality_filter.kept_indices_ == [0]

        X2 = [np.full(20, np.nan)]
        quality_filter.transform(X2)
        assert quality_filter.kept_indices_ == []

    def test_custom_frozen_threshold(self):
        config = InstanceQualityConfig(
            frozen_threshold=0.5,
            max_nan_ratio=1.0,
            max_frozen_ratio=0.10,
        )
        qf = InstanceQualityFilter(config)
        # Series with small differences all <= 0.5
        series = np.array(
            [
                1.0,
                1.1,
                1.2,
                1.3,
                1.4,
                1.5,
                1.6,
                1.7,
                1.8,
                1.9,
                2.0,
                2.1,
                2.2,
                2.3,
                2.4,
                2.5,
                2.6,
                2.7,
                2.8,
                2.9,
            ]
        )
        result = qf.fit_transform([series])
        assert len(result) == 0


class TestInstanceQualityConfig:
    """Test suite for InstanceQualityConfig validation."""

    def test_default_values(self):
        config = InstanceQualityConfig()
        assert config.frozen_threshold == 0.0
        assert config.max_nan_ratio == 0.10
        assert config.max_frozen_ratio == 0.10

    def test_rejects_negative_frozen_threshold(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            InstanceQualityConfig(frozen_threshold=-1.0)

    def test_rejects_nan_ratio_above_one(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            InstanceQualityConfig(max_nan_ratio=1.5)
