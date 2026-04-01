"""Tests for BaseFeatureExtractor and related mixins."""

import pytest
from pydantic import ValidationError

from ThreeWToolkit.core import (
    BaseFeatureExtractor,
    BaseFeatureExtractorConfig,
    OverlapOffsetMixin,
    EpsMixin,
    WindowSizeMixin,
    FeatureSelectionMixin,
    DatasetOutputs,
)


class TestOverlapOffsetMixin:
    """Test OverlapOffsetMixin validation."""

    def test_valid_overlap_zero(self):
        """Test valid overlap of 0."""

        class TestConfig(OverlapOffsetMixin):
            pass

        config = TestConfig(overlap=0.0)
        assert config.overlap == 0.0

    def test_valid_overlap_positive(self):
        """Test valid overlap in range (0, 1)."""

        class TestConfig(OverlapOffsetMixin):
            pass

        config = TestConfig(overlap=0.5)
        assert config.overlap == 0.5

    def test_overlap_boundary_below_one(self):
        """Test overlap at boundary just below 1."""

        class TestConfig(OverlapOffsetMixin):
            pass

        config = TestConfig(overlap=0.99)
        assert config.overlap == 0.99

    def test_invalid_overlap_equals_one(self):
        """Test that overlap=1.0 raises error."""

        class TestConfig(OverlapOffsetMixin):
            pass

        with pytest.raises(ValidationError):
            TestConfig(overlap=1.0)

    def test_invalid_overlap_greater_than_one(self):
        """Test that overlap > 1.0 raises error."""

        class TestConfig(OverlapOffsetMixin):
            pass

        with pytest.raises(ValidationError):
            TestConfig(overlap=1.5)

    def test_invalid_overlap_negative(self):
        """Test that negative overlap raises error."""

        class TestConfig(OverlapOffsetMixin):
            pass

        with pytest.raises(ValidationError):
            TestConfig(overlap=-0.1)

    def test_valid_offset_zero(self):
        """Test valid offset of 0."""

        class TestConfig(OverlapOffsetMixin):
            pass

        config = TestConfig(offset=0)
        assert config.offset == 0

    def test_valid_offset_positive(self):
        """Test valid positive offset."""

        class TestConfig(OverlapOffsetMixin):
            pass

        config = TestConfig(offset=100)
        assert config.offset == 100

    def test_invalid_offset_negative(self):
        """Test that negative offset raises error."""

        class TestConfig(OverlapOffsetMixin):
            pass

        with pytest.raises(ValidationError):
            TestConfig(offset=-1)

    def test_default_values(self):
        """Test default values for overlap and offset."""

        class TestConfig(OverlapOffsetMixin):
            pass

        config = TestConfig()
        assert config.overlap == 0.0
        assert config.offset == 0


class TestEpsMixin:
    """Test EpsMixin validation."""

    def test_valid_eps_default(self):
        """Test default epsilon value."""

        class TestConfig(EpsMixin):
            pass

        config = TestConfig()
        assert config.eps == 1e-6

    def test_valid_eps_custom(self):
        """Test custom positive epsilon."""

        class TestConfig(EpsMixin):
            pass

        config = TestConfig(eps=1e-8)
        assert config.eps == 1e-8

    def test_valid_eps_large(self):
        """Test larger epsilon value."""

        class TestConfig(EpsMixin):
            pass

        config = TestConfig(eps=0.001)
        assert config.eps == 0.001

    def test_invalid_eps_zero(self):
        """Test that eps=0 raises error."""

        class TestConfig(EpsMixin):
            pass

        with pytest.raises(ValidationError):
            TestConfig(eps=0)

    def test_invalid_eps_negative(self):
        """Test that negative eps raises error."""

        class TestConfig(EpsMixin):
            pass

        with pytest.raises(ValidationError):
            TestConfig(eps=-1e-6)


class TestWindowSizeMixin:
    """Test WindowSizeMixin validation."""

    def test_valid_window_size_default(self):
        """Test default window size."""

        class TestConfig(WindowSizeMixin):
            pass

        config = TestConfig()
        assert config.window_size == 100

    def test_valid_window_size_custom(self):
        """Test custom window size."""

        class TestConfig(WindowSizeMixin):
            pass

        config = TestConfig(window_size=256)
        assert config.window_size == 256

    def test_valid_window_size_one(self):
        """Test window size of 1."""

        class TestConfig(WindowSizeMixin):
            pass

        config = TestConfig(window_size=1)
        assert config.window_size == 1

    def test_invalid_window_size_zero(self):
        """Test that window_size=0 raises error."""

        class TestConfig(WindowSizeMixin):
            pass

        with pytest.raises(ValidationError):
            TestConfig(window_size=0)

    def test_invalid_window_size_negative(self):
        """Test that negative window_size raises error."""

        class TestConfig(WindowSizeMixin):
            pass

        with pytest.raises(ValidationError):
            TestConfig(window_size=-10)


class TestFeatureSelectionMixin:
    """Test FeatureSelectionMixin."""

    def test_default_selected_features_none(self):
        """Test default selected_features is None."""

        class TestConfig(FeatureSelectionMixin):
            pass

        config = TestConfig()
        assert config.selected_features is None

    def test_selected_features_list(self):
        """Test setting selected features list."""

        class TestConfig(FeatureSelectionMixin):
            pass

        config = TestConfig(selected_features=["feature_a", "feature_b"])
        assert config.selected_features == ["feature_a", "feature_b"]

    def test_selected_features_empty_list(self):
        """Test setting empty selected features list."""

        class TestConfig(FeatureSelectionMixin):
            pass

        config = TestConfig(selected_features=[])
        assert config.selected_features == []


class TestCombinedMixins:
    """Test using multiple mixins together."""

    def test_combined_mixins(self):
        """Test config with multiple mixins."""

        class CombinedConfig(OverlapOffsetMixin, WindowSizeMixin, EpsMixin):
            pass

        config = CombinedConfig(overlap=0.25, offset=10, window_size=512, eps=1e-5)

        assert config.overlap == 0.25
        assert config.offset == 10
        assert config.window_size == 512
        assert config.eps == 1e-5

    def test_combined_mixins_defaults(self):
        """Test combined mixins with default values."""

        class CombinedConfig(OverlapOffsetMixin, WindowSizeMixin, EpsMixin):
            pass

        config = CombinedConfig()

        assert config.overlap == 0.0
        assert config.offset == 0
        assert config.window_size == 100
        assert config.eps == 1e-6
