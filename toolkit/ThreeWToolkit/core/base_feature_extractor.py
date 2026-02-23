from typing import ClassVar
from .enums import (
    AvailableWaveletsEnum,
    AvailableEWStatisticalFeaturesEnum,
    AvailableStatisticalFeaturesEnum,
)
from pydantic import BaseModel, field_validator


class OverlapOffsetMixin(BaseModel):
    """Mixin with common validations for overlap and offset."""

    overlap: float = 0.0
    offset: int = 0

    @field_validator("overlap")
    @classmethod
    def check_overlap_range(cls: type["OverlapOffsetMixin"], overlap: float) -> float:
        """
        Validate that overlap is in the [0, 1) range.

        Args:
            cls (OverlapOffsetMixin): The class reference.
            overlap (float): Overlap value to validate.

        Returns:
            float: Validated overlap value.

        Raises:
            ValueError: If overlap is not in [0, 1).
        """
        if not 0 <= overlap < 1:
            raise ValueError("Overlap must be in the range [0, 1)")
        return overlap

    @field_validator("offset")
    @classmethod
    def check_offset_value(cls: type["OverlapOffsetMixin"], offset: int) -> int:
        """
        Validate that offset is not negative.

        Args:
            cls (OverlapOffsetMixin): The class reference.
            offset (int): Offset value to validate.

        Returns:
            int: Validated offset value.

        Raises:
            ValueError: If offset is negative.
        """
        if offset < 0:
            raise ValueError("Offset must be a non-negative integer.")
        return offset


class EpsMixin(BaseModel):
    """Mixin for positive epsilon validation."""

    eps: float = 1e-6

    @field_validator("eps")
    @classmethod
    def check_eps_value(cls: type["EpsMixin"], eps: float) -> float:
        """
        Validate that epsilon is a small, positive number.

        Args:
            cls (EpsMixin): The class reference.
            eps (float): Epsilon value to validate.

        Returns:
            float: Validated epsilon value.

        Raises:
            ValueError: If eps is not positive.
        """
        if eps <= 0:
            raise ValueError("Epsilon (eps) must be positive.")
        return eps


class WindowSizeMixin(BaseModel):
    """Mixin for window_size validation."""

    window_size: int = 100

    @field_validator("window_size")
    @classmethod
    def check_window_size(cls: type["WindowSizeMixin"], window_size: int) -> int:
        """
        Validate that window_size is positive.

        Args:
            cls (WindowSizeMixin): The class reference.
            window_size (int): Window size to validate.

        Returns:
            int: Validated window size.

        Raises:
            ValueError: If window_size is not positive.
        """
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        return window_size


class FeatureSelectionMixin(BaseModel):
    """Mixin for feature selection."""

    selected_features: list[str] | None = None


class StatisticalConfig(
    OverlapOffsetMixin, EpsMixin, WindowSizeMixin, FeatureSelectionMixin
):
    """
    Configuration for the Statistical feature extractor.

    Attributes:
        overlap (float): Fractional overlap between windows (0 <= overlap < 1).
        offset (int): Offset for windowing, must be non-negative.
        eps (float): Small positive value for numerical stability.
        window_size (int): Size of the window for feature extraction (must be > 0).
        selected_features (list[str] | None): List of statistical features to extract.
          From: (mean, std, var, skewness, kurtosis, min, max, median, rms, ptp, crest_factor, shape_factor, impulse_factor, margin_factor).

    """

    AVAILABLE_FEATURES: ClassVar[list[str]] = [
        feature.value for feature in AvailableStatisticalFeaturesEnum
    ]

    @field_validator("selected_features")
    @classmethod
    def validate_selected_features(
        cls: type["StatisticalConfig"], selected_features: list[str] | None
    ) -> list[str] | None:
        """
        Validate that selected features are available.

        Args:
            cls (StatisticalConfig): The class reference.
            selected_features (list[str] | None): List of features selected for extraction.

        Returns:
            list[str] | None: Validated list of selected features.

        Raises:
            ValueError: If any selected feature is not available.
        """
        if selected_features is not None:
            invalid_features = set(selected_features) - set(cls.AVAILABLE_FEATURES)
            if invalid_features:
                raise ValueError(
                    f"Invalid features: {invalid_features}. "
                    f"Available features: {cls.AVAILABLE_FEATURES}"
                )
        return selected_features


class EWStatisticalConfig(
    OverlapOffsetMixin, EpsMixin, WindowSizeMixin, FeatureSelectionMixin
):
    """
    Configuration for the Exponentially Weighted Statistical feature extractor.

    Attributes:
        overlap (float): Fractional overlap between windows (0 <= overlap < 1).
        offset (int): Offset for windowing, must be non-negative.
        eps (float): Small positive value for numerical stability.
        window_size (int): Size of the window for feature extraction (must be > 0).
        decay (float): Decay factor for exponential weighting (0 < decay <= 1).
        selected_features (list[str] | None): List of exponentially weighted statistical features to extract.
            From: (ew_mean, ew_std, ew_skew, ew_kurt, ew_min, ew_1qrt, ew_med, ew_3qrt, ew_max).

    """

    decay: float = 0.95

    AVAILABLE_FEATURES: ClassVar[list[str]] = [
        feature.value for feature in AvailableEWStatisticalFeaturesEnum
    ]

    @field_validator("decay")
    @classmethod
    def check_decay_range(cls: type["EWStatisticalConfig"], decay: float) -> float:
        """
        Validate that decay is in the (0, 1] range.

        Args:
            cls (EWStatisticalConfig): The class reference.
            decay (float): Decay value to validate.

        Returns:
            float: Validated decay value.

        Raises:
            ValueError: If decay is not in (0, 1].
        """
        if not 0 < decay <= 1:
            raise ValueError("Decay must be in the range (0, 1]")
        return decay

    @field_validator("selected_features")
    @classmethod
    def validate_selected_features(
        cls: type["EWStatisticalConfig"], selected_features: list[str] | None
    ) -> list[str] | None:
        """
        Validate that selected features are available.

        Args:
            cls (EWStatisticalConfig): The class reference.
            selected_features (list[str] | None): List of features selected for extraction.

        Returns:
            list[str] | None: Validated list of selected features.

        Raises:
            ValueError: If any selected feature is not available.
        """
        if selected_features is not None:
            invalid_features = set(selected_features) - set(cls.AVAILABLE_FEATURES)
            if invalid_features:
                raise ValueError(
                    f"Invalid features: {invalid_features}. "
                    f"Available features: {cls.AVAILABLE_FEATURES}"
                )
        return selected_features


class WaveletConfig(OverlapOffsetMixin, FeatureSelectionMixin):
    """
    Configuration for the Wavelet feature extractor.

    Attributes:
        level (int): The decomposition level for the wavelet transform (must be >= 1).
        wavelet (str): The wavelet type to use for feature extraction.
          From: (haar, db1, db2, db3, db4, db5, db6, db7, db8, db9, db10, bior2.2, bior4.4, coif2, coif4, dmey).

    This configuration ensures that only supported wavelet types are used and that the level is a positive integer.
    """

    level: int = 1
    wavelet: str = "haar"
    AVAILABLE_WAVELETS: ClassVar[list[str]] = [
        wavelet.value for wavelet in AvailableWaveletsEnum
    ]

    @field_validator("level")
    @classmethod
    def check_level_is_positive(cls: type["WaveletConfig"], level: int) -> int:
        """
        Validate that the wavelet level is a positive integer.

        Args:
            cls (WaveletConfig): The class reference.
            level (int): Decomposition level to validate.

        Returns:
            int: Validated level.

        Raises:
            ValueError: If level is less than 1.
        """
        if level < 1:
            raise ValueError("Wavelet level must be a positive integer (>= 1).")
        return level

    @field_validator("wavelet")
    @classmethod
    def check_wavelet_name(cls: type["WaveletConfig"], wavelet: str) -> str:
        """
        Validate that the wavelet name is supported by AvailableWaveletsEnum.

        Args:
            cls (WaveletConfig): The class reference.
            wavelet (str): Wavelet name to validate.

        Returns:
            str: Validated wavelet name.

        Raises:
            ValueError: If wavelet name is not supported.
        """
        if wavelet not in cls.AVAILABLE_WAVELETS:
            raise ValueError(
                f"Wavelet '{wavelet}' is not supported. "
                f"Available wavelets: {cls.AVAILABLE_WAVELETS}"
            )
        return wavelet
