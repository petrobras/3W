from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, field_validator
from .base_instantiable import Instantiable
from .dataset_outputs import DatasetOutputs


class BaseFeatureExtractorConfig(BaseModel, Instantiable):
    """Base configuration for feature extractors."""

    _target: type["BaseFeatureExtractor"]


class BaseFeatureExtractor(ABC):
    """Base class for feature extractors."""

    def __init__(self, config: BaseFeatureExtractorConfig):
        self.config = config

    @abstractmethod
    def transform(self, data: DatasetOutputs) -> DatasetOutputs:
        """Transform the data using the feature extractor."""
        raise NotImplementedError("Subclasses must implement the transform method.")


class OverlapOffsetMixin(BaseModel):
    """Mixin with common validations for overlap and offset."""

    overlap: float = Field(default=0.0, description="Overlap fraction between windows.")
    offset: int = Field(default=0, description="Offset for the window.")

    @field_validator("overlap")
    @classmethod
    def check_overlap_range(cls, overlap):
        """Validates that overlap is in the [0, 1) range."""
        if not 0 <= overlap < 1:
            raise ValueError("Overlap must be in the range [0, 1)")
        return overlap

    @field_validator("offset")
    @classmethod
    def check_offset_value(cls, offset):
        """Validates that offset is not negative."""
        if offset < 0:
            raise ValueError("Offset must be a non-negative integer.")
        return offset


class EpsMixin(BaseModel):
    """Mixin for positive epsilon validation."""

    eps: float = Field(default=1e-6, description="Small epsilon value for stability.")

    @field_validator("eps")
    @classmethod
    def check_eps_value(cls, eps):
        """Validates that epsilon is a small, positive number."""
        if eps <= 0:
            raise ValueError("Epsilon (eps) must be positive.")
        return eps


class WindowSizeMixin(BaseModel):
    """Mixin for window_size validation."""

    window_size: int = Field(default=100, description="Size of the window.")

    @field_validator("window_size")
    @classmethod
    def check_window_size(cls, window_size):
        """Validates that window_size is positive."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        return window_size


class FeatureSelectionMixin(BaseModel):
    """Mixin for feature selection."""

    selected_features: list[str] | None = Field(
        default=None, description="List of features to select."
    )
