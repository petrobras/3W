from pydantic import BaseModel, field_validator
import pandas as pd
from abc import ABC, abstractmethod
from .base_instantiable import Instantiable


class BaseFeatureExtractorConfig(BaseModel, Instantiable):
    """Base configuration for feature extractors."""

    target_: type["BaseFeatureExtractor"]


class BaseFeatureExtractor(ABC):
    """Base class for feature extractors."""

    def __init__(self, config: BaseFeatureExtractorConfig):
        self.config = config

    # @abstractmethod
    # def fit(self, data: pd.DataFrame) -> pd.DataFrame:
    #     """If needed, fit the feature extractor to the data.
    #     By default, this does nothing, as some methods won't need fitting."""
    #     return data

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using the fitted feature extractor
        or directly if no fitting is needed."""
        raise NotImplementedError("Subclasses must implement the transform method.")

    # def fit_and_transform(self, data: pd.DataFrame) -> pd.DataFrame:
    #     """Convenience method to fit and transform the data in one step."""
    #     self.fit(data)
    #     return self.transform(data)


class OverlapOffsetMixin(BaseModel):
    """Mixin with common validations for overlap and offset."""

    overlap: float = 0.0
    offset: int = 0

    @field_validator("overlap")
    @classmethod
    def check_overlap_range(cls, v):
        """Validates that overlap is in the [0, 1) range."""
        if not 0 <= v < 1:
            raise ValueError("Overlap must be in the range [0, 1)")
        return v

    @field_validator("offset")
    @classmethod
    def check_offset_value(cls, v):
        """Validates that offset is not negative."""
        if v < 0:
            raise ValueError("Offset must be a non-negative integer.")
        return v


class EpsMixin(BaseModel):
    """Mixin for positive epsilon validation."""

    eps: float = 1e-6

    @field_validator("eps")
    @classmethod
    def check_eps_value(cls, v):
        """Validates that epsilon is a small, positive number."""
        if v <= 0:
            raise ValueError("Epsilon (eps) must be positive.")
        return v


class WindowSizeMixin(BaseModel):
    """Mixin for window_size validation."""

    window_size: int = 100

    @field_validator("window_size")
    @classmethod
    def check_window_size(cls, v):
        """Validates that window_size is positive."""
        if v <= 0:
            raise ValueError("Window size must be a positive integer.")
        return v


class FeatureSelectionMixin(BaseModel):
    """Mixin for feature selection."""

    selected_features: list[str] | None = None
