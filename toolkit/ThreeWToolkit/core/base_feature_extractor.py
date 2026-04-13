from abc import ABC, abstractmethod
from pydantic import BaseModel
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
