from pydantic import BaseModel
from abc import ABC, abstractmethod
from .base_instantiable import Instantiable
from .dataset_outputs import DatasetOutputs


class BasePreprocessingConfig(BaseModel, Instantiable):
    """Base configuration for preprocessing steps."""

    target_: type["BasePreprocessing"]


class BasePreprocessing(ABC):
    """Base class for preprocessing steps."""

    def __init__(self, config: BasePreprocessingConfig):
        self.config = config

    @abstractmethod
    def transform(self, data: DatasetOutputs) -> DatasetOutputs:
        """Transform the data using the fitted preprocessing step."""
        raise NotImplementedError("Subclasses must implement the transform method.")

    def fit(self, data: DatasetOutputs) -> None:
        """If needed, fit the preprocessing step to the data."""
        pass

    def compute(self) -> None:
        """Compute statistics after fitting. Override if needed."""
        pass
