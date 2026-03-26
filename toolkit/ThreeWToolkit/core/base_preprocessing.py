from pydantic import BaseModel
from abc import ABC, abstractmethod
from .base_dataset import BaseDataset
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

    @abstractmethod
    def fit(self, data: BaseDataset) -> None:
        """If needed, fit the preprocessing step to the dataset."""
        pass
