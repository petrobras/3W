from pydantic import BaseModel
from abc import ABC, abstractmethod
from .base_dataset import BaseDataset
from .base_instantiable import Instantiable
from .dataset_outputs import DatasetOutputs


class BasePreprocessingConfig(BaseModel, Instantiable):
    """Base configuration for preprocessing steps."""

    _target: type["BasePreprocessing"]


class BasePreprocessing(ABC):
    """Base class for preprocessing steps."""

    def __init__(self, config: BasePreprocessingConfig):
        self.config = config

    @abstractmethod
    def transform(self, data: DatasetOutputs) -> DatasetOutputs:
        """Transform the data using the fitted preprocessing step.

        Args:
            data: Input dataset outputs to transform.

        Returns:
            Transformed dataset outputs.
        """
        raise NotImplementedError("Subclasses must implement the transform method.")

    def fit(self, data: BaseDataset) -> None:
        """Fit the preprocessing step to the dataset if needed.

        Args:
            data: Dataset to fit the step on.
        """
        pass  # do nothing by default.
