from ThreeWToolkit.core.base_dataset import BaseDataset
from ThreeWToolkit.core.dataset_outputs import DatasetOutputs
from pydantic import BaseModel
from abc import ABC, abstractmethod
from .base_instantiable import Instantiable


class BaseTransformConfig(BaseModel, Instantiable):
    """Base configuration for general transformation steps."""

    _target: type["BaseTransform"]


class BaseTransform(ABC):
    """Base class for general transformation steps."""

    def __init__(self, config: BaseTransformConfig):
        self.config = config

    @abstractmethod
    def fit(self, dataset: BaseDataset) -> None:
        """If needed, fit the transformation step to the data.
        By default, this does nothing, as some methods won't need fitting."""
        pass

    def transform_event(self, data: DatasetOutputs) -> DatasetOutputs:
        """Transform a single event using the fitted transformation step.
        By default, this method raises NotImplementedError, as subclasses should implement it."""
        raise NotImplementedError(
            "Subclasses must implement the transform_event method."
        )

    def transform(self, dataset: BaseDataset) -> BaseDataset:
        raise NotImplementedError("Subclasses must implement the transform method.")
