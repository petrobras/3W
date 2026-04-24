from pydantic import BaseModel
from abc import ABC, abstractmethod

from .base_dataset import BaseDataset
from .dataset_outputs import DatasetOutputs
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
        """Fit the transformation step to the dataset if needed.

        Args:
            dataset: Dataset to fit on.
        """
        pass

    def transform_event(self, data: DatasetOutputs) -> DatasetOutputs:
        """Transform a single event using the fitted transformation step.

        Args:
            data: Single event's dataset outputs to transform.

        Returns:
            Transformed dataset outputs.
        """
        raise NotImplementedError(
            "Subclasses must implement the transform_event method."
        )

    def transform(self, dataset: BaseDataset) -> BaseDataset:
        raise NotImplementedError("Subclasses must implement the transform method.")
