""" Definition for the base dataset class. """

from abc import ABC, abstractmethod
from pydantic import BaseModel

from .dataset_outputs import DatasetOutputs
from .base_instantiable import Instantiable


class BaseDatasetConfig(BaseModel, Instantiable):
    """Base configuration for datasets."""
    target_: type["BaseDataset"]



class BaseDataset(ABC):
    """Base class for datasets."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        raise NotImplementedError("Subclasses must implement the __len__ method.")

    @abstractmethod
    def __getitem__(self, idx: int) -> DatasetOutputs:
        """Return the sample at the given index."""
        raise NotImplementedError("Subclasses must implement the __getitem__ method.")
