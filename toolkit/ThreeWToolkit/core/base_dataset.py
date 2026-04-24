"""Definition for the base dataset class."""

from abc import ABC, abstractmethod
from typing import Iterator
from pydantic import BaseModel

from .dataset_outputs import DatasetOutputs
from .base_instantiable import Instantiable


class BaseDatasetConfig(BaseModel, Instantiable):
    """Base configuration for datasets."""

    _target: type["BaseDataset"]


class BaseDataset(ABC):
    """Base class for datasets."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Total number of samples.
        """
        raise NotImplementedError("Subclasses must implement the __len__ method.")

    @abstractmethod
    def __getitem__(self, idx: int) -> DatasetOutputs:
        """Return the sample at the given index.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            DatasetOutputs containing the sample's signal, label, and metadata.
        """
        raise NotImplementedError("Subclasses must implement the __getitem__ method.")

    def __iter__(self) -> Iterator[DatasetOutputs]:
        """Return an iterator over the dataset.

        Returns:
            Iterator yielding DatasetOutputs for each sample in order.
        """
        yield from (self[i] for i in range(len(self)))
