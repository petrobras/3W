"""Dataset adapter for transformed datasets."""

from typing import Callable
from ..core.base_dataset import BaseDataset, DatasetOutputs


class TransformedDataset(BaseDataset):
    """Dataset class for transformed datasets.
    This class adapts an existing dataset by applying a transformation function to its outputs. The transformation
    function is applied on-the-fly when accessing the dataset items, allowing for dynamic transformations without
    modifying the original dataset.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        transform: Callable[[DatasetOutputs], DatasetOutputs],
    ):
        """Initialize the TransformedDataset with the original dataset and the transformation function.
        Args:
            dataset (BaseDataset): The original dataset to be transformed.
            transform (Callable[[DatasetOutputs], DatasetOutputs]): A function that takes the output of the original
            dataset and transforms it into the desired format. This function will be applied to each item of the dataset
            when accessed.
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        """Return the length of the dataset, which is the same as the original dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> DatasetOutputs:
        """Return the transformed output of the original dataset at the given index. The transformation is applied
        on-the-fly when accessing the item.
        Args:
            idx (int): The index of the item to access.
        Returns:
            DatasetOutputs: The transformed output of the original dataset at the given index.
        """
        return self.transform(self.dataset[idx])
