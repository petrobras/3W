"""Dataset adapter for transformed datasets."""

from typing import Callable
from ..core.base_dataset import BaseDataset, DatasetOutputs


class TransformedDataset(BaseDataset):
    """Dataset class for transformed datasets."""

    def __init__(
        self,
        dataset: BaseDataset,
        transform: Callable[[DatasetOutputs], DatasetOutputs],
    ):
        """Initialize the TransformedDataset with the original dataset and the transformation function."""
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> DatasetOutputs:
        return self.transform(self.dataset[idx])
