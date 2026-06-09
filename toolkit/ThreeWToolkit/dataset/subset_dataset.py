"""SubsetDataset for slicing datasets by indices."""

from typing import Sequence
import numpy as np
import numpy.typing as npt

from ..core.base_dataset import BaseDataset
from ..core.dataset_outputs import DatasetOutputs


class SubsetDataset(BaseDataset):
    """
    Dataset wrapper that provides a subset view using indices.

    This is useful for:
    - Train/validation splits
    - Cross-validation folds
    - Sampling subsets for debugging

    Similar to torch.utils.data.Subset but works with any BaseDataset.
    """

    def __init__(
        self, dataset: BaseDataset, indices: Sequence[int] | npt.NDArray[np.integer]
    ):
        """
        Initialize subset dataset.

        Args:
            dataset: The underlying dataset to subset.
            indices: List of indices to include in this subset.

        Raises:
            ValueError: If indices are out of bounds or empty.
        """
        self.dataset = dataset
        self.indices = indices

        if len(self.indices) == 0:
            raise ValueError("Indices list cannot be empty")

        # Validate indices are within bounds
        dataset_len = len(dataset)
        for idx in self.indices:
            if idx < 0 or idx >= dataset_len:
                raise ValueError(
                    f"Index {idx} out of bounds for dataset of length {dataset_len}"
                )

    def __len__(self) -> int:
        """Return the number of samples in this subset.

        Returns:
            Number of samples in the subset.
        """
        return len(self.indices)

    def __getitem__(self, idx: int) -> DatasetOutputs:
        """
        Get sample at the given subset index.

        Args:
            idx: Index in the subset (0 to len(self)-1).

        Returns:
            DatasetOutputs from the underlying dataset.
        """
        if idx < 0 or idx >= len(self.indices):
            raise IndexError(
                f"Subset index {idx} out of bounds for subset of length {len(self.indices)}"
            )

        # Map subset index to original dataset index
        original_idx = self.indices[idx]
        return self.dataset[original_idx]
