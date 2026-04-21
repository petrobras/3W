from typing import Iterator

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

from pydantic import BaseModel, Field
from ..core.base_dataset import BaseDataset
from ..dataset.subset_dataset import SubsetDataset


class KFoldSplitter(BaseModel):
    """Utility class for splitting data into training and testing sets.
    Args:
        num_splits (int): Number of folds for cross-validation (must be at least 2).
        random_state (int | None): Random seed used when shuffling folds. Use None for non-deterministic splits.
        stratify_by (list[str]): List of metadata keys to stratify by (e.g. ['event_class', 'event_type']).\
                If empty, no stratification is applied.
    """

    num_splits: int = Field(
        default=5,
        gt=1,
        description="Number of folds for cross-validation (must be at least 2).",
    )
    random_state: int | None = Field(
        default=None,
        description="Random seed used when shuffling folds. Use None for non-deterministic splits.",
    )

    stratify_by: list[str] = Field(
        default_factory=list,
        description="List of metadata keys to stratify by (e.g. ['event_class', 'event_type']). If empty, no stratification is applied.",
    )

    def split_data(
        self, data: BaseDataset
    ) -> Iterator[tuple[SubsetDataset, SubsetDataset]]:
        """Splits a dataset into training and testing partitions using K-fold.
        Optionally stratifies the splits based on specified metadata keys.
        Args:
            data (BaseDataset): The dataset to split.
        Yields:
            tuple[SubsetDataset, SubsetDataset]: A tuple containing the training and testing subsets for each fold.
        Raises:
            ValueError: If any of the specified stratification keys are missing from the event metadata.
        """

        if len(self.stratify_by) == 0:
            # No stratification, just use KFold
            splitter = KFold(
                n_splits=self.num_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            for train_idx, test_idx in splitter.split(range(len(data))):
                yield SubsetDataset(data, train_idx), SubsetDataset(data, test_idx)
            return

        # Assign a pseudo-label for each event based on the requested stratification criteria
        event_pseudo_label = []
        for event in data:
            if not (all(key in event.metadata for key in self.stratify_by)):
                missing_keys = [
                    key for key in self.stratify_by if key not in event.metadata
                ]
                raise ValueError(
                    f"Event metadata missing required keys for stratification: {missing_keys}"
                )
            pseudo_label = tuple(event.metadata[key] for key in self.stratify_by)
            event_pseudo_label.append(pseudo_label)

        # assign unique integer values to each unique combination of class/type for stratification
        pseudo_label_to_int = {
            label: idx for idx, label in enumerate(set(event_pseudo_label))
        }

        # map the pseudo labels to integers for stratification
        stratify_y = [pseudo_label_to_int[label] for label in event_pseudo_label]
        stratify_x = list(range(len(data)))  # dummy x

        splitter = StratifiedKFold(
            n_splits=self.num_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        for train_idx, test_idx in splitter.split(stratify_x, stratify_y):
            yield SubsetDataset(data, train_idx), SubsetDataset(data, test_idx)


class TrainTestSplitter(BaseModel):
    """Utility class for splitting data into training and test sets.

    Args:
        size_training (float): Proportion of data to use for training (must be between 0 and 1).
        size_test (float): Proportion of data to use for test (must be between 0 and 1).
        shuffle (bool): Whether to shuffle indices before splitting.
        random_state (int | None): Random seed for reproducible splits. Use None for non-deterministic splits.
    """

    size_training: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Proportion of data to use for training (must be between 0 and 1).",
    )
    size_test: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Proportion of data to use for test (must be between 0 and 1).",
    )
    shuffle: bool = Field(
        default=True,
        description="Whether to shuffle indices before splitting.",
    )
    random_state: int | None = Field(
        default=None,
        description="Random seed for reproducible splits. Use None for non-deterministic splits.",
    )

    def split_data(self, data: BaseDataset) -> tuple[SubsetDataset, SubsetDataset]:
        """Splits a dataset into training and test subsets.

        Args:
            data (BaseDataset): The dataset to split.

        Returns:
            tuple[SubsetDataset, SubsetDataset]: A tuple containing (training_subset, test_subset).

        Raises:
            ValueError: If size_training + size_test != 1.
        """
        if self.size_training + self.size_test != 1.0:
            raise ValueError(
                f"The sum of size_training ({self.size_training}) and "
                f"size_test ({self.size_test}) must equal 1.0"
            )

        len_dataset = len(data)
        lst_indices = np.arange(len_dataset)

        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(lst_indices)

        split_point = int(len_dataset * self.size_training)
        training_indices = [int(i) for i in lst_indices[:split_point]]
        test_indices = [int(i) for i in lst_indices[split_point:]]

        training_set = SubsetDataset(data, indices=training_indices)
        test_set = SubsetDataset(data, indices=test_indices)

        return training_set, test_set
