"""
Data Augmentation Utilities for 3W Dataset

The 3W Toolkit has no class balancing, so the course keeps these helpers:
- Random undersampling for majority classes
- Random oversampling for minority classes
- A combined strategy of both

They are applied to the training set only.
"""

import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict, Optional
import random


class DataAugmentor:
    """
    Data augmentation utilities specialized for 3W time series dataset.
    """

    def __init__(self, random_state: int = 42, verbose: bool = True):
        """
        Initialize DataAugmentor.

        Args:
            random_state (int): Random state for reproducibility
            verbose (bool): Whether to print detailed information
        """
        self.random_state = random_state
        self.verbose = verbose
        np.random.seed(random_state)
        random.seed(random_state)

    def random_undersample(
        self,
        dfs: List[pd.DataFrame],
        classes: List[str],
        target_class_counts: Optional[Dict[str, int]] = None,
        strategy: str = "auto",
        internal_verbose: bool = False,
    ) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        Perform random undersampling to reduce majority classes.

        Args:
            dfs (List[pd.DataFrame]): List of dataframes
            classes (List[str]): List of corresponding classes
            target_class_counts (Dict[str, int], optional): Target count per class
            strategy (str): 'auto' (balance to minority), 'majority' (reduce majority only)

        Returns:
            tuple: (undersampled_dfs, undersampled_classes)
        """

        class_counts = Counter(classes)
        if internal_verbose:
            print(f"   • Original class distribution: {dict(class_counts)}")
            print(f"📉 Random Undersampling (strategy: {strategy})")

        if target_class_counts is None:
            if strategy == "auto":
                # Balance to the minority class
                min_count = min(class_counts.values())
                target_class_counts = {cls: min_count for cls in class_counts.keys()}
            elif strategy == "majority":
                # Only reduce classes that are above median
                median_count = np.median(list(class_counts.values()))
                target_class_counts = {
                    cls: min(count, int(median_count))
                    for cls, count in class_counts.items()
                }

        # Group data by class
        class_to_indices = {}
        for i, cls in enumerate(classes):
            if cls not in class_to_indices:
                class_to_indices[cls] = []
            class_to_indices[cls].append(i)

        # Undersample each class
        undersampled_indices = []
        for cls, target_count in target_class_counts.items():
            if cls in class_to_indices:
                indices = class_to_indices[cls]
                if len(indices) > target_count:
                    # Randomly sample target_count indices
                    sampled_indices = random.sample(indices, target_count)
                else:
                    # Keep all if below target
                    sampled_indices = indices
                undersampled_indices.extend(sampled_indices)

        # Create undersampled datasets
        undersampled_dfs = [dfs[i] for i in undersampled_indices]
        undersampled_classes = [classes[i] for i in undersampled_indices]

        if internal_verbose:
            final_counts = Counter(undersampled_classes)
            print(f"   • Target class counts: {target_class_counts}")
            print(f"   • Final class distribution: {dict(final_counts)}")
            print(f"   • Samples: {len(classes)} → {len(undersampled_classes)}")

        return undersampled_dfs, undersampled_classes

    def random_oversample(
        self,
        dfs: List[pd.DataFrame],
        classes: List[str],
        target_class_counts: Optional[Dict[str, int]] = None,
        strategy: str = "auto",
        internal_verbose: bool = False,
    ) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        Perform random oversampling to increase minority classes.

        Args:
            dfs (List[pd.DataFrame]): List of dataframes
            classes (List[str]): List of corresponding classes
            target_class_counts (Dict[str, int], optional): Target count per class
            strategy (str): 'auto' (balance to majority), 'minority' (increase minority only)

        Returns:
            tuple: (oversampled_dfs, oversampled_classes)
        """
        if internal_verbose:
            print(f"📈 Random Oversampling (strategy: {strategy})")

        class_counts = Counter(classes)
        if internal_verbose:
            print(f"   • Original class distribution: {dict(class_counts)}")

        if target_class_counts is None:
            if strategy == "auto":
                # Balance to the majority class
                max_count = max(class_counts.values())
                target_class_counts = {cls: max_count for cls in class_counts.keys()}
            elif strategy == "minority":
                # Only increase classes that are below median
                median_count = np.median(list(class_counts.values()))
                target_class_counts = {
                    cls: max(count, int(median_count))
                    for cls, count in class_counts.items()
                }

        # Group data by class
        class_to_data = {}
        for i, cls in enumerate(classes):
            if cls not in class_to_data:
                class_to_data[cls] = []
            class_to_data[cls].append((dfs[i], cls))

        # Oversample each class
        oversampled_dfs = []
        oversampled_classes = []

        for cls, target_count in target_class_counts.items():
            if cls in class_to_data:
                class_data = class_to_data[cls]
                current_count = len(class_data)

                if current_count < target_count:
                    # Add original samples
                    for df, class_label in class_data:
                        oversampled_dfs.append(df)
                        oversampled_classes.append(class_label)

                    # Add resampled samples
                    needed_samples = target_count - current_count
                    for _ in range(needed_samples):
                        # Randomly select a sample to duplicate
                        df, class_label = random.choice(class_data)
                        oversampled_dfs.append(df.copy())
                        oversampled_classes.append(class_label)
                else:
                    # Keep all original samples
                    for df, class_label in class_data:
                        oversampled_dfs.append(df)
                        oversampled_classes.append(class_label)

        if internal_verbose:
            final_counts = Counter(oversampled_classes)
            print(f"   • Target class counts: {target_class_counts}")
            print(f"   • Final class distribution: {dict(final_counts)}")
            print(f"   • Samples: {len(classes)} → {len(oversampled_classes)}")

        return oversampled_dfs, oversampled_classes

    def balance_classes(
        self,
        dfs: List[pd.DataFrame],
        classes: List[str],
        strategy: str = "combined",
        target_samples_per_class: Optional[int] = None,
        internal_verbose: bool = False,
    ) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        Balance classes using combined over/undersampling strategy.

        Args:
            dfs (List[pd.DataFrame]): List of dataframes
            classes (List[str]): List of corresponding classes
            strategy (str): 'undersample', 'oversample', 'combined'
            target_samples_per_class (int, optional): Target number of samples per class

        Returns:
            tuple: (balanced_dfs, balanced_classes)
        """
        if internal_verbose:
            print(f"⚖️ Class Balancing Strategy: {strategy}")

        class_counts = Counter(classes)
        if internal_verbose:
            print(f"   • Original class distribution: {dict(class_counts)}")

        if target_samples_per_class is None:
            if strategy == "undersample":
                target_samples_per_class = min(class_counts.values())
            elif strategy == "oversample":
                target_samples_per_class = max(class_counts.values())
            elif strategy == "combined":
                target_samples_per_class = int(np.median(list(class_counts.values())))

        target_counts = {cls: target_samples_per_class for cls in class_counts.keys()}

        if strategy == "undersample":
            return self.random_undersample(
                dfs, classes, target_counts, "auto", internal_verbose
            )
        elif strategy == "oversample":
            return self.random_oversample(
                dfs, classes, target_counts, "auto", internal_verbose
            )
        elif strategy == "combined":
            # First undersample majority classes
            balanced_dfs, balanced_classes = self.random_undersample(
                dfs, classes, target_counts, "auto", internal_verbose
            )
            # Then oversample minority classes
            return self.random_oversample(
                balanced_dfs, balanced_classes, target_counts, "auto", internal_verbose
            )

        return dfs, classes


def quick_balance_classes(
    dfs: List[pd.DataFrame],
    classes: List[str],
    strategy: str = "combined",
    min_samples_per_class: Optional[int] = None,
    internal_verbose: bool = False,
) -> Tuple[List[pd.DataFrame], List[str]]:
    """
    Quick class balancing with default settings.

    Args:
        dfs: List of dataframes
        classes: List of class labels
        strategy: Balancing strategy ('combined', 'oversample', 'undersample')
        min_samples_per_class: Minimum samples per class (used as target_samples_per_class)

    Returns:
        tuple: (balanced_dfs, balanced_classes)
    """
    augmentor = DataAugmentor(verbose=True)
    return augmentor.balance_classes(
        dfs,
        classes,
        strategy=strategy,
        target_samples_per_class=min_samples_per_class,
        internal_verbose=internal_verbose,
    )
