"""
Data Augmentation Utilities for 3W Dataset

This module provides data augmentation techniques for the 3W dataset including:
- Noise injection (Gaussian, uniform, sensor-specific)
- Random undersampling for majority classes
- Random oversampling for minority classes
- SMOTE-like oversampling for time series data
- Class balancing strategies

These techniques should be applied during training to improve model robustness
and handle class imbalance effectively.
"""

import numpy as np
import pandas as pd
from sklearn.utils import resample
from collections import Counter
from typing import List, Tuple, Dict, Optional, Union
import random
from copy import deepcopy


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

    def add_noise(
        self,
        dfs: List[pd.DataFrame],
        classes: List[str],
        noise_type: str = "gaussian",
        noise_level: float = 0.01,
        target_columns: Optional[List[str]] = None,
    ) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        Add noise to the time series data for data augmentation.

        Args:
            dfs (List[pd.DataFrame]): List of dataframes
            classes (List[str]): List of corresponding classes
            noise_type (str): Type of noise ('gaussian', 'uniform', 'sensor_specific')
            noise_level (float): Noise level as fraction of signal standard deviation
            target_columns (List[str], optional): Columns to add noise to. If None, all numeric columns

        Returns:
            tuple: (augmented_dfs, augmented_classes) - original + noisy data
        """
        if self.verbose:
            print(f"🔊 Adding {noise_type} noise (level: {noise_level})")

        augmented_dfs = dfs.copy()
        augmented_classes = classes.copy()

        for i, df in enumerate(dfs):
            # Create noisy copy
            noisy_df = df.copy()

            # Determine which columns to add noise to
            if target_columns is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                numeric_cols = [col for col in numeric_cols if col != "class"]
            else:
                numeric_cols = target_columns

            for col in numeric_cols:
                if col in df.columns:
                    signal = df[col].values
                    signal_std = np.std(signal)

                    if noise_type == "gaussian":
                        noise = np.random.normal(
                            0, noise_level * signal_std, len(signal)
                        )
                    elif noise_type == "uniform":
                        noise_range = noise_level * signal_std
                        noise = np.random.uniform(
                            -noise_range, noise_range, len(signal)
                        )
                    elif noise_type == "sensor_specific":
                        # Different noise levels for different sensor types
                        if "P-TPT" in col:  # Pressure sensor
                            noise = np.random.normal(
                                0, noise_level * signal_std * 0.8, len(signal)
                            )
                        elif "T-TPT" in col:  # Temperature sensor
                            noise = np.random.normal(
                                0, noise_level * signal_std * 1.2, len(signal)
                            )
                        else:
                            noise = np.random.normal(
                                0, noise_level * signal_std, len(signal)
                            )

                    noisy_df[col] = signal + noise

            augmented_dfs.append(noisy_df)
            augmented_classes.append(classes[i])

        if self.verbose:
            print(f"   • Original samples: {len(dfs)}")
            print(f"   • Augmented samples: {len(augmented_dfs)} (2x original)")
            print(f"   • Noise applied to columns: {list(numeric_cols)}")

        return augmented_dfs, augmented_classes

    def random_undersample(
        self,
        dfs: List[pd.DataFrame],
        classes: List[str],
        target_class_counts: Optional[Dict[str, int]] = None,
        strategy: str = "auto",
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
        if self.verbose:
            print(f"📉 Random Undersampling (strategy: {strategy})")

        class_counts = Counter(classes)
        if self.verbose:
            print(f"   • Original class distribution: {dict(class_counts)}")

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

        if self.verbose:
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
        if self.verbose:
            print(f"📈 Random Oversampling (strategy: {strategy})")

        class_counts = Counter(classes)
        if self.verbose:
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

        if self.verbose:
            final_counts = Counter(oversampled_classes)
            print(f"   • Target class counts: {target_class_counts}")
            print(f"   • Final class distribution: {dict(final_counts)}")
            print(f"   • Samples: {len(classes)} → {len(oversampled_classes)}")

        return oversampled_dfs, oversampled_classes

    def smote_like_oversample(
        self,
        dfs: List[pd.DataFrame],
        classes: List[str],
        k_neighbors: int = 5,
        target_class_counts: Optional[Dict[str, int]] = None,
    ) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        SMOTE-like oversampling for time series data.
        Creates synthetic samples by interpolating between existing samples.

        Args:
            dfs (List[pd.DataFrame]): List of dataframes
            classes (List[str]): List of corresponding classes
            k_neighbors (int): Number of neighbors to consider for interpolation
            target_class_counts (Dict[str, int], optional): Target count per class

        Returns:
            tuple: (oversampled_dfs, oversampled_classes)
        """
        if self.verbose:
            print(f"🎯 SMOTE-like Oversampling (k={k_neighbors})")

        class_counts = Counter(classes)
        if self.verbose:
            print(f"   • Original class distribution: {dict(class_counts)}")

        if target_class_counts is None:
            max_count = max(class_counts.values())
            target_class_counts = {cls: max_count for cls in class_counts.keys()}

        # Group data by class
        class_to_data = {}
        for i, cls in enumerate(classes):
            if cls not in class_to_data:
                class_to_data[cls] = []
            class_to_data[cls].append((dfs[i], cls))

        oversampled_dfs = []
        oversampled_classes = []

        for cls, target_count in target_class_counts.items():
            if cls in class_to_data:
                class_data = class_to_data[cls]
                current_count = len(class_data)

                # Add original samples
                for df, class_label in class_data:
                    oversampled_dfs.append(df)
                    oversampled_classes.append(class_label)

                if current_count < target_count and current_count > 1:
                    needed_samples = target_count - current_count

                    for _ in range(needed_samples):
                        # Select random sample as base
                        base_idx = random.randint(0, current_count - 1)
                        base_df, base_class = class_data[base_idx]

                        # Select random neighbor for interpolation
                        neighbor_idx = random.randint(0, current_count - 1)
                        while neighbor_idx == base_idx and current_count > 1:
                            neighbor_idx = random.randint(0, current_count - 1)

                        neighbor_df, _ = class_data[neighbor_idx]

                        # Create synthetic sample by interpolation
                        alpha = random.random()  # Random interpolation factor
                        synthetic_df = base_df.copy()

                        numeric_cols = base_df.select_dtypes(
                            include=[np.number]
                        ).columns
                        numeric_cols = [col for col in numeric_cols if col != "class"]

                        for col in numeric_cols:
                            if col in base_df.columns and col in neighbor_df.columns:
                                # Ensure both series have same length
                                min_len = min(len(base_df[col]), len(neighbor_df[col]))
                                base_values = base_df[col].values[:min_len]
                                neighbor_values = neighbor_df[col].values[:min_len]

                                # Linear interpolation
                                synthetic_values = (
                                    1 - alpha
                                ) * base_values + alpha * neighbor_values
                                synthetic_df[col] = synthetic_values

                        oversampled_dfs.append(synthetic_df)
                        oversampled_classes.append(base_class)

        if self.verbose:
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
    ) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        Balance classes using combined over/undersampling strategy.

        Args:
            dfs (List[pd.DataFrame]): List of dataframes
            classes (List[str]): List of corresponding classes
            strategy (str): 'undersample', 'oversample', 'combined', 'smote'
            target_samples_per_class (int, optional): Target number of samples per class

        Returns:
            tuple: (balanced_dfs, balanced_classes)
        """
        if self.verbose:
            print(f"⚖️ Class Balancing Strategy: {strategy}")

        class_counts = Counter(classes)
        if self.verbose:
            print(f"   • Original class distribution: {dict(class_counts)}")

        if target_samples_per_class is None:
            if strategy == "undersample":
                target_samples_per_class = min(class_counts.values())
            elif strategy in ["oversample", "smote"]:
                target_samples_per_class = max(class_counts.values())
            elif strategy == "combined":
                target_samples_per_class = int(np.median(list(class_counts.values())))

        target_counts = {cls: target_samples_per_class for cls in class_counts.keys()}

        if strategy == "undersample":
            return self.random_undersample(dfs, classes, target_counts, "auto")
        elif strategy == "oversample":
            return self.random_oversample(dfs, classes, target_counts, "auto")
        elif strategy == "smote":
            return self.smote_like_oversample(
                dfs, classes, target_class_counts=target_counts
            )
        elif strategy == "combined":
            # First undersample majority classes
            balanced_dfs, balanced_classes = self.random_undersample(
                dfs, classes, target_counts, "auto"
            )
            # Then oversample minority classes
            return self.random_oversample(
                balanced_dfs, balanced_classes, target_counts, "auto"
            )

        return dfs, classes

    def create_augmentation_pipeline(
        self, dfs: List[pd.DataFrame], classes: List[str], config: Dict
    ) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        Create a complete data augmentation pipeline.

        Args:
            dfs (List[pd.DataFrame]): List of dataframes
            classes (List[str]): List of corresponding classes
            config (Dict): Augmentation configuration

        Returns:
            tuple: (augmented_dfs, augmented_classes)
        """
        if self.verbose:
            print("🔄 Data Augmentation Pipeline")
            print("=" * 50)

        augmented_dfs = dfs.copy()
        augmented_classes = classes.copy()

        # Step 1: Add noise (if enabled)
        if config.get("add_noise", False):
            noise_config = config.get("noise_config", {})
            augmented_dfs, augmented_classes = self.add_noise(
                augmented_dfs,
                augmented_classes,
                noise_type=noise_config.get("type", "gaussian"),
                noise_level=noise_config.get("level", 0.01),
                target_columns=noise_config.get("columns", None),
            )

        # Step 2: Balance classes (if enabled)
        if config.get("balance_classes", False):
            balance_config = config.get("balance_config", {})
            augmented_dfs, augmented_classes = self.balance_classes(
                augmented_dfs,
                augmented_classes,
                strategy=balance_config.get("strategy", "combined"),
                target_samples_per_class=balance_config.get("target_samples", None),
            )

        if self.verbose:
            original_counts = Counter(classes)
            final_counts = Counter(augmented_classes)
            print(f"\n📊 Pipeline Summary:")
            print(f"   • Original: {len(classes)} samples, {dict(original_counts)}")
            print(f"   • Final: {len(augmented_classes)} samples, {dict(final_counts)}")
            print(
                f"   • Augmentation factor: {len(augmented_classes)/len(classes):.2f}x"
            )

        return augmented_dfs, augmented_classes


# Convenience functions for common use cases
def quick_noise_augmentation(
    dfs: List[pd.DataFrame], classes: List[str], noise_level: float = 0.01
) -> Tuple[List[pd.DataFrame], List[str]]:
    """Quick noise augmentation with default settings."""
    augmentor = DataAugmentor(verbose=True)
    return augmentor.add_noise(dfs, classes, noise_level=noise_level)


def quick_balance_classes(
    dfs: List[pd.DataFrame],
    classes: List[str],
    strategy: str = "combined",
    min_samples_per_class: Optional[int] = None,
) -> Tuple[List[pd.DataFrame], List[str]]:
    """
    Quick class balancing with default settings.

    Args:
        dfs: List of dataframes
        classes: List of class labels
        strategy: Balancing strategy ('combined', 'oversample', 'undersample', 'smote')
        min_samples_per_class: Minimum samples per class (used as target_samples_per_class)

    Returns:
        tuple: (balanced_dfs, balanced_classes)
    """
    augmentor = DataAugmentor(verbose=True)
    return augmentor.balance_classes(
        dfs, classes, strategy=strategy, target_samples_per_class=min_samples_per_class
    )
