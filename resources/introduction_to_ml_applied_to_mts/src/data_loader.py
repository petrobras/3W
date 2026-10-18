"""
Data Loading Module for 3W Dataset

This module provides functionality to load and organize the 3W dataset
containing sensor data for fault detection in oil wells.
"""

import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter

from ThreeWToolkit.dataset import ParquetDataset, ParquetDatasetConfig

from . import config

CLASS_FOLDERS = [str(i) for i in range(10)]


class DataLoader:
    """
    A class for loading and organizing the 3W dataset.

    File discovery and reading are delegated to the official 3W Toolkit
    (`ThreeWToolkit.dataset.ParquetDataset`). The dataset is organized in
    folders representing the fault classes (0-9), and each file is an event
    of one of three provenances: real (`WELL`), simulated (`SIMULATED`) or
    hand-drawn (`DRAWN`).
    """

    def __init__(self, dataset_path: str = "../../dataset/"):
        """
        Initialize the DataLoader.

        Args:
            dataset_path (str): Path to the 3W dataset directory
        """
        self.dataset_path = dataset_path
        self.class_folders = []
        self.stats = {}

    def _open_events(self, class_folders: List[str], event_types: List[str]):
        """Open the 3W events of the given classes and provenances (sorted by path)."""
        return ParquetDataset(
            ParquetDatasetConfig(
                path=self.dataset_path,
                target_class=[int(c) for c in class_folders],
                event_type=event_types,
                target_column=None,
            )
        )

    @staticmethod
    def _group_by_class(dataset: ParquetDataset) -> Dict[str, List[int]]:
        """Map each class folder to the indices of its events in `dataset`."""
        groups: Dict[str, List[int]] = {}
        for idx, event in enumerate(dataset.files_events):
            groups.setdefault(event.parent.name, []).append(idx)
        return groups

    def explore_structure(self) -> Dict[str, int | str | List[str]]:
        """
        Explore the dataset structure and return information about files.

        Returns:
            Dict containing information about the dataset structure
        """
        print("Dataset Structure:")
        print("=" * 50)

        structure_info = {
            "classes": [],
            "total_files": 0,
            "real_files": 0,
            "simulated_files": 0,
            "drawn_files": 0,
        }

        per_type = {
            event_type: self._group_by_class(
                self._open_events(CLASS_FOLDERS, [event_type])
            )
            for event_type in ("real", "simulated", "drawn")
        }

        for folder_name in CLASS_FOLDERS:
            if not os.path.isdir(os.path.join(self.dataset_path, folder_name)):
                continue

            self.class_folders.append(folder_name)
            structure_info["classes"].append(folder_name)

            n_real = len(per_type["real"].get(folder_name, []))
            n_simulated = len(per_type["simulated"].get(folder_name, []))
            n_drawn = len(per_type["drawn"].get(folder_name, []))
            total = n_real + n_simulated + n_drawn
            structure_info["total_files"] += total

            print(f"Class {folder_name}: {total} parquet files")

            structure_info["real_files"] += n_real
            structure_info["simulated_files"] += n_simulated
            structure_info["drawn_files"] += n_drawn

            print(f"  └── Real data: {n_real} files")
            print(f"  └── Simulated data: {n_simulated} files")
            print(f"  └── Hand-drawn data: {n_drawn} files")
            print()
        print(f"Total classes found: {len(self.class_folders)}")
        print(f"Classes: {self.class_folders}")

        return structure_info

    def load_dataset(
        self, class_folders: List[str] = None, max_files_per_class: int = 150
    ) -> Tuple[
        List[pd.DataFrame],
        List[str],
        Dict[str, int | float | str | List[str] | pd.DataFrame],
    ]:
        """
        Load the 3W dataset from parquet files.

        Args:
            class_folders (List[str], optional): List of class folders to load.
                                               If None, loads all available classes.
            max_files_per_class (int): Maximum number of files to load per class.
                                     Real data files are prioritized over simulated.

        Returns:
            Tuple containing:
                - List of dataframes with loaded data
                - List of corresponding class labels
                - Dictionary with dataset statistics
        """
        if class_folders is None:
            class_folders = self.class_folders

        # Initialize storage containers - separate real and simulated
        dfs_real = []
        dfs_simulated = []
        classes_real = []
        classes_simulated = []
        filenames_real = []
        filenames_simulated = []

        # Statistics tracking (enhanced for file limiting)
        stats = {
            "total_files_available": 0,
            "total_files_loaded": 0,
            "real_files_available": 0,
            "real_files_loaded": 0,
            "simulated_files_available": 0,
            "simulated_files_loaded": 0,
            "classes_count": {
                str(i): {
                    "real_available": 0,
                    "real_loaded": 0,
                    "simulated_available": 0,
                    "simulated_loaded": 0,
                    "total_available": 0,
                    "total_loaded": 0,
                }
                for i in range(10)
            },
            "empty_files": 0,
            "total_samples": 0,
            "file_tracking": {"real": [], "simulated": []},
            "max_files_per_class": max_files_per_class,
            "memory_optimization": True,
        }

        # Process each class folder
        selected = sorted(class_folders)
        real_events = self._open_events(selected, ["real"])
        other_events = self._open_events(selected, list(config.NON_REAL_EVENT_TYPES))
        real_groups = self._group_by_class(real_events)
        other_groups = self._group_by_class(other_events)

        for class_folder in tqdm(selected, desc="Processing classes"):
            real_idx = real_groups.get(class_folder, [])
            simulated_idx = other_groups.get(class_folder, [])
            n_available = len(real_idx) + len(simulated_idx)

            # Update available counts
            stats["classes_count"][class_folder]["real_available"] = len(real_idx)
            stats["classes_count"][class_folder]["simulated_available"] = len(
                simulated_idx
            )
            stats["classes_count"][class_folder]["total_available"] = n_available
            stats["total_files_available"] += n_available
            stats["real_files_available"] += len(real_idx)
            stats["simulated_files_available"] += len(simulated_idx)

            # Prioritize real files, then add simulated files up to the limit
            files_to_load = [
                (real_events, i, True) for i in real_idx[:max_files_per_class]
            ]

            remaining_slots = max_files_per_class - len(files_to_load)
            if remaining_slots > 0:
                files_to_load.extend(
                    (other_events, i, False) for i in simulated_idx[:remaining_slots]
                )

            # Load the selected files
            loaded_real_count = 0
            loaded_simulated_count = 0

            for source, idx, is_real_data in tqdm(
                files_to_load, desc=f"Class {class_folder}", leave=False
            ):
                filename = source.files_events[idx].name
                try:
                    df = self._clean_events(source[idx].signal, stats)

                    if df is None or len(df) == 0:
                        continue

                    stats["total_samples"] += len(df)
                    stats["total_files_loaded"] += 1

                    if is_real_data:
                        dfs_real.append(df)
                        classes_real.append(class_folder)
                        filenames_real.append(filename)
                        stats["real_files_loaded"] += 1
                        loaded_real_count += 1
                        stats["file_tracking"]["real"].append(filename)
                    else:
                        dfs_simulated.append(df)
                        classes_simulated.append(class_folder)
                        filenames_simulated.append(filename)
                        stats["simulated_files_loaded"] += 1
                        loaded_simulated_count += 1
                        stats["file_tracking"]["simulated"].append(filename)

                except Exception as e:
                    print(f"ERROR: Error loading {filename}: {str(e)}")
                    stats["empty_files"] += 1
                    continue

            # Update loaded counts for this class
            stats["classes_count"][class_folder]["real_loaded"] = loaded_real_count
            stats["classes_count"][class_folder]["simulated_loaded"] = (
                loaded_simulated_count
            )
            stats["classes_count"][class_folder]["total_loaded"] = (
                loaded_real_count + loaded_simulated_count
            )

        # Combine real and simulated data
        dfs_3w = dfs_real + dfs_simulated
        classes_3w = classes_real + classes_simulated

        # Add metadata about data sources
        stats["real_data"] = {
            "dfs": dfs_real,
            "classes": classes_real,
            "filenames": filenames_real,
        }
        stats["simulated_data"] = {
            "dfs": dfs_simulated,
            "classes": classes_simulated,
            "filenames": filenames_simulated,
        }

        self.stats = stats
        return dfs_3w, classes_3w, stats

    def get_separated_data(self) -> Dict[str, List[pd.DataFrame] | List[str]]:
        """
        Get real and non-real (simulated and hand-drawn) data separately.

        The provenances treated as non-real are set in `config.NON_REAL_EVENT_TYPES`.

        Returns:
            Dictionary containing separated real and simulated data
        """
        if not hasattr(self, "stats") or not self.stats:
            raise ValueError("No data loaded yet. Call load_dataset() first.")

        return {
            "real": self.stats["real_data"],
            "simulated": self.stats["simulated_data"],
        }

    def _clean_events(
        self, df: pd.DataFrame, stats: Dict[str, int | float | str]
    ) -> pd.DataFrame:
        """
        Clean the dataframe of a single event.

        Args:
            df (pd.DataFrame): Raw signals of the event (including the class column)
            stats (Dict): Statistics dictionary to update

        Returns:
            pd.DataFrame: Cleaned dataframe or None if the event is invalid
        """
        # Skip files without any data
        if len(df) == 0:
            stats["empty_files"] += 1
            return None

        # Check if we have class column
        if "class" not in df.columns:
            stats["empty_files"] += 1
            return None

        # Remove rows with missing class labels
        df = df.dropna(subset=["class"]).reset_index(drop=True)

        if len(df) == 0:
            stats["empty_files"] += 1
            return None

        # Apply data sampling if enabled in config
        if hasattr(config, "ENABLE_DATA_SAMPLING") and config.ENABLE_DATA_SAMPLING:
            original_len = len(df)
            if hasattr(config, "SAMPLING_RATE") and config.SAMPLING_RATE > 1:
                sampling_rate = config.SAMPLING_RATE
                if (
                    hasattr(config, "SAMPLING_METHOD")
                    and config.SAMPLING_METHOD == "random"
                ):
                    # Random sampling
                    sample_size = len(df) // sampling_rate
                    df = (
                        df.sample(n=sample_size, random_state=config.RANDOM_SEED)
                        .sort_index()
                        .reset_index(drop=True)
                    )
                else:
                    # Uniform sampling (default) - take every nth row
                    df = df.iloc[::sampling_rate].reset_index(drop=True)

        # Handle missing values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].ffill().bfill()

        # Remove rows with all NaN values in numeric columns
        df = df.dropna(subset=numeric_cols, how="all").reset_index(drop=True)

        if len(df) == 0:
            stats["empty_files"] += 1
            return None

        return df

    def filter_target_features(
        self, dfs: List[pd.DataFrame], classes: List[str], target_features: List[str]
    ) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        Filter dataframes to contain only target features.

        Args:
            dfs (List[pd.DataFrame]): List of dataframes to filter
            classes (List[str]): List of corresponding class labels
            target_features (List[str]): List of features to keep

        Returns:
            Tuple containing filtered dataframes and their classes
        """
        print("Filtering Data to Key Sensor Variables")
        print("=" * 60)
        print(f"Target features: {target_features}")

        filtered_dfs = []
        filtered_classes = []

        for i, (df, class_label) in enumerate(zip(dfs, classes)):
            # Check which target features are available
            available_features = [col for col in target_features if col in df.columns]

            if len(available_features) >= 2:  # Need at least 2 features
                filtered_df = df[available_features].copy()

                # Only keep if we have some data
                if len(filtered_df.dropna()) > 0:
                    filtered_dfs.append(filtered_df)
                    filtered_classes.append(class_label)

                    # Show info for first few dataframes
                    if len(filtered_dfs) <= 3:
                        print(
                            f"\n📄 DataFrame {len(filtered_dfs)} (Class {class_label}):"
                        )
                        print(f"  Available features: {available_features}")
                        print(f"  Shape after filtering: {filtered_df.shape}")
                        print("  Missing values per column:")
                        for col in filtered_df.columns:
                            missing_count = filtered_df[col].isnull().sum()
                            missing_pct = (missing_count / len(filtered_df)) * 100
                            print(f"    {col}: {missing_count} ({missing_pct:.1f}%)")

        print("\nFiltering Results:")
        print("=" * 40)
        print(f"Datasets after filtering: {len(filtered_dfs)}")
        print(f"Total samples: {sum(len(df) for df in filtered_dfs):,}")

        # Check feature availability
        if len(filtered_dfs) > 0:
            feature_availability = {}
            for feature in target_features:
                count = sum(1 for df in filtered_dfs if feature in df.columns)
                feature_availability[feature] = count

            print("\nFeature Availability Across Datasets:")
            print("-" * 40)
            for feature, count in feature_availability.items():
                percentage = (count / len(filtered_dfs)) * 100
                print(
                    f"{feature}: {count}/{len(filtered_dfs)} datasets ({percentage:.1f}%)"
                )

        return filtered_dfs, filtered_classes

    def get_dataset_info(
        self, dfs: List[pd.DataFrame], classes: List[str]
    ) -> Dict[str, int | float | str | List[str] | pd.DataFrame]:
        """
        Get comprehensive information about the loaded dataset.

        Args:
            dfs (List[pd.DataFrame]): List of dataframes
            classes (List[str]): List of corresponding class labels

        Returns:
            Dict containing dataset information
        """
        if not dfs:
            return {}

        info = {
            "total_dataframes": len(dfs),
            "class_distribution": dict(Counter(classes)),
            "sample_dataframe_shape": dfs[0].shape,
            "sample_columns": list(dfs[0].columns),
            "total_samples": sum(len(df) for df in dfs),
            "memory_usage_mb": sum(df.memory_usage(deep=True).sum() for df in dfs)
            / 1024**2,
        }

        return info
