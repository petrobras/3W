"""
Data Loading Module for 3W Dataset

This module provides functionality to load and organize the 3W dataset
containing sensor data for fault detection in oil wells.
"""

import os
from glob import glob
import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Dict, Any
from collections import Counter


class DataLoader:
    """
    A class for loading and organizing the 3W dataset.

    The 3W dataset contains sensor data from oil wells organized in folders
    representing different fault classes (0-9).
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

    def explore_structure(self) -> Dict[str, Any]:
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
        }

        for folder_name in sorted(os.listdir(self.dataset_path)):
            if folder_name not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                continue

            self.class_folders.append(folder_name)
            structure_info["classes"].append(folder_name)
            folder_path = os.path.join(self.dataset_path, folder_name)

            if os.path.isdir(folder_path):
                files = os.listdir(folder_path)
                parquet_files = [f for f in files if f.endswith(".parquet")]
                structure_info["total_files"] += len(parquet_files)

                print(f"Class {folder_name}: {len(parquet_files)} parquet files")

                # Show file types (real vs simulated)
                real_files = [f for f in parquet_files if "WELL" in f]
                simulated_files = [f for f in parquet_files if "WELL" not in f]

                structure_info["real_files"] += len(real_files)
                structure_info["simulated_files"] += len(simulated_files)

                print(f"  └── Real data: {len(real_files)} files")
                print(f"  └── Simulated data: {len(simulated_files)} files")
                print()

        print(f"Total classes found: {len(self.class_folders)}")
        print(f"Classes: {self.class_folders}")

        return structure_info

    def load_dataset(
        self, class_folders: List[str] = None, max_files_per_class: int = 150
    ) -> Tuple[List[pd.DataFrame], List[str], Dict[str, Any]]:
        """
        Load the 3W dataset from parquet files with memory optimization.

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
            "classes_count": {str(i): {"real_available": 0, "real_loaded": 0, 
                                      "simulated_available": 0, "simulated_loaded": 0, 
                                      "total_available": 0, "total_loaded": 0} for i in range(10)},
            "empty_files": 0,
            "total_samples": 0,
            "file_tracking": {"real": [], "simulated": []},
            "max_files_per_class": max_files_per_class,
            "memory_optimization": True,
        }

        print("Loading 3W Dataset with Memory Optimization...")
        print(f"Maximum files per class: {max_files_per_class}")
        print("=" * 50)

        # Process each class folder
        for class_folder in tqdm(sorted(class_folders), desc="Processing classes"):
            folder_path = os.path.join(self.dataset_path, class_folder)
            parquet_files = glob(os.path.join(folder_path, "*.parquet"))
            
            # Separate files into real and simulated first for prioritization
            real_files = []
            simulated_files = []
            
            for file_path in parquet_files:
                filename = os.path.basename(file_path)
                if self._is_real_data(filename):
                    real_files.append(file_path)
                else:
                    simulated_files.append(file_path)
            
            # Update available counts
            stats["classes_count"][class_folder]["real_available"] = len(real_files)
            stats["classes_count"][class_folder]["simulated_available"] = len(simulated_files)
            stats["classes_count"][class_folder]["total_available"] = len(parquet_files)
            stats["total_files_available"] += len(parquet_files)
            stats["real_files_available"] += len(real_files)
            stats["simulated_files_available"] += len(simulated_files)
            
            # Prioritize real files, then add simulated files up to the limit
            files_to_load = []
            files_to_load.extend(real_files[:max_files_per_class])  # Real files first
            
            remaining_slots = max_files_per_class - len(files_to_load)
            if remaining_slots > 0:
                files_to_load.extend(simulated_files[:remaining_slots])  # Fill with simulated
            
            # Load the selected files
            loaded_real_count = 0
            loaded_simulated_count = 0
            file_counter = 0
            
            for file_path in tqdm(files_to_load, desc=f"Class {class_folder}", leave=False):
                try:
                    df = self._load_and_clean_file(file_path, stats, file_counter)
                    file_counter += 1

                    if df is None or len(df) == 0:
                        continue

                    # Categorize as real or simulated data based on filename patterns
                    filename = os.path.basename(file_path)
                    is_real_data = self._is_real_data(filename)

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
                    print(f"ERROR: Error loading {file_path}: {str(e)}")
                    stats["empty_files"] += 1
                    continue
            
            # Update loaded counts for this class
            stats["classes_count"][class_folder]["real_loaded"] = loaded_real_count
            stats["classes_count"][class_folder]["simulated_loaded"] = loaded_simulated_count
            stats["classes_count"][class_folder]["total_loaded"] = loaded_real_count + loaded_simulated_count

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

        # Print loading summary
        self._print_loading_summary(stats, len(dfs_3w))

        self.stats = stats
        return dfs_3w, classes_3w, stats

    def _is_real_data(self, filename: str) -> bool:
        """
        Determine if a file contains real or simulated data based on filename patterns.

        Args:
            filename (str): Name of the file

        Returns:
            bool: True if real data, False if simulated
        """
        # Based on 3W dataset conventions:
        # Real data: Contains 'WELL' in filename
        # Simulated data: Starts with 'SIMULATED' or 'OLGA'
        # Hand-dcompleten data: Starts with 'DRAWN' or 'DESENHADA'

        filename_upper = filename.upper()

        # Check for simulated patterns first
        if (
            filename_upper.startswith("SIMULATED")
            or filename_upper.startswith("OLGA")
            or filename_upper.startswith("DRAWN")
            or filename_upper.startswith("DESENHADA")
        ):
            return False

        # Check for real data patterns
        if "WELL" in filename_upper:
            return True

        # Default assumption for other patterns (could be refined)
        # If it doesn't match simulated patterns and contains well-like patterns, consider real
        return True

    def get_separated_data(self) -> Dict[str, Any]:
        """
        Get real and simulated data separately.

        Returns:
            Dictionary containing separated real and simulated data
        """
        if not hasattr(self, "stats") or not self.stats:
            raise ValueError("No data loaded yet. Call load_dataset() first.")

        return {
            "real": self.stats["real_data"],
            "simulated": self.stats["simulated_data"],
        }

    def _load_and_clean_file(
        self, file_path: str, stats: Dict[str, Any], file_counter: int = 0
    ) -> pd.DataFrame:
        """
        Load and clean a single parquet file.

        Args:
            file_path (str): Path to the parquet file
            stats (Dict): Statistics dictionary to update
            file_counter (int): Current file number for debugging

        Returns:
            pd.DataFrame: Cleaned dataframe or None if file is invalid
        """
        from . import config

        # Load data
        df = pd.read_parquet(file_path)

        # Debug information for first few files
        if file_counter < 3:
            print(f"Debug - File: {os.path.basename(file_path)}")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Has 'class' column: {'class' in df.columns}")

        # Skip files without any data
        if len(df) == 0:
            stats["empty_files"] += 1
            return None

        # Check if we have class column
        if "class" not in df.columns:
            stats["empty_files"] += 1
            if file_counter < 3:
                print(f"  Skipping - no 'class' column")
            return None

        # Remove rows with missing class labels
        df = df.dropna(subset=["class"]).reset_index(drop=True)

        if len(df) == 0:
            stats["empty_files"] += 1
            if file_counter < 3:
                print(f"  Skipping - all class values are NaN")
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

                if file_counter < 3:
                    print(
                        f"  Sampling: {original_len} → {len(df)} samples (1/{sampling_rate})"
                    )

        # Handle missing values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].ffill().bfill()

        # Remove rows with all NaN values in numeric columns
        df = df.dropna(subset=numeric_cols, how="all").reset_index(drop=True)

        if len(df) == 0:
            stats["empty_files"] += 1
            if file_counter < 3:
                print(f"  Skipping - no valid data after cleaning")
            return None

        if file_counter < 3:
            print(f"  Successfully loaded: {len(df)} samples")

        return df

    def _print_loading_summary(self, stats: Dict[str, Any], total_loaded: int) -> None:
        """Print loading summary statistics with memory optimization details."""
        print("\nLoading Summary:")
        print("=" * 50)
        
        # Check if this is the new optimized format
        if "memory_optimization" in stats and stats["memory_optimization"]:
            print(f"Total files available: {stats['total_files_available']}")
            print(f"Total files loaded: {stats['total_files_loaded']}")
            print(f"Real data - Available: {stats['real_files_available']}, Loaded: {stats['real_files_loaded']}")
            print(f"Simulated data - Available: {stats['simulated_files_available']}, Loaded: {stats['simulated_files_loaded']}")
            print(f"Empty/invalid files: {stats['empty_files']}")
            print(f"Total samples: {stats['total_samples']:,}")
            
            # Print per-class breakdown
            print(f"\nPer-Class Breakdown:")
            for class_id in sorted(stats['classes_count'].keys()):
                class_data = stats['classes_count'][class_id]
                if class_data['total_available'] > 0:
                    print(f"  Class {class_id}: {class_data['total_loaded']}/{class_data['total_available']} files "
                          f"(R: {class_data['real_loaded']}/{class_data['real_available']}, "
                          f"S: {class_data['simulated_loaded']}/{class_data['simulated_available']})")
        else:
            # Legacy format for backward compatibility
            print(f"Total files processed: {stats.get('total_files', 0)}")
            print(f"Successfully loaded: {total_loaded} files")
            print(f"Real data files: {stats.get('real_files', 0)}")
            print(f"Simulated data files: {stats.get('simulated_files', 0)}")
            print(f"Empty/invalid files: {stats.get('empty_files', 0)}")
            print(f"Total samples: {stats.get('total_samples', 0):,}")

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
                        print(f"  Missing values per column:")
                        for col in filtered_df.columns:
                            missing_count = filtered_df[col].isnull().sum()
                            missing_pct = (missing_count / len(filtered_df)) * 100
                            print(f"    {col}: {missing_count} ({missing_pct:.1f}%)")

        print(f"\nFiltering Results:")
        print("=" * 40)
        print(f"Datasets after filtering: {len(filtered_dfs)}")
        print(f"Total samples: {sum(len(df) for df in filtered_dfs):,}")

        # Check feature availability
        if len(filtered_dfs) > 0:
            feature_availability = {}
            for feature in target_features:
                count = sum(1 for df in filtered_dfs if feature in df.columns)
                feature_availability[feature] = count

            print(f"\nFeature Availability Across Datasets:")
            print("-" * 40)
            for feature, count in feature_availability.items():
                percentage = (count / len(filtered_dfs)) * 100
                print(
                    f"{feature}: {count}/{len(filtered_dfs)} datasets ({percentage:.1f}%)"
                )

        return filtered_dfs, filtered_classes

    def get_dataset_info(
        self, dfs: List[pd.DataFrame], classes: List[str]
    ) -> Dict[str, Any]:
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
