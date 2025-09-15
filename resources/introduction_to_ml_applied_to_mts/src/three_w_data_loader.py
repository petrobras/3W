"""
3W Dataset Comprehensive Data Loader

This module provides utilities for loading windowed time series data from all available
folds for both training and testing. It supports flexible configuration, memory-efficient
loading, and comprehensive data validation.

Author: 3W Team
Date: September 2025
"""

import time
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union


class ThreeWDataLoader:
    """
    Comprehensive data loader for the 3W oil well dataset.

    Supports loading windowed time series data from all available folds
    with flexible configuration options and comprehensive validation.
    """

    def __init__(self, windowed_base_dir: str, persistence, config):
        """
        Initialize the data loader.

        Args:
            windowed_base_dir: Path to the windowed data directory
            persistence: DataPersistence instance for loading data
            config: Configuration module with SAVE_FORMAT
        """
        self.windowed_base_dir = windowed_base_dir
        self.persistence = persistence
        self.config = config
        self.all_fold_data = {}
        self.loading_stats = {
            "total_train_windows": 0,
            "total_test_windows": 0,
            "successful_folds": 0,
            "failed_folds": 0,
            "total_load_time": 0,
            "all_classes": set(),
        }

    def discover_folds(self) -> List[str]:
        """
        Discover all available fold directories.

        Returns:
            List of fold directory names
        """
        if not os.path.exists(self.windowed_base_dir):
            raise FileNotFoundError(
                f"Windowed directory not found: {self.windowed_base_dir}"
            )

        fold_dirs = []
        for item in os.listdir(self.windowed_base_dir):
            item_path = os.path.join(self.windowed_base_dir, item)
            if os.path.isdir(item_path) and item.startswith("fold_"):
                fold_dirs.append(item)

        fold_dirs.sort()  # Ensure consistent ordering
        return fold_dirs

    def load_fold_data(
        self,
        fold_dir: str,
        load_train: bool = True,
        load_test: bool = True,
        max_windows: Optional[int] = None,
    ) -> Dict:
        """
        Load windowed data from a specific fold directory.

        Args:
            fold_dir: Name of the fold directory
            load_train: Whether to load training data
            load_test: Whether to load testing data
            max_windows: Maximum number of windows per data type (None = load all)

        Returns:
            Dictionary containing fold data and statistics
        """
        fold_path = os.path.join(self.windowed_base_dir, fold_dir)
        fold_number = fold_dir.replace("fold_", "")

        result = {
            "fold_name": fold_dir,
            "fold_number": int(fold_number),
            "train_data": None,
            "test_data": None,
            "metadata": None,
            "stats": {
                "train_windows": 0,
                "test_windows": 0,
                "load_time": 0,
                "train_classes": [],
                "test_classes": [],
            },
        }

        load_start = time.time()

        # Load training data
        if load_train:
            result["train_data"] = self._load_data_type(fold_path, "train", max_windows)
            if result["train_data"] is not None:
                train_dfs, train_classes = result["train_data"]
                result["stats"]["train_windows"] = len(train_dfs)
                result["stats"]["train_classes"] = list(np.unique(train_classes))

        # Load testing data
        if load_test:
            result["test_data"] = self._load_data_type(fold_path, "test", max_windows)
            if result["test_data"] is not None:
                test_dfs, test_classes = result["test_data"]
                result["stats"]["test_windows"] = len(test_dfs)
                result["stats"]["test_classes"] = list(np.unique(test_classes))

        # Load metadata if available
        metadata_file = os.path.join(fold_path, "windowing_metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as f:
                    result["metadata"] = json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading metadata from {metadata_file}: {e}")

        result["stats"]["load_time"] = time.time() - load_start
        return result

    def _load_data_type(
        self, fold_path: str, data_type: str, max_windows: Optional[int] = None
    ) -> Optional[Tuple[List, List]]:
        """
        Load data of a specific type (train/test) from a fold.

        Args:
            fold_path: Path to the fold directory
            data_type: 'train' or 'test'
            max_windows: Maximum number of windows to load

        Returns:
            Tuple of (dataframes, classes) or None if not found
        """
        # Try pickle format first
        pickle_file = os.path.join(
            fold_path, f"{data_type}_windowed.{self.config.SAVE_FORMAT}"
        )
        parquet_file = os.path.join(fold_path, f"{data_type}_windowed.parquet")

        data_dfs, data_classes = None, None

        if os.path.exists(pickle_file):
            try:
                data_dfs, data_classes = self.persistence._load_dataframes(
                    pickle_file, self.config.SAVE_FORMAT
                )
            except Exception as e:
                print(f"⚠️ Error loading {data_type} data from {pickle_file}: {e}")

        elif os.path.exists(parquet_file):
            try:
                data_dfs, data_classes = self.persistence._load_from_parquet(
                    parquet_file
                )
            except Exception as e:
                print(f"⚠️ Error loading {data_type} data from {parquet_file}: {e}")

        # Apply window limit if specified
        if data_dfs is not None and max_windows and len(data_dfs) > max_windows:
            indices = np.random.choice(len(data_dfs), max_windows, replace=False)
            data_dfs = [data_dfs[i] for i in indices]
            data_classes = [data_classes[i] for i in indices]

        return (data_dfs, data_classes) if data_dfs is not None else None

    def load_all_folds(
        self,
        load_all_folds: bool = True,
        specific_folds: List[int] = None,
        load_train_data: bool = True,
        load_test_data: bool = True,
        max_windows_per_fold: Optional[int] = None,
    ) -> Dict:
        """
        Load data from all or specific folds.

        Args:
            load_all_folds: Whether to load all available folds
            specific_folds: List of specific fold numbers to load (if load_all_folds=False)
            load_train_data: Whether to load training data
            load_test_data: Whether to load testing data
            max_windows_per_fold: Maximum windows per fold per data type

        Returns:
            Dictionary containing loading statistics
        """
        # Discover available folds
        fold_dirs = self.discover_folds()

        # Determine which folds to process
        if load_all_folds:
            folds_to_process = fold_dirs
        else:
            specific_folds = specific_folds or []
            folds_to_process = [
                f"fold_{i}" for i in specific_folds if f"fold_{i}" in fold_dirs
            ]

        print(
            f"🎯 Processing {len(folds_to_process)} folds: {', '.join(folds_to_process)}"
        )

        # Reset statistics
        self.loading_stats = {
            "total_train_windows": 0,
            "total_test_windows": 0,
            "successful_folds": 0,
            "failed_folds": 0,
            "total_load_time": 0,
            "all_classes": set(),
        }

        # Process each fold
        for fold_dir in folds_to_process:
            print(f"\n📁 Processing {fold_dir}...")

            try:
                fold_data = self.load_fold_data(
                    fold_dir=fold_dir,
                    load_train=load_train_data,
                    load_test=load_test_data,
                    max_windows=max_windows_per_fold,
                )

                self.all_fold_data[fold_dir] = fold_data

                # Update statistics
                stats = fold_data["stats"]
                self.loading_stats["total_train_windows"] += stats["train_windows"]
                self.loading_stats["total_test_windows"] += stats["test_windows"]
                self.loading_stats["total_load_time"] += stats["load_time"]
                self.loading_stats["successful_folds"] += 1
                self.loading_stats["all_classes"].update(stats["train_classes"])
                self.loading_stats["all_classes"].update(stats["test_classes"])

                # Print fold summary
                print(f"   ✅ {fold_dir} loaded successfully")
                print(f"      🚄 Train windows: {stats['train_windows']}")
                print(f"      🧪 Test windows: {stats['test_windows']}")
                print(f"      ⏱️ Load time: {stats['load_time']:.2f}s")

                if stats["train_classes"]:
                    print(f"      🏷️ Train classes: {sorted(stats['train_classes'])}")
                if stats["test_classes"]:
                    print(f"      🏷️ Test classes: {sorted(stats['test_classes'])}")

            except Exception as e:
                print(f"   ❌ Failed to load {fold_dir}: {e}")
                self.loading_stats["failed_folds"] += 1
                continue

        return self.loading_stats

    def get_fold_data(
        self, fold_name: str, data_type: str = "both"
    ) -> Optional[Union[Dict, Tuple]]:
        """
        Get data from a specific fold.

        Args:
            fold_name: Name of fold (e.g., 'fold_0')
            data_type: 'train', 'test', or 'both'

        Returns:
            Requested data or None if not found
        """
        if fold_name not in self.all_fold_data:
            print(f"❌ Fold '{fold_name}' not found in loaded data")
            return None

        fold_data = self.all_fold_data[fold_name]

        if data_type == "train":
            return fold_data["train_data"]
        elif data_type == "test":
            return fold_data["test_data"]
        elif data_type == "both":
            return {
                "train": fold_data["train_data"],
                "test": fold_data["test_data"],
                "metadata": fold_data["metadata"],
                "stats": fold_data["stats"],
            }
        else:
            print(f"❌ Invalid data_type: {data_type}. Use 'train', 'test', or 'both'")
            return None

    def get_all_train_data(self) -> Tuple[List, List]:
        """
        Combine training data from all folds.

        Returns:
            Tuple of (all_train_dfs, all_train_classes)
        """
        all_train_dfs = []
        all_train_classes = []

        for fold_name, fold_data in self.all_fold_data.items():
            if fold_data["train_data"] is not None:
                train_dfs, train_classes = fold_data["train_data"]
                all_train_dfs.extend(train_dfs)
                all_train_classes.extend(train_classes)

        return all_train_dfs, all_train_classes

    def get_all_test_data(self) -> Tuple[List, List]:
        """
        Combine testing data from all folds.

        Returns:
            Tuple of (all_test_dfs, all_test_classes)
        """
        all_test_dfs = []
        all_test_classes = []

        for fold_name, fold_data in self.all_fold_data.items():
            if fold_data["test_data"] is not None:
                test_dfs, test_classes = fold_data["test_data"]
                all_test_dfs.extend(test_dfs)
                all_test_classes.extend(test_classes)

        return all_test_dfs, all_test_classes

    def get_class_distribution(
        self, data_type: str = "both", by_fold: bool = False
    ) -> Dict:
        """
        Get class distribution statistics.

        Args:
            data_type: 'train', 'test', or 'both'
            by_fold: If True, show distribution per fold

        Returns:
            Class distribution statistics
        """
        if by_fold:
            distribution = {}

            for fold_name, fold_data in self.all_fold_data.items():
                fold_dist = {}

                if (
                    data_type in ["train", "both"]
                    and fold_data["train_data"] is not None
                ):
                    _, train_classes = fold_data["train_data"]
                    unique, counts = np.unique(train_classes, return_counts=True)
                    fold_dist["train"] = dict(zip(unique, counts))

                if data_type in ["test", "both"] and fold_data["test_data"] is not None:
                    _, test_classes = fold_data["test_data"]
                    unique, counts = np.unique(test_classes, return_counts=True)
                    fold_dist["test"] = dict(zip(unique, counts))

                distribution[fold_name] = fold_dist

            return distribution

        else:
            # Overall distribution
            all_classes = []

            if data_type in ["train", "both"]:
                _, train_classes = self.get_all_train_data()
                all_classes.extend(train_classes)

            if data_type in ["test", "both"]:
                _, test_classes = self.get_all_test_data()
                all_classes.extend(test_classes)

            unique, counts = np.unique(all_classes, return_counts=True)
            return dict(zip(unique, counts))

    def list_available_folds(self) -> List[str]:
        """
        List all successfully loaded folds.

        Returns:
            List of available fold names
        """
        return list(self.all_fold_data.keys())

    def validate_data_consistency(self) -> Dict[str, bool]:
        """
        Validate that all loaded data has consistent shapes and columns.

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "shapes_consistent": True,
            "columns_consistent": True,
            "reference_shape": None,
            "reference_columns": None,
            "inconsistencies": [],
        }

        reference_shape = None
        reference_columns = None

        for fold_name, fold_data in self.all_fold_data.items():
            # Check training data
            if fold_data["train_data"] is not None:
                train_dfs, _ = fold_data["train_data"]
                if train_dfs:
                    sample_shape = train_dfs[0].shape
                    sample_columns = list(train_dfs[0].columns)

                    if reference_shape is None:
                        reference_shape = sample_shape
                        reference_columns = sample_columns
                        validation_results["reference_shape"] = reference_shape
                        validation_results["reference_columns"] = len(reference_columns)
                    else:
                        if sample_shape != reference_shape:
                            validation_results["shapes_consistent"] = False
                            validation_results["inconsistencies"].append(
                                f"Shape mismatch in {fold_name} train: {sample_shape} vs {reference_shape}"
                            )
                        if sample_columns != reference_columns:
                            validation_results["columns_consistent"] = False
                            validation_results["inconsistencies"].append(
                                f"Column mismatch in {fold_name} train"
                            )

            # Check testing data
            if fold_data["test_data"] is not None:
                test_dfs, _ = fold_data["test_data"]
                if test_dfs:
                    sample_shape = test_dfs[0].shape
                    sample_columns = list(test_dfs[0].columns)

                    if reference_shape is not None:
                        if sample_shape != reference_shape:
                            validation_results["shapes_consistent"] = False
                            validation_results["inconsistencies"].append(
                                f"Shape mismatch in {fold_name} test: {sample_shape} vs {reference_shape}"
                            )
                        if sample_columns != reference_columns:
                            validation_results["columns_consistent"] = False
                            validation_results["inconsistencies"].append(
                                f"Column mismatch in {fold_name} test"
                            )

        return validation_results

    def estimate_memory_usage(self) -> Dict[str, Union[int, float]]:
        """
        Estimate memory usage of the loaded dataset.

        Returns:
            Dictionary with memory usage statistics
        """
        validation_results = self.validate_data_consistency()
        reference_shape = validation_results["reference_shape"]

        if reference_shape is None:
            return {"error": "No data loaded for memory estimation"}

        total_windows = (
            self.loading_stats["total_train_windows"]
            + self.loading_stats["total_test_windows"]
        )

        # Estimate memory per window (rough estimate)
        elements_per_window = reference_shape[0] * reference_shape[1]
        bytes_per_element = 8  # Assuming float64
        bytes_per_window = elements_per_window * bytes_per_element
        total_memory_bytes = total_windows * bytes_per_window

        return {
            "window_dimensions": reference_shape,
            "elements_per_window": elements_per_window,
            "total_windows": total_windows,
            "bytes_per_window": bytes_per_window,
            "total_memory_bytes": total_memory_bytes,
            "total_memory_mb": total_memory_bytes / (1024 * 1024),
            "total_memory_gb": total_memory_bytes / (1024 * 1024 * 1024),
        }

    def print_summary(self):
        """Print a comprehensive summary of the loaded dataset."""
        print("\n" + "=" * 60)
        print("📊 3W DATASET COMPREHENSIVE LOADING SUMMARY")
        print("=" * 60)

        # Basic statistics
        print(
            f"✅ Successfully processed folds: {self.loading_stats['successful_folds']}"
        )
        print(f"❌ Failed folds: {self.loading_stats['failed_folds']}")
        print(
            f"🚄 Total training windows: {self.loading_stats['total_train_windows']:,}"
        )
        print(f"🧪 Total testing windows: {self.loading_stats['total_test_windows']:,}")
        print(
            f"🎯 Total windows: {self.loading_stats['total_train_windows'] + self.loading_stats['total_test_windows']:,}"
        )
        print(f"⏱️ Total loading time: {self.loading_stats['total_load_time']:.2f}s")
        print(f"🏷️ All classes found: {sorted(list(self.loading_stats['all_classes']))}")

        # Performance metrics
        if self.loading_stats["successful_folds"] > 0:
            avg_load_time = (
                self.loading_stats["total_load_time"]
                / self.loading_stats["successful_folds"]
            )
            print(f"\n⚡ Performance Summary:")
            print(f"   • Average fold load time: {avg_load_time:.2f}s")

            if self.loading_stats["total_load_time"] > 0:
                windows_per_sec = (
                    self.loading_stats["total_train_windows"]
                    + self.loading_stats["total_test_windows"]
                ) / self.loading_stats["total_load_time"]
                print(f"   • Windows loaded per second: {windows_per_sec:.0f}")

        # Memory estimation
        memory_stats = self.estimate_memory_usage()
        if "error" not in memory_stats:
            print(f"\n💾 Memory Usage Estimation:")
            print(f"   • Window dimensions: {memory_stats['window_dimensions']}")
            print(f"   • Elements per window: {memory_stats['elements_per_window']:,}")
            print(f"   • Total windows: {memory_stats['total_windows']:,}")
            print(
                f"   • Estimated memory usage: {memory_stats['total_memory_mb']:.1f} MB"
            )

            if memory_stats["total_memory_mb"] > 1024:
                print(f"   • Large dataset: {memory_stats['total_memory_gb']:.1f} GB")


# Convenience functions for backward compatibility with notebook usage
def create_data_loader(windowed_base_dir: str, persistence, config) -> ThreeWDataLoader:
    """
    Create a ThreeWDataLoader instance.

    Args:
        windowed_base_dir: Path to the windowed data directory
        persistence: DataPersistence instance
        config: Configuration module

    Returns:
        ThreeWDataLoader instance
    """
    return ThreeWDataLoader(windowed_base_dir, persistence, config)


def load_complete_dataset(
    windowed_base_dir: str,
    persistence,
    config,
    load_all_folds: bool = True,
    specific_folds: List[int] = None,
    load_train_data: bool = True,
    load_test_data: bool = True,
    max_windows_per_fold: Optional[int] = None,
) -> ThreeWDataLoader:
    """
    Convenience function to load the complete dataset with default settings.

    Args:
        windowed_base_dir: Path to the windowed data directory
        persistence: DataPersistence instance
        config: Configuration module
        load_all_folds: Whether to load all available folds
        specific_folds: List of specific fold numbers to load
        load_train_data: Whether to load training data
        load_test_data: Whether to load testing data
        max_windows_per_fold: Maximum windows per fold per data type

    Returns:
        ThreeWDataLoader instance with data loaded
    """
    loader = ThreeWDataLoader(windowed_base_dir, persistence, config)
    loader.load_all_folds(
        load_all_folds=load_all_folds,
        specific_folds=specific_folds,
        load_train_data=load_train_data,
        load_test_data=load_test_data,
        max_windows_per_fold=max_windows_per_fold,
    )
    return loader
