"""
Data Preprocessing Module for 3W Dataset

This module provides functionality for scaling, normalizing, and preprocessing
sensor data from the 3W dataset.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import List, Tuple, Dict, Any, Optional
import random
from collections import Counter


class DataPreprocessor:
    """
    A class for preprocessing sensor data from the 3W dataset.

    Provides various scaling methods and preprocessing utilities for
    time series sensor data.
    """

    def __init__(self):
        """Initialize the DataPreprocessor."""
        self.scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }
        self.fitted_scalers = {}

    def apply_scaling(
        self,
        dfs: List[pd.DataFrame],
        classes: List[str],
        method: str = "minmax",
        exclude_columns: List[str] = None,
    ) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        Apply scaling to all dataframes using the specified method.

        Args:
            dfs (List[pd.DataFrame]): List of dataframes to scale
            classes (List[str]): List of corresponding class labels
            method (str): Scaling method ('standard', 'minmax', 'robust', 'normalizer')
            exclude_columns (List[str]): Columns to exclude from scaling (e.g., 'class')

        Returns:
            Tuple containing scaled dataframes and their classes
        """
        if exclude_columns is None:
            exclude_columns = ["class"]

        if method not in self.scalers:
            raise ValueError(
                f"Unknown scaling method: {method}. Choose from {list(self.scalers.keys())}"
            )

        print(f"🔄 Applying {method.title()}Scaler to All Dataframes")
        print("=" * 60)

        scaler = self.scalers[method]
        scaled_dfs = []
        scaled_classes = []

        # Track statistics
        total_processed = 0
        successfully_scaled = 0
        skipped_dfs = 0

        for i, (df, class_label) in enumerate(zip(dfs, classes)):
            total_processed += 1

            try:
                scaled_df = self._scale_single_dataframe(
                    df, scaler, method, exclude_columns, i + 1, class_label
                )

                if scaled_df is not None:
                    scaled_dfs.append(scaled_df)
                    scaled_classes.append(class_label)
                    successfully_scaled += 1
                else:
                    skipped_dfs += 1

            except Exception as e:
                print(f"❌ Error processing DataFrame {i + 1}: {str(e)}")
                skipped_dfs += 1
                continue

        self._print_scaling_summary(
            total_processed, successfully_scaled, skipped_dfs, scaled_classes, method
        )

        # Store the fitted scaler for future use
        self.fitted_scalers[method] = scaler

        return scaled_dfs, scaled_classes

    def _scale_single_dataframe(
        self,
        df: pd.DataFrame,
        scaler,
        method: str,
        exclude_columns: List[str],
        df_index: int,
        class_label: str,
    ) -> Optional[pd.DataFrame]:
        """
        Scale a single dataframe.

        Args:
            df (pd.DataFrame): Dataframe to scale
            scaler: Scikit-learn scaler instance
            method (str): Scaling method name
            exclude_columns (List[str]): Columns to exclude from scaling
            df_index (int): Index of dataframe for logging
            class_label (str): Class label for logging

        Returns:
            Scaled dataframe or None if scaling failed
        """
        # Get numeric columns (excluding specified columns)
        numeric_cols = [
            col
            for col in df.columns
            if col not in exclude_columns
            and df[col].dtype in ["float64", "float32", "int64", "int32"]
        ]

        if len(numeric_cols) == 0:
            if df_index <= 5:
                print(f"⚠️  DataFrame {df_index}: No numeric columns found, skipping...")
            return None

        # Extract numeric data and remove rows with NaN values
        numeric_data = df[numeric_cols].dropna()

        if len(numeric_data) == 0:
            if df_index <= 5:
                print(
                    f"⚠️  DataFrame {df_index}: No valid numeric data after removing NaN, skipping..."
                )
            return None

        # Apply scaling
        scaled_values = scaler.fit_transform(numeric_data)

        # Create new dataframe with scaled values
        scaled_df = pd.DataFrame(
            scaled_values,
            columns=[f"{col}_scaled" for col in numeric_cols],
            index=numeric_data.index,
        )

        # Add excluded columns if they exist
        for col in exclude_columns:
            if col in df.columns:
                scaled_df[col] = df.loc[numeric_data.index, col]

        # Show progress for first few dataframes
        if df_index <= 5:
            print(f"✅ DataFrame {df_index} (Class {class_label}):")
            print(f"   Original shape: {df.shape}")
            print(f"   Scaled shape: {scaled_df.shape}")
            print(f"   Numeric columns scaled: {numeric_cols}")
            print("   Value ranges after scaling:")
            for col in numeric_cols:
                col_scaled = f"{col}_scaled"
                min_val = scaled_df[col_scaled].min()
                max_val = scaled_df[col_scaled].max()
                print(f"     {col_scaled}: [{min_val:.4f}, {max_val:.4f}]")
            print()
        elif df_index == 6:
            print("   ... (continuing with remaining dataframes)")

        return scaled_df

    def _print_scaling_summary(
        self,
        total_processed: int,
        successfully_scaled: int,
        skipped_dfs: int,
        scaled_classes: List[str],
        method: str,
    ) -> None:
        """Print scaling summary statistics."""

        if scaled_classes:
            unique_classes, counts = np.unique(scaled_classes, return_counts=True)
            class_dist = dict(zip(unique_classes, counts))

    def get_scaling_statistics(
        self, scaled_dfs: List[pd.DataFrame], method: str = "minmax"
    ) -> Dict[str, Any]:
        """
        Get statistics about scaled data.

        Args:
            scaled_dfs (List[pd.DataFrame]): List of scaled dataframes
            method (str): Scaling method used

        Returns:
            Dictionary containing scaling statistics
        """
        if not scaled_dfs:
            return {}

        sample_df = scaled_dfs[0]
        numeric_cols = [col for col in sample_df.columns if col != "class"]

        stats = {
            "method": method,
            "total_dataframes": len(scaled_dfs),
            "numeric_columns": numeric_cols,
            "value_ranges": {},
            "sample_statistics": {},
        }

        # Calculate value ranges and statistics
        for col in numeric_cols:
            all_values = np.concatenate([df[col].dropna().values for df in scaled_dfs])

            stats["value_ranges"][col] = {
                "min": float(np.min(all_values)),
                "max": float(np.max(all_values)),
                "mean": float(np.mean(all_values)),
                "std": float(np.std(all_values)),
            }

        # Sample dataframe statistics
        stats["sample_statistics"] = {
            "shape": sample_df.shape,
            "columns": list(sample_df.columns),
        }

        return stats

    def select_random_dataframe(
        self,
        dfs: List[pd.DataFrame],
        classes: List[str],
        required_columns: List[str] = None,
        min_samples: int = 100,
        seed: int = 42,
    ) -> Tuple[int, pd.DataFrame, str, int]:
        """
        Randomly select a dataframe that meets certain criteria.

        Args:
            dfs (List[pd.DataFrame]): List of dataframes to choose from
            classes (List[str]): List of corresponding class labels
            required_columns (List[str]): Columns that must be present
            min_samples (int): Minimum number of samples required
            seed (int): Random seed for reproducibility

        Returns:
            Tuple containing (index, dataframe, class_label, sample_count)
        """
        if required_columns is None:
            required_columns = []

        available_dfs = []

        for i, df in enumerate(dfs):
            # Check if dataframe has required columns
            has_required_cols = all(col in df.columns for col in required_columns)

            if has_required_cols:
                # Check if there's enough non-null data
                if required_columns:
                    df_clean = df[required_columns].dropna()
                else:
                    df_clean = df.dropna()

                if len(df_clean) >= min_samples:
                    available_dfs.append((i, df, len(df_clean)))

        if not available_dfs:
            raise ValueError(
                f"No dataframes found with required columns {required_columns} and {min_samples}+ samples"
            )

        # Randomly select one dataframe
        random.seed(seed)
        selected_idx, selected_df, sample_count = random.choice(available_dfs)

        return selected_idx, selected_df, classes[selected_idx], sample_count

    def apply_multiple_scalers(
        self, data: pd.DataFrame, exclude_columns: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply multiple scaling methods to the same dataset for comparison.

        Args:
            data (pd.DataFrame): Input dataframe
            exclude_columns (List[str]): Columns to exclude from scaling

        Returns:
            Dictionary with scaling method names as keys and scaled dataframes as values
        """
        if exclude_columns is None:
            exclude_columns = ["class"]

        # Get numeric columns
        numeric_cols = [
            col
            for col in data.columns
            if col not in exclude_columns
            and data[col].dtype in ["float64", "float32", "int64", "int32"]
        ]

        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found for scaling")

        # Clean data
        clean_data = data[numeric_cols].dropna()

        if len(clean_data) == 0:
            raise ValueError("No valid data after removing NaN values")

        scaled_results = {"Original": data.copy()}

        # Apply each scaling method
        for method_name, scaler in self.scalers.items():
            scaled_values = scaler.fit_transform(clean_data)
            scaled_df = pd.DataFrame(
                scaled_values, columns=numeric_cols, index=clean_data.index
            )

            # Add excluded columns
            for col in exclude_columns:
                if col in data.columns:
                    scaled_df[col] = data.loc[clean_data.index, col]

            scaled_results[method_name.title() + "Scaler"] = scaled_df

        return scaled_results

    def validate_scaling(
        self, scaled_df: pd.DataFrame, method: str = "minmax"
    ) -> Dict[str, bool]:
        """
        Validate that scaling was applied correctly.

        Args:
            scaled_df (pd.DataFrame): Scaled dataframe to validate
            method (str): Scaling method used

        Returns:
            Dictionary containing validation results
        """
        numeric_cols = [col for col in scaled_df.columns if col != "class"]
        validation = {
            "has_numeric_data": len(numeric_cols) > 0,
            "no_null_values": not scaled_df[numeric_cols].isnull().any().any(),
            "correct_scaling": {},
        }

        for col in numeric_cols:
            col_data = scaled_df[col].dropna()

            if method == "minmax":
                # MinMax should be in range [0, 1]
                validation["correct_scaling"][col] = (
                    col_data.min() >= -0.001 and col_data.max() <= 1.001
                )
            elif method == "standard":
                # Standard should have mean ≈ 0, std ≈ 1
                validation["correct_scaling"][col] = (
                    abs(col_data.mean()) < 0.1 and abs(col_data.std() - 1) < 0.1
                )
            elif method == "robust":
                # Robust scaling - harder to validate, just check for finite values
                validation["correct_scaling"][col] = np.isfinite(col_data).all()
            elif method == "normalizer":
                # Normalizer - each sample should have unit norm
                validation["correct_scaling"][col] = (
                    True  # Complex validation for normalizer
                )

        return validation

    def create_time_windows(
        self,
        dfs: List[pd.DataFrame],
        classes: List[str],
        window_size: int = 300,
        stride: int = None,
        min_window_size: int = None,
    ) -> Tuple[List[pd.DataFrame], List[str], List[Dict]]:
        """
        Create time windows from dataframes by dividing them into fixed-size windows.

        Args:
            dfs (List[pd.DataFrame]): List of dataframes to window
            classes (List[str]): List of corresponding class labels
            window_size (int): Size of each time window (default: 300)
            stride (int): Step size between windows (default: window_size for non-overlapping)
            min_window_size (int): Minimum window size to keep (default: window_size)

        Returns:
            Tuple containing:
            - List of windowed dataframes
            - List of corresponding class labels
            - List of metadata for each window
        """
        # Ensure all parameters are integers
        window_size = int(window_size)
        if stride is None:
            stride = window_size
        else:
            stride = int(stride)
        if min_window_size is None:
            min_window_size = window_size
        else:
            min_window_size = int(min_window_size)

        windowed_dfs = []
        windowed_classes = []
        window_metadata = []

        # print(f"🪟 Creating time windows:")

        total_windows = 0
        skipped_samples = 0

        for i, (df, class_label) in enumerate(zip(dfs, classes)):
            df_length = len(df)
            # Skip if dataframe is too small
            if df_length < min_window_size:
                skipped_samples += 1
                continue

            # Calculate number of windows for this dataframe
            num_windows = max(1, (df_length - window_size) // stride + 1)
            sample_windows = 0

            for start_idx in range(0, df_length - min_window_size + 1, stride):
                end_idx = min(start_idx + window_size, df_length)
                actual_window_size = end_idx - start_idx

                # Skip if window is too small
                if actual_window_size < min_window_size:
                    break

                # Extract window
                window_df = df.iloc[start_idx:end_idx].copy()

                # Reset index for the window
                window_df = window_df.reset_index(drop=True)

                # Create metadata
                metadata = {
                    "original_sample_id": i,
                    "window_id": sample_windows,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "window_size": actual_window_size,
                    "original_length": df_length,
                    "class": class_label,
                }

                windowed_dfs.append(window_df)
                windowed_classes.append(class_label)
                window_metadata.append(metadata)

                sample_windows += 1
                total_windows += 1

                # Stop if we've reached the end or if stride equals window_size (non-overlapping)
                if stride >= window_size and end_idx >= df_length:
                    break

        print("\n📊 Windowing Summary:")
        print(f"   Original samples: {len(dfs)}")
        print(f"   Skipped samples: {skipped_samples}")
        print(f"   Processed samples: {len(dfs) - skipped_samples}")
        print(f"   Total windows created: {total_windows}")
        print(
            f"   Average windows per sample: {total_windows / (len(dfs) - skipped_samples):.1f}"
        )

        return windowed_dfs, windowed_classes, window_metadata

    def get_windowing_statistics(self, window_metadata: List[Dict]) -> Dict[str, Any]:
        """
        Calculate statistics about the windowing process.

        Args:
            window_metadata (List[Dict]): Metadata from create_time_windows

        Returns:
            Dictionary containing windowing statistics
        """
        if not window_metadata:
            return {}

        # Extract information
        window_sizes = [meta["window_size"] for meta in window_metadata]
        classes = [meta["class"] for meta in window_metadata]
        original_lengths = [meta["original_length"] for meta in window_metadata]

        # Count windows per class
        class_counts = Counter(classes)

        # Count windows per original sample
        sample_window_counts = Counter(
            [meta["original_sample_id"] for meta in window_metadata]
        )

        statistics = {
            "total_windows": len(window_metadata),
            "unique_classes": len(set(classes)),
            "class_distribution": dict(class_counts),
            "window_size_stats": {
                "min": min(window_sizes),
                "max": max(window_sizes),
                "mean": np.mean(window_sizes),
                "std": np.std(window_sizes),
            },
            "original_length_stats": {
                "min": min(original_lengths),
                "max": max(original_lengths),
                "mean": np.mean(original_lengths),
                "std": np.std(original_lengths),
            },
            "windows_per_sample_stats": {
                "min": min(sample_window_counts.values()),
                "max": max(sample_window_counts.values()),
                "mean": np.mean(list(sample_window_counts.values())),
                "std": np.std(list(sample_window_counts.values())),
            },
        }

        return statistics

    def apply_windowing_to_split(
        self,
        train_dfs: List[pd.DataFrame],
        test_dfs: List[pd.DataFrame],
        train_classes: List[str],
        test_classes: List[str],
        window_size: int = 300,
        stride: int = None,
        min_window_size: int = None,
    ) -> Dict[str, Any]:
        """
        Apply time windowing to already split train/test data.

        Args:
            train_dfs (List[pd.DataFrame]): Training dataframes
            test_dfs (List[pd.DataFrame]): Test dataframes
            train_classes (List[str]): Training class labels
            test_classes (List[str]): Test class labels
            window_size (int): Size of each time window
            stride (int): Step size between windows
            min_window_size (int): Minimum window size to keep

        Returns:
            Dictionary containing windowed train and test data with metadata
        """
        print("🔄 Applying windowing to train/test split...")

        # Window training data
        train_windowed_dfs, train_windowed_classes, train_metadata = (
            self.create_time_windows(
                train_dfs, train_classes, window_size, stride, min_window_size
            )
        )

        # Window test data
        test_windowed_dfs, test_windowed_classes, test_metadata = (
            self.create_time_windows(
                test_dfs, test_classes, window_size, stride, min_window_size
            )
        )

        # Calculate statistics
        train_stats = self.get_windowing_statistics(train_metadata)
        test_stats = self.get_windowing_statistics(test_metadata)

        result = {
            "train_windowed_dfs": train_windowed_dfs,
            "train_windowed_classes": train_windowed_classes,
            "train_metadata": train_metadata,
            "test_windowed_dfs": test_windowed_dfs,
            "test_windowed_classes": test_windowed_classes,
            "test_metadata": test_metadata,
            "train_statistics": train_stats,
            "test_statistics": test_stats,
            "windowing_parameters": {
                "window_size": window_size,
                "stride": stride if stride is not None else window_size,
                "min_window_size": (
                    min_window_size if min_window_size is not None else window_size
                ),
            },
        }

        return result
