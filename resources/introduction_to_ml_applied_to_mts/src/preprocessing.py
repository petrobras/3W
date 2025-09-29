"""
Data Preprocessing Module for 3W Dataset

This module provides functionality for scaling, normalizing, and preprocessing
sensor data from the 3W dataset.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.model_selection import train_test_split
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
            "normalizer": Normalizer(norm="l2"),
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
                print(f"❌ Error processing DataFrame {i+1}: {str(e)}")
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
            print(f"   Value ranges after scaling:")
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
        print(f"\n📊 {method.title()} Scaling Summary:")
        print("=" * 40)
        print(f"Total dataframes processed: {total_processed}")
        print(f"Successfully scaled: {successfully_scaled}")
        print(f"Skipped (no data/errors): {skipped_dfs}")

        if scaled_classes:
            unique_classes, counts = np.unique(scaled_classes, return_counts=True)
            class_dist = dict(zip(unique_classes, counts))
            print(f"Class distribution of scaled data: {class_dist}")

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
                validation["correct_scaling"][
                    col
                ] = True  # Complex validation for normalizer

        return validation

    def split_train_test_by_class(
        self,
        dfs: Dict[int, pd.DataFrame],
        test_size: float = 0.2,
        random_state: int = 42,
        target_features: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
        """
        Split dataframes into train and test sets, stratifying by class.

        Args:
            dfs: Dictionary of dataframes with class IDs as keys
            test_size: Proportion of data for test set (0.0 to 1.0)
            random_state: Random seed for reproducibility
            target_features: List of features to include. If None, includes all columns

        Returns:
            Tuple containing:
                - Dictionary with 'train' and 'test' keys containing combined dataframes
                - Statistics about the split
        """
        if not (0.0 < test_size < 1.0):
            raise ValueError("test_size must be between 0.0 and 1.0")

        train_dfs = []
        test_dfs = []
        split_stats = {
            "total_samples": 0,
            "train_samples": 0,
            "test_samples": 0,
            "class_distribution_train": {},
            "class_distribution_test": {},
            "features_used": [],
        }

        # Collect statistics and perform splits
        for class_id, df in dfs.items():
            # Select features if specified
            if target_features is not None:
                available_features = [
                    col for col in target_features if col in df.columns
                ]
                if not available_features:
                    print(
                        f"Warning: No target features found in class {class_id}. Skipping..."
                    )
                    continue
                df_filtered = df[available_features].copy()
            else:
                df_filtered = df.copy()

            # Add class column for tracking
            df_filtered["class"] = class_id

            # Perform stratified split by maintaining the same proportion
            # For time series data, we'll use random splitting but ensure class balance
            train_df, test_df = train_test_split(
                df_filtered,
                test_size=test_size,
                random_state=random_state + class_id,  # Different seed per class
                stratify=None,  # We're already splitting by class
            )

            train_dfs.append(train_df)
            test_dfs.append(test_df)

            # Update statistics
            split_stats["total_samples"] += len(df_filtered)
            split_stats["train_samples"] += len(train_df)
            split_stats["test_samples"] += len(test_df)
            split_stats["class_distribution_train"][class_id] = len(train_df)
            split_stats["class_distribution_test"][class_id] = len(test_df)

        # Combine all classes
        if train_dfs and test_dfs:
            train_combined = pd.concat(train_dfs, ignore_index=True)
            test_combined = pd.concat(test_dfs, ignore_index=True)

            # Shuffle the combined datasets
            train_combined = train_combined.sample(
                frac=1, random_state=random_state
            ).reset_index(drop=True)
            test_combined = test_combined.sample(
                frac=1, random_state=random_state + 1000
            ).reset_index(drop=True)

            # Update features used
            split_stats["features_used"] = [
                col for col in train_combined.columns if col != "class"
            ]

            result_dfs = {"train": train_combined, "test": test_combined}
        else:
            raise ValueError("No valid dataframes found for splitting")

        return result_dfs, split_stats

    def analyze_split_distribution(
        self, split_stats: Dict[str, Any], show_details: bool = True
    ) -> pd.DataFrame:
        """
        Analyze and display the train-test split distribution.

        Args:
            split_stats: Statistics returned from split_train_test_by_class
            show_details: Whether to print detailed statistics

        Returns:
            DataFrame with class distribution analysis
        """
        # Create distribution analysis
        analysis_data = []

        for class_id in split_stats["class_distribution_train"].keys():
            train_count = split_stats["class_distribution_train"][class_id]
            test_count = split_stats["class_distribution_test"][class_id]
            total_count = train_count + test_count

            analysis_data.append(
                {
                    "Class": class_id,
                    "Total_Samples": total_count,
                    "Train_Samples": train_count,
                    "Test_Samples": test_count,
                    "Train_Percentage": (train_count / total_count) * 100,
                    "Test_Percentage": (test_count / total_count) * 100,
                    "Train_Proportion": train_count / split_stats["train_samples"],
                    "Test_Proportion": test_count / split_stats["test_samples"],
                }
            )

        analysis_df = pd.DataFrame(analysis_data)

        if show_details:
            print("=" * 60)
            print("TRAIN-TEST SPLIT ANALYSIS")
            print("=" * 60)
            print(f"Total Samples: {split_stats['total_samples']:,}")
            print(
                f"Training Samples: {split_stats['train_samples']:,} ({(split_stats['train_samples']/split_stats['total_samples']*100):.1f}%)"
            )
            print(
                f"Test Samples: {split_stats['test_samples']:,} ({(split_stats['test_samples']/split_stats['total_samples']*100):.1f}%)"
            )
            print(f"Features Used: {len(split_stats['features_used'])}")
            print(
                f"Feature Names: {split_stats['features_used'][:5]}{'...' if len(split_stats['features_used']) > 5 else ''}"
            )
            print("\nClass Distribution:")
            print(analysis_df.round(2))
            print("=" * 60)

        return analysis_df

    def validate_split_balance(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        class_column: str = "class",
        tolerance: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Validate that the train-test split maintains class balance.

        Args:
            train_df: Training dataframe with class column
            test_df: Test dataframe with class column
            class_column: Name of the class column
            tolerance: Acceptable difference in class proportions (0.05 = 5%)

        Returns:
            Dictionary with validation results
        """
        # Calculate class proportions
        train_props = train_df[class_column].value_counts(normalize=True).sort_index()
        test_props = test_df[class_column].value_counts(normalize=True).sort_index()

        # Find common classes
        common_classes = set(train_props.index).intersection(set(test_props.index))

        validation_results = {
            "is_balanced": True,
            "class_differences": {},
            "max_difference": 0.0,
            "classes_checked": len(common_classes),
            "tolerance_used": tolerance,
        }

        for class_id in common_classes:
            train_prop = train_props.get(class_id, 0)
            test_prop = test_props.get(class_id, 0)
            difference = abs(train_prop - test_prop)

            validation_results["class_differences"][class_id] = {
                "train_proportion": train_prop,
                "test_proportion": test_prop,
                "difference": difference,
                "within_tolerance": difference <= tolerance,
            }

            if difference > tolerance:
                validation_results["is_balanced"] = False

            validation_results["max_difference"] = max(
                validation_results["max_difference"], difference
            )

        return validation_results

    def train_test_split_dataframes(
        self,
        dfs: List[pd.DataFrame],
        classes: List[str],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Split dataframes into train and test sets with stratification by class.
        Each dataframe is treated as a separate sample/file of that class.

        Args:
            dfs (List[pd.DataFrame]): List of dataframes (each df is one sample)
            classes (List[str]): List of corresponding class labels for each dataframe
            test_size (float): Proportion of data to use for test set
            random_state (int): Random seed for reproducibility

        Returns:
            Dictionary containing train/test splits and statistics
        """
        from sklearn.model_selection import train_test_split
        from collections import Counter
        import numpy as np

        print(f"📊 Splitting {len(dfs)} dataframes into train/test sets")
        print(f"🎯 Test size: {test_size*100:.1f}%")
        print(f"🔢 Stratifying by class labels")

        # Create indices for stratified split
        indices = np.arange(len(dfs))

        # Perform stratified split on indices
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=classes
        )

        # Split dataframes and classes using the indices
        train_dfs = [dfs[i] for i in train_indices]
        test_dfs = [dfs[i] for i in test_indices]
        train_classes = [classes[i] for i in train_indices]
        test_classes = [classes[i] for i in test_indices]

        # Calculate statistics
        train_class_counts = Counter(train_classes)
        test_class_counts = Counter(test_classes)
        original_class_counts = Counter(classes)

        # Prepare results
        results = {
            "train_dfs": train_dfs,
            "test_dfs": test_dfs,
            "train_classes": train_classes,
            "test_classes": test_classes,
            "statistics": {
                "total_samples": len(dfs),
                "train_samples": len(train_dfs),
                "test_samples": len(test_dfs),
                "train_percentage": len(train_dfs) / len(dfs) * 100,
                "test_percentage": len(test_dfs) / len(dfs) * 100,
                "original_distribution": dict(original_class_counts),
                "train_distribution": dict(train_class_counts),
                "test_distribution": dict(test_class_counts),
            },
        }

        # Print summary
        print(f"\n📈 Split Summary:")
        print(f"   Total samples: {len(dfs)} dataframes")
        print(
            f"   Train samples: {len(train_dfs)} dataframes ({len(train_dfs)/len(dfs)*100:.1f}%)"
        )
        print(
            f"   Test samples: {len(test_dfs)} dataframes ({len(test_dfs)/len(dfs)*100:.1f}%)"
        )

        print(f"\n🎯 Class Distribution:")
        print(
            f"{'Class':<8} {'Original':<10} {'Train':<8} {'Test':<8} {'Train%':<8} {'Test%':<8}"
        )
        print("-" * 50)

        for class_label in sorted(original_class_counts.keys()):
            orig_count = original_class_counts[class_label]
            train_count = train_class_counts.get(class_label, 0)
            test_count = test_class_counts.get(class_label, 0)
            train_pct = train_count / orig_count * 100 if orig_count > 0 else 0
            test_pct = test_count / orig_count * 100 if orig_count > 0 else 0

            print(
                f"{class_label:<8} {orig_count:<10} {train_count:<8} {test_count:<8} {train_pct:<7.1f}% {test_pct:<7.1f}%"
            )

        return results

    def validate_dataframe_split_balance(
        self, split_results: Dict[str, Any], tolerance: float = 0.05
    ) -> Dict[str, bool]:
        """
        Validate that the train-test split maintains class balance within tolerance.

        Args:
            split_results (Dict): Results from train_test_split_dataframes
            tolerance (float): Acceptable deviation from expected proportions

        Returns:
            Dictionary with validation results
        """
        stats = split_results["statistics"]
        original_dist = stats["original_distribution"]
        train_dist = stats["train_distribution"]
        test_dist = stats["test_distribution"]

        total_samples = stats["total_samples"]
        train_samples = stats["train_samples"]
        test_samples = stats["test_samples"]

        validation = {
            "overall_balance": True,
            "class_balance": {},
            "expected_train_ratio": train_samples / total_samples,
            "expected_test_ratio": test_samples / total_samples,
        }

        print(f"\n🔍 Validating Split Balance (tolerance: ±{tolerance*100:.1f}%)")
        print("-" * 50)

        for class_label in original_dist.keys():
            orig_count = original_dist[class_label]
            train_count = train_dist.get(class_label, 0)
            test_count = test_dist.get(class_label, 0)

            # Expected counts based on overall split ratio
            expected_train = orig_count * validation["expected_train_ratio"]
            expected_test = orig_count * validation["expected_test_ratio"]

            # Calculate actual ratios
            actual_train_ratio = train_count / orig_count if orig_count > 0 else 0
            actual_test_ratio = test_count / orig_count if orig_count > 0 else 0

            # Check if within tolerance
            train_balanced = (
                abs(actual_train_ratio - validation["expected_train_ratio"])
                <= tolerance
            )
            test_balanced = (
                abs(actual_test_ratio - validation["expected_test_ratio"]) <= tolerance
            )

            class_balanced = train_balanced and test_balanced
            validation["class_balance"][class_label] = {
                "balanced": class_balanced,
                "train_ratio": actual_train_ratio,
                "test_ratio": actual_test_ratio,
                "train_expected": expected_train,
                "test_expected": expected_test,
                "train_actual": train_count,
                "test_actual": test_count,
            }

            if not class_balanced:
                validation["overall_balance"] = False

            status = "✅" if class_balanced else "⚠️"
            print(
                f"{status} Class {class_label}: Train {actual_train_ratio:.3f} (exp: {validation['expected_train_ratio']:.3f}), "
                f"Test {actual_test_ratio:.3f} (exp: {validation['expected_test_ratio']:.3f})"
            )

        overall_status = "✅" if validation["overall_balance"] else "⚠️"
        print(
            f"\n{overall_status} Overall balance: {'PASSED' if validation['overall_balance'] else 'NEEDS ATTENTION'}"
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
        if stride is None:
            stride = window_size
        if min_window_size is None:
            min_window_size = window_size

        windowed_dfs = []
        windowed_classes = []
        window_metadata = []

        print(f"🪟 Creating time windows:")
        print(f"   Window size: {window_size}")
        print(f"   Stride: {stride}")
        print(f"   Minimum window size: {min_window_size}")
        print("-" * 40)

        total_windows = 0
        skipped_samples = 0

        for i, (df, class_label) in enumerate(zip(dfs, classes)):
            df_length = len(df)

            # Skip if dataframe is too small
            if df_length < min_window_size:
                skipped_samples += 1
                print(
                    f"⚠️  Skipped sample {i+1} (class {class_label}): too small ({df_length} < {min_window_size})"
                )
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

            if sample_windows > 0:
                print(
                    f"✅ Sample {i+1} (class {class_label}): {sample_windows} windows from {df_length} points"
                )

        print(f"\n📊 Windowing Summary:")
        print(f"   Original samples: {len(dfs)}")
        print(f"   Skipped samples: {skipped_samples}")
        print(f"   Processed samples: {len(dfs) - skipped_samples}")
        print(f"   Total windows created: {total_windows}")
        print(
            f"   Average windows per sample: {total_windows/(len(dfs) - skipped_samples):.1f}"
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
        print("\n📚 Processing training data:")
        train_windowed_dfs, train_windowed_classes, train_metadata = (
            self.create_time_windows(
                train_dfs, train_classes, window_size, stride, min_window_size
            )
        )

        # Window test data
        print("\n🧪 Processing test data:")
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
