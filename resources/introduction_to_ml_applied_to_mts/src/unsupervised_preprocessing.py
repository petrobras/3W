"""
Data Preparation Utilities for Unsupervised Learning

This module provides utilities for preparing time series data for LSTM autoencoders,
including data loading, sampling, validation, and conversion functions.
"""

import numpy as np
import random
import time
import os
from collections import defaultdict, Counter


class UnsupervisedDataLoader:
    """Data loader for unsupervised learning tasks."""

    def __init__(self, persistence, config):
        """
        Initialize the data loader.

        Args:
            persistence: DataPersistence instance
            config: Configuration module
        """
        self.persistence = persistence
        self.config = config
        self.windowed_dir = os.path.join(persistence.cv_splits_dir, "windowed")

    def load_unsupervised_data(
        self,
        use_single_fold=False,
        target_fold="fold_1",
        max_normal_samples=2000,
        max_anomaly_samples=1000,
        selected_anomaly_classes=[3, 4, 8],
        enable_sampling=True,
    ):
        """
        Load and organize data for unsupervised anomaly detection.

        Args:
            use_single_fold (bool): Whether to use only one fold
            target_fold (str): Which fold to use if single fold loading
            max_normal_samples (int): Maximum normal samples to load
            max_anomaly_samples (int): Maximum anomaly samples to load
            selected_anomaly_classes (list): List of anomaly classes to include
            enable_sampling (bool): Whether to apply sampling limits

        Returns:
            tuple: (normal_windows, normal_classes, anomaly_windows, anomaly_classes, load_info)
        """
        print("Loading 3W Dataset for Unsupervised Anomaly Detection")
        print("=" * 60)

        # Validate windowed directory
        if not os.path.exists(self.windowed_dir):
            raise FileNotFoundError(
                f"Windowed data directory not found: {self.windowed_dir}. "
                "Please run Data Treatment notebook first."
            )

        # Find available folds
        fold_dirs = [
            d
            for d in os.listdir(self.windowed_dir)
            if d.startswith("fold_")
            and os.path.isdir(os.path.join(self.windowed_dir, d))
        ]
        fold_dirs.sort()

        if not fold_dirs:
            raise FileNotFoundError("No fold directories found in windowed data.")

        # Determine which folds to process
        process_folds = self._determine_folds_to_process(
            fold_dirs, use_single_fold, target_fold
        )

        print(f"Configuration:")
        print(f"   • Processing folds: {process_folds}")
        print(f"   • Selected anomaly classes: {selected_anomaly_classes}")
        print(f"   • Sampling enabled: {enable_sampling}")
        if enable_sampling:
            print(f"   • Max normal samples: {max_normal_samples}")
            print(f"   • Max anomaly samples: {max_anomaly_samples}")

        # Load and organize data
        normal_windows, normal_classes, anomaly_windows, anomaly_classes = (
            self._load_and_organize_data(
                process_folds,
                selected_anomaly_classes,
                enable_sampling,
                max_normal_samples,
                max_anomaly_samples,
            )
        )

        # Prepare load information
        load_info = {
            "folds_processed": process_folds,
            "selected_anomaly_classes": selected_anomaly_classes,
            "normal_count": len(normal_windows),
            "anomaly_count": len(anomaly_windows),
            "sampling_enabled": enable_sampling,
        }

        return (
            normal_windows,
            normal_classes,
            anomaly_windows,
            anomaly_classes,
            load_info,
        )

    def _determine_folds_to_process(self, fold_dirs, use_single_fold, target_fold):
        """Determine which folds to process based on configuration."""
        if use_single_fold:
            if target_fold in fold_dirs:
                process_folds = [target_fold]
                print(f"Using single fold: {target_fold}")
            else:
                process_folds = [fold_dirs[0]]
                print(
                    f"⚠️ Target fold '{target_fold}' not found, using: {process_folds[0]}"
                )
        else:
            process_folds = fold_dirs
            print(f"Using all {len(fold_dirs)} folds for better class coverage")

        return process_folds

    def _load_and_organize_data(
        self,
        process_folds,
        selected_anomaly_classes,
        enable_sampling,
        max_normal_samples,
        max_anomaly_samples,
    ):
        """Load and organize data from folds."""
        normal_windows, normal_classes = [], []
        anomaly_windows, anomaly_classes = [], []

        class_counts = {str(cls): 0 for cls in selected_anomaly_classes}
        class_counts["0"] = 0

        load_start = time.time()
        total_files_processed = 0

        for fold_idx, fold_name in enumerate(process_folds):
            fold_path = os.path.join(self.windowed_dir, fold_name)
            print(f"\nProcessing {fold_name} ({fold_idx + 1}/{len(process_folds)})...")

            # Load both train and test data for each fold
            all_fold_dfs, all_fold_classes = self._load_fold_data(fold_path, fold_name)

            # Separate by class
            self._separate_by_class(
                all_fold_dfs,
                all_fold_classes,
                selected_anomaly_classes,
                normal_windows,
                normal_classes,
                anomaly_windows,
                anomaly_classes,
                class_counts,
                enable_sampling,
                max_normal_samples,
                max_anomaly_samples,
            )

            total_files_processed += 2  # train + test

        load_time = time.time() - load_start
        self._print_loading_summary(
            normal_windows,
            anomaly_windows,
            anomaly_classes,
            selected_anomaly_classes,
            load_time,
            total_files_processed,
        )

        return normal_windows, normal_classes, anomaly_windows, anomaly_classes

    def _load_fold_data(self, fold_path, fold_name):
        """Load train and test data from a single fold."""
        all_fold_dfs, all_fold_classes = [], []

        print(f"   Loading and merging train+test data...", end=" ")

        for data_type in ["train", "test"]:
            pickle_file = os.path.join(
                fold_path, f"{data_type}_windowed.{self.config.SAVE_FORMAT}"
            )
            parquet_file = os.path.join(fold_path, f"{data_type}_windowed.parquet")

            fold_dfs, fold_classes = self._try_load_file(
                pickle_file, parquet_file, data_type
            )

            all_fold_dfs.extend(fold_dfs)
            all_fold_classes.extend(fold_classes)

        print(f"  -> Total: {len(all_fold_dfs)} windows")
        return all_fold_dfs, all_fold_classes

    def _try_load_file(self, pickle_file, parquet_file, data_type):
        """Try to load data from pickle or parquet file."""
        fold_dfs, fold_classes = [], []

        if os.path.exists(pickle_file):
            try:
                fold_dfs, fold_classes = self.persistence._load_dataframes(
                    pickle_file, self.config.SAVE_FORMAT
                )
                print(f"  {data_type}({len(fold_dfs)})", end="")
            except Exception:
                print(f"  {data_type}(pickle error)", end="")
                if os.path.exists(parquet_file):
                    try:
                        fold_dfs, fold_classes = self.persistence._load_from_parquet(
                            parquet_file
                        )
                        print(f"  {data_type}({len(fold_dfs)} parquet)", end="")
                    except Exception:
                        print(f"  {data_type}(failed)", end="")
        elif os.path.exists(parquet_file):
            try:
                fold_dfs, fold_classes = self.persistence._load_from_parquet(
                    parquet_file
                )
                print(f"  {data_type}({len(fold_dfs)})", end="")
            except Exception:
                print(f"  {data_type}(error)", end="")
        else:
            print(f"  {data_type}(not found)", end="")

        return fold_dfs, fold_classes

    def _separate_by_class(
        self,
        all_fold_dfs,
        all_fold_classes,
        selected_anomaly_classes,
        normal_windows,
        normal_classes,
        anomaly_windows,
        anomaly_classes,
        class_counts,
        enable_sampling,
        max_normal_samples,
        max_anomaly_samples,
    ):
        """Separate data by class (normal vs selected anomalies)."""
        for df, cls in zip(all_fold_dfs, all_fold_classes):
            cls_str = str(cls)

            if cls_str == "0":  # Normal operation
                if not enable_sampling or len(normal_windows) < max_normal_samples:
                    normal_windows.append(df)
                    normal_classes.append(cls)
                    class_counts["0"] += 1
            elif int(cls) in selected_anomaly_classes:  # Selected fault classes
                current_class_count = class_counts.get(cls_str, 0)
                max_per_class = max_anomaly_samples // len(selected_anomaly_classes)

                if not enable_sampling or current_class_count < max_per_class:
                    anomaly_windows.append(df)
                    anomaly_classes.append(cls)
                    class_counts[cls_str] += 1

    def _print_loading_summary(
        self,
        normal_windows,
        anomaly_windows,
        anomaly_classes,
        selected_anomaly_classes,
        load_time,
        total_files_processed,
    ):
        """Print loading summary and validation."""
        if normal_windows and anomaly_windows:
            print(f"\n✅ Data loading completed successfully!")
            print(f"   • Normal windows (class 0): {len(normal_windows)}")
            print(f"   • Anomaly windows: {len(anomaly_windows)}")
            print(f"   • Loading time: {load_time:.3f} seconds")
            print(f"   • Files processed: {total_files_processed}")

            # Show class distribution
            anomaly_unique, anomaly_counts = np.unique(
                anomaly_classes, return_counts=True
            )
            print(f"\nAnomaly Class Distribution:")
            for cls, count in zip(anomaly_unique, anomaly_counts):
                print(f"   • Class {cls}: {count} windows")

            # Check for missing classes
            expected_classes = set(str(cls) for cls in selected_anomaly_classes)
            found_classes = set(str(cls) for cls in anomaly_unique)
            missing_classes = expected_classes - found_classes

            if missing_classes:
                print(
                    f"\n⚠️ Warning: Missing classes from loaded data: {sorted(missing_classes)}"
                )
                print(f"   • Consider using more folds or increasing sample limits")
            else:
                print(f"\n✅ All expected anomaly classes found in the data!")

            # Show sample window info
            if normal_windows:
                sample_window = normal_windows[0]
                print(f"\nSample Window Information:")
                print(f"   • Shape: {sample_window.shape}")
                print(f"   • Features: {list(sample_window.columns)}")
        else:
            print("⚠️ Insufficient data found for novelty detection")
            print(f"   • Normal windows: {len(normal_windows)}")
            print(f"   • Anomaly windows: {len(anomaly_windows)}")


class UnsupervisedDataPreprocessor:
    """Data preprocessor for unsupervised learning tasks."""

    def __init__(
        self, max_training_samples=1000, max_anomaly_samples=300, random_seed=42
    ):
        """
        Initialize the data preprocessor.

        Args:
            max_training_samples (int): Maximum number of normal samples for training
            max_anomaly_samples (int): Maximum number of anomaly samples for testing
            random_seed (int): Random seed for reproducibility
        """
        self.max_training_samples = max_training_samples
        self.max_anomaly_samples = max_anomaly_samples
        self.random_seed = random_seed
        random.seed(random_seed)

    def sample_windows(self, normal_windows, anomaly_windows, anomaly_classes=None):
        """
        Sample windows for training efficiency with balanced class representation.

        Args:
            normal_windows (list): List of normal operation windows
            anomaly_windows (list): List of anomaly windows
            anomaly_classes (list, optional): List of anomaly class labels for balanced sampling

        Returns:
            tuple: (sampled_normal_windows, sampled_anomaly_windows, sampled_anomaly_classes)
        """
        print("⚡ Smart Data Sampling for Training Efficiency")
        print("=" * 50)

        print(f"🎯 Training optimization settings:")
        print(f"   • Max normal samples for training: {self.max_training_samples}")
        print(f"   • Max anomaly samples for testing: {self.max_anomaly_samples}")

        # Sample normal data for training
        if len(normal_windows) > self.max_training_samples:
            print(
                f"📊 Sampling {self.max_training_samples} normal windows from {len(normal_windows)} available..."
            )
            sampled_indices = random.sample(
                range(len(normal_windows)), self.max_training_samples
            )
            sampled_normal_windows = [normal_windows[i] for i in sampled_indices]
        else:
            print(f"📊 Using all {len(normal_windows)} normal windows...")
            sampled_normal_windows = normal_windows

        # Handle anomaly data sampling with class balance
        if (
            anomaly_classes is not None
            and len(anomaly_windows) > self.max_anomaly_samples
        ):
            print(
                f"📊 Balanced sampling {self.max_anomaly_samples} anomaly windows from {len(anomaly_windows)} available..."
            )

            # Group by class for balanced sampling
            from collections import defaultdict

            class_indices = defaultdict(list)

            for i, cls in enumerate(anomaly_classes):
                class_indices[str(cls)].append(i)

            unique_classes = list(class_indices.keys())
            samples_per_class = self.max_anomaly_samples // len(unique_classes)

            print(f"   • Target samples per class: {samples_per_class}")

            sampled_indices = []
            sampled_anomaly_classes = []

            for cls in unique_classes:
                cls_indices = class_indices[cls]
                if len(cls_indices) > samples_per_class:
                    cls_sampled = random.sample(cls_indices, samples_per_class)
                else:
                    cls_sampled = cls_indices

                sampled_indices.extend(cls_sampled)
                sampled_anomaly_classes.extend(
                    [anomaly_classes[i] for i in cls_sampled]
                )

                print(
                    f"   • Class {cls}: {len(cls_sampled)} samples selected from {len(cls_indices)} available"
                )

            sampled_anomaly_windows = [anomaly_windows[i] for i in sampled_indices]

        else:
            print(f"📊 Using all {len(anomaly_windows)} anomaly windows...")
            sampled_anomaly_windows = anomaly_windows
            sampled_anomaly_classes = (
                anomaly_classes
                if anomaly_classes is not None
                else [None] * len(anomaly_windows)
            )

        print(f"\n✅ Sampling complete:")
        print(f"   • Normal windows: {len(sampled_normal_windows)}")
        print(f"   • Anomaly windows: {len(sampled_anomaly_windows)}")

        if anomaly_classes is not None:
            # Show final class distribution
            from collections import Counter

            class_dist = Counter(str(cls) for cls in sampled_anomaly_classes)
            print(f"   • Anomaly class distribution: {dict(class_dist)}")

        return sampled_normal_windows, sampled_anomaly_windows, sampled_anomaly_classes

    def convert_windows_to_arrays(
        self, windows, window_type="normal", progress_step=200
    ):
        """
        Convert window DataFrames to numpy arrays with quality checks.

        Args:
            windows (list): List of window DataFrames
            window_type (str): Type of windows ("normal" or "anomaly")
            progress_step (int): Show progress every N windows

        Returns:
            list: List of valid numpy arrays
        """
        print(f"📊 Converting {window_type} windows to arrays...", end=" ")
        start_conversion = time.time()

        arrays = []
        for i, window in enumerate(windows):
            if i % progress_step == 0 and i > 0:
                print(
                    f"\r📊 Converting {window_type} windows to arrays... {i}/{len(windows)}",
                    end="",
                )

            # Get the DataFrame values and remove the class column
            if hasattr(window, "values"):
                window_data = window.copy()
                # Remove 'class' column if it exists
                if "class" in window_data.columns:
                    window_data = window_data.drop("class", axis=1)
                window_array = window_data.values
            else:
                # If it's already an array, assume last column is class and remove it
                if len(window.shape) == 2 and window.shape[1] > 1:
                    window_array = window[:, :-1]  # Remove last column (class)
                else:
                    window_array = window

            # Check for NaN or infinite values
            if np.isfinite(window_array).all():
                arrays.append(window_array)
            else:
                print(f"\n⚠️ Skipping {window_type} window {i} due to non-finite values")

        conversion_time = time.time() - start_conversion
        print(
            f"\r📊 Converting {window_type} windows to arrays... ✅ ({len(arrays)} valid processed)"
        )

        return arrays

    def validate_and_prepare_arrays(self, normal_arrays, anomaly_arrays):
        """
        Validate array shapes and prepare final datasets.

        Args:
            normal_arrays (list): List of normal arrays
            anomaly_arrays (list): List of anomaly arrays

        Returns:
            tuple: (normal_data, anomaly_data) as numpy arrays
        """
        print("🔍 Validating array shapes and data quality...")

        # Check if we have valid arrays
        if not normal_arrays or not anomaly_arrays:
            raise ValueError("No valid arrays found after conversion")

        # Get the shape from the first normal array
        expected_shape = normal_arrays[0].shape
        print(f"Expected shape: {expected_shape}")

        # Filter arrays to ensure consistent shapes
        valid_normal_arrays = []
        valid_anomaly_arrays = []

        for arr in normal_arrays:
            if arr.shape == expected_shape and np.isfinite(arr).all():
                valid_normal_arrays.append(arr)

        for arr in anomaly_arrays:
            if arr.shape == expected_shape and np.isfinite(arr).all():
                valid_anomaly_arrays.append(arr)

        print(
            f"✅ Valid arrays: {len(valid_normal_arrays)} normal, {len(valid_anomaly_arrays)} anomaly"
        )

        normal_data = np.array(valid_normal_arrays, dtype=np.float32)
        anomaly_data = np.array(valid_anomaly_arrays, dtype=np.float32)

        # Data quality checks
        print(f"📊 Data quality checks:")
        print(
            f"   • Normal data range: [{np.min(normal_data):.3f}, {np.max(normal_data):.3f}]"
        )
        print(
            f"   • Anomaly data range: [{np.min(anomaly_data):.3f}, {np.max(anomaly_data):.3f}]"
        )
        print(f"   • Normal data finite: {np.isfinite(normal_data).all()}")
        print(f"   • Anomaly data finite: {np.isfinite(anomaly_data).all()}")

        return normal_data, anomaly_data

    def apply_stability_scaling(self, normal_data, anomaly_data):
        """
        Apply additional normalization for numerical stability.

        Args:
            normal_data (np.array): Normal data array
            anomaly_data (np.array): Anomaly data array

        Returns:
            tuple: (normal_scaled, anomaly_scaled)
        """
        print("📊 Additional Data Normalization for Stability")
        print("=" * 45)

        # Convert to float32 and clip extreme values for stability
        normal_scaled = normal_data.astype(np.float32)
        anomaly_scaled = anomaly_data.astype(np.float32)

        # Clip extreme values for stability
        normal_scaled = np.clip(normal_scaled, 0.001, 0.999)
        anomaly_scaled = np.clip(anomaly_scaled, 0.001, 0.999)

        print(f"📏 Enhanced data characteristics:")
        print(
            f"   • Normal data range: [{np.min(normal_scaled):.3f}, {np.max(normal_scaled):.3f}]"
        )
        print(
            f"   • Anomaly data range: [{np.min(anomaly_scaled):.3f}, {np.max(anomaly_scaled):.3f}]"
        )
        print(f"   • Data clipped to avoid extreme values")
        print(f"   • Float32 precision for stability")

        return normal_scaled, anomaly_scaled

    def prepare_full_pipeline(
        self, normal_windows, anomaly_windows, anomaly_classes=None
    ):
        """
        Run the complete data preparation pipeline.

        Args:
            normal_windows (list): List of normal operation windows
            anomaly_windows (list): List of anomaly windows
            anomaly_classes (list, optional): List of anomaly class labels

        Returns:
            tuple: (normal_scaled, anomaly_scaled, data_info, sampled_anomaly_classes)
        """
        print("🔧 Complete Data Preparation Pipeline")
        print("=" * 50)

        # Step 1: Sample windows with balanced class representation
        sampled_normal, sampled_anomaly, sampled_anomaly_classes = self.sample_windows(
            normal_windows, anomaly_windows, anomaly_classes
        )

        # Step 2: Convert to arrays
        normal_arrays = self.convert_windows_to_arrays(sampled_normal, "normal")
        anomaly_arrays = self.convert_windows_to_arrays(
            sampled_anomaly, "anomaly", progress_step=100
        )

        # Step 3: Validate and prepare
        normal_data, anomaly_data = self.validate_and_prepare_arrays(
            normal_arrays, anomaly_arrays
        )

        # Step 4: Apply stability scaling
        normal_scaled, anomaly_scaled = self.apply_stability_scaling(
            normal_data, anomaly_data
        )

        # Get data info
        n_normal_samples, time_steps, n_features = normal_scaled.shape
        data_info = {
            "time_steps": time_steps,
            "n_features": n_features,
            "n_normal_samples": n_normal_samples,
            "n_anomaly_samples": anomaly_scaled.shape[0],
        }

        print(f"\n📐 Final Data Shapes:")
        print(f"   • Normal data: {normal_scaled.shape}")
        print(f"   • Anomaly data: {anomaly_scaled.shape}")
        print(f"   • Time steps per window: {time_steps}")
        print(f"   • Features per time step: {n_features} (class column removed)")

        # Adjust anomaly classes to match the final array size
        if (
            sampled_anomaly_classes is not None
            and len(sampled_anomaly_classes) != anomaly_scaled.shape[0]
        ):
            print(
                f"   ⚠️ Adjusting anomaly classes length from {len(sampled_anomaly_classes)} to {anomaly_scaled.shape[0]}"
            )
            sampled_anomaly_classes = sampled_anomaly_classes[: anomaly_scaled.shape[0]]

        return normal_scaled, anomaly_scaled, data_info, sampled_anomaly_classes
