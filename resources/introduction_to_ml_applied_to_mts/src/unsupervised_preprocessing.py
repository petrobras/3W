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

    def load_per_fold_data(
        self,
        selected_anomaly_classes=[3, 4, 8],
        max_normal_samples=500,
        max_anomaly_samples=200,
    ):
        """
        Load data organized by fold for per-fold evaluation.
        
        Returns:
            dict: {fold_name: {'train_normal': arrays, 'test_normal': arrays, 'test_anomaly': arrays, 'test_anomaly_classes': classes}}
        """
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

        fold_data = {}
        
        for fold_name in fold_dirs:
            fold_path = os.path.join(self.windowed_dir, fold_name)
            
            try:
                # Load train data (only normal for training)
                train_dfs, train_classes = self._load_single_file_type(fold_path, "train")
                train_normal_windows = [df for df, cls in zip(train_dfs, train_classes) if int(cls) == 0]
                
                # Load test data (both normal and anomaly)
                test_dfs, test_classes = self._load_single_file_type(fold_path, "test")
                
                # Separate test data
                test_normal_windows = []
                test_anomaly_windows = []
                test_anomaly_classes = []
                
                for df, cls in zip(test_dfs, test_classes):
                    if int(cls) == 0:
                        test_normal_windows.append(df)
                    elif int(cls) in selected_anomaly_classes:
                        test_anomaly_windows.append(df)
                        test_anomaly_classes.append(cls)
                
                # Sample data if limits specified
                if max_normal_samples and len(train_normal_windows) > max_normal_samples:
                    indices = np.random.choice(len(train_normal_windows), max_normal_samples, replace=False)
                    train_normal_windows = [train_normal_windows[i] for i in indices]
                
                if max_normal_samples and len(test_normal_windows) > max_normal_samples//2:
                    indices = np.random.choice(len(test_normal_windows), max_normal_samples//2, replace=False)
                    test_normal_windows = [test_normal_windows[i] for i in indices]
                    
                if max_anomaly_samples and len(test_anomaly_windows) > max_anomaly_samples:
                    indices = np.random.choice(len(test_anomaly_windows), max_anomaly_samples, replace=False)
                    test_anomaly_windows = [test_anomaly_windows[i] for i in indices]
                    test_anomaly_classes = [test_anomaly_classes[i] for i in indices]
                
                # Convert to arrays and preprocess
                if len(train_normal_windows) > 0 and (len(test_normal_windows) > 0 or len(test_anomaly_windows) > 0):
                    preprocessor = UnsupervisedDataPreprocessor(
                        max_training_samples=len(train_normal_windows),
                        max_anomaly_samples=len(test_anomaly_windows) + len(test_normal_windows),
                        random_seed=42,
                    )
                    
                    # Combine test data for preprocessing
                    all_test_windows = test_normal_windows + test_anomaly_windows
                    all_test_classes = [0] * len(test_normal_windows) + test_anomaly_classes
                    
                    train_scaled, test_scaled, data_info, processed_test_classes = (
                        preprocessor.prepare_full_pipeline(
                            train_normal_windows, all_test_windows, all_test_classes
                        )
                    )
                    
                    # Split test data back into normal and anomaly
                    n_test_normal = len(test_normal_windows)
                    test_normal_scaled = test_scaled[:n_test_normal]
                    test_anomaly_scaled = test_scaled[n_test_normal:]
                    test_anomaly_classes_final = processed_test_classes[n_test_normal:] if processed_test_classes else []
                    
                    fold_data[fold_name] = {
                        'train_normal': train_scaled,
                        'test_normal': test_normal_scaled,
                        'test_anomaly': test_anomaly_scaled,
                        'test_anomaly_classes': test_anomaly_classes_final,
                        'data_info': data_info
                    }
                    
            except Exception as e:
                print(f"Error loading {fold_name}: {str(e)}")
        
        return fold_data
    
    def _load_single_file_type(self, fold_path, data_type):
        """Load a single file type (train or test) from a fold."""
        pickle_file = os.path.join(
            fold_path, f"{data_type}_windowed.{self.config.SAVE_FORMAT}"
        )
        parquet_file = os.path.join(fold_path, f"{data_type}_windowed.parquet")
        
        return self._try_load_file(pickle_file, parquet_file, data_type)


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
        # Sample normal data for training
        if len(normal_windows) > self.max_training_samples:
            sampled_indices = random.sample(
                range(len(normal_windows)), self.max_training_samples
            )
            sampled_normal_windows = [normal_windows[i] for i in sampled_indices]
        else:
            sampled_normal_windows = normal_windows

        # Handle anomaly data sampling with class balance
        if (
            anomaly_classes is not None
            and len(anomaly_windows) > self.max_anomaly_samples
        ):
            # Group by class for balanced sampling
            from collections import defaultdict

            class_indices = defaultdict(list)

            for i, cls in enumerate(anomaly_classes):
                class_indices[str(cls)].append(i)

            unique_classes = list(class_indices.keys())
            samples_per_class = self.max_anomaly_samples // len(unique_classes)

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

            sampled_anomaly_windows = [anomaly_windows[i] for i in sampled_indices]

        else:
            sampled_anomaly_windows = anomaly_windows
            sampled_anomaly_classes = (
                anomaly_classes
                if anomaly_classes is not None
                else [None] * len(anomaly_windows)
            )

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
        arrays = []
        for i, window in enumerate(windows):
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
        # Check if we have valid arrays
        if not normal_arrays or not anomaly_arrays:
            raise ValueError("No valid arrays found after conversion")

        # Get the shape from the first normal array
        expected_shape = normal_arrays[0].shape

        # Filter arrays to ensure consistent shapes
        valid_normal_arrays = []
        valid_anomaly_arrays = []

        for arr in normal_arrays:
            if arr.shape == expected_shape and np.isfinite(arr).all():
                valid_normal_arrays.append(arr)

        for arr in anomaly_arrays:
            if arr.shape == expected_shape and np.isfinite(arr).all():
                valid_anomaly_arrays.append(arr)

        normal_data = np.array(valid_normal_arrays, dtype=np.float32)
        anomaly_data = np.array(valid_anomaly_arrays, dtype=np.float32)

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
        # Convert to float32 and clip extreme values for stability
        normal_scaled = normal_data.astype(np.float32)
        anomaly_scaled = anomaly_data.astype(np.float32)

        # Clip extreme values for stability
        normal_scaled = np.clip(normal_scaled, 0.001, 0.999)
        anomaly_scaled = np.clip(anomaly_scaled, 0.001, 0.999)

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

        # Adjust anomaly classes to match the final array size
        if (
            sampled_anomaly_classes is not None
            and len(sampled_anomaly_classes) != anomaly_scaled.shape[0]
        ):
            sampled_anomaly_classes = sampled_anomaly_classes[: anomaly_scaled.shape[0]]

        return normal_scaled, anomaly_scaled, data_info, sampled_anomaly_classes


class DistanceAnomalyDetector:
    """
    One-Class SVM based anomaly detector for time series data.
    
    This class provides a distance-based approach to anomaly detection using 
    One-Class Support Vector Machines (OCSVM), which learns a decision boundary
    around normal data and classifies samples based on their distance to this boundary.
    """
    
    def __init__(self, nu=0.05, gamma='scale', kernel='rbf'):
        """
        Initialize the distance-based anomaly detector.
        
        Args:
            nu (float): Upper bound on fraction of training errors (0.01-0.1)
                       Controls the trade-off between smoothness and training error
            gamma (str/float): Kernel coefficient for RBF kernel
                              'scale' uses 1/(n_features * X.var()) as value
            kernel (str): Kernel type for SVM ('rbf', 'linear', 'poly', 'sigmoid')
        """
        self.nu = nu
        self.gamma = gamma
        self.kernel = kernel
        self.model = None
        self._is_fitted = False
        
    def _flatten_time_series(self, data):
        """
        Flatten time series data for SVM input.
        
        Args:
            data (np.array): Time series data with shape (n_samples, time_steps, n_features)
            
        Returns:
            np.array: Flattened data with shape (n_samples, time_steps * n_features)
        """
        return data.reshape(data.shape[0], -1)
    
    def fit(self, normal_data, verbose=True):
        """
        Train the One-Class SVM on normal data.
        
        Args:
            normal_data (np.array): Normal training data with shape (n_samples, time_steps, n_features)
            verbose (bool): Whether to print training information
            
        Returns:
            self: Fitted detector instance
        """
        from sklearn.svm import OneClassSVM
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        # Flatten time series to feature vectors
        X_flat = self._flatten_time_series(normal_data)
        
        # Create pipeline with scaling and OCSVM
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('ocsvm', OneClassSVM(nu=self.nu, gamma=self.gamma, kernel=self.kernel))
        ])
        
        # Fit the model
        self.model.fit(X_flat)
        self._is_fitted = True
        
        if verbose:
            print(f"✅ One-Class SVM trained:")
            print(f"   • Training samples: {len(normal_data)}")
            print(f"   • Features per sample: {X_flat.shape[1]}")
            print(f"   • Nu parameter: {self.nu}")
            print(f"   • Kernel: {self.kernel}")
            print(f"   • Support vectors: {self.model.named_steps['ocsvm'].support_vectors_.shape[0]}")
        
        return self
        
    def predict(self, data):
        """
        Predict anomalies in data.
        
        Args:
            data (np.array): Data to predict with shape (n_samples, time_steps, n_features)
            
        Returns:
            np.array: Predictions (1=normal, -1=anomaly)
        """
        if not self._is_fitted:
            raise ValueError("Detector must be fitted before making predictions. Call fit() first.")
            
        X_flat = self._flatten_time_series(data)
        return self.model.predict(X_flat)
    
    def decision_function(self, data):
        """
        Compute anomaly scores (signed distance to boundary).
        
        Args:
            data (np.array): Data to score with shape (n_samples, time_steps, n_features)
            
        Returns:
            np.array: Anomaly scores (negative = anomaly, positive = normal)
        """
        if not self._is_fitted:
            raise ValueError("Detector must be fitted before computing scores. Call fit() first.")
            
        X_flat = self._flatten_time_series(data)
        return self.model.decision_function(X_flat)
    
    def get_anomaly_scores(self, data):
        """
        Get anomaly scores converted to reconstruction-error-like format.
        
        This method converts OCSVM decision scores to a format similar to reconstruction
        errors, where higher values indicate more anomalous behavior.
        
        Args:
            data (np.array): Data to score
            
        Returns:
            np.array: Anomaly scores (higher = more anomalous)
        """
        # Get decision function scores (negative for anomalies)
        decision_scores = self.decision_function(data)
        
        # Convert to "error-like" scores (higher = more anomalous)
        # Negate the scores so anomalies have higher values
        anomaly_scores = -decision_scores
        
        return anomaly_scores
    
    def evaluate_performance(self, normal_data, anomaly_data, anomaly_classes=None):
        """
        Evaluate detector performance on test data.
        
        Args:
            normal_data (np.array): Normal test data
            anomaly_data (np.array): Anomaly test data
            anomaly_classes (list): Class labels for anomaly data (optional)
            
        Returns:
            dict: Performance metrics
        """
        if not self._is_fitted:
            raise ValueError("Detector must be fitted before evaluation. Call fit() first.")
        
        # Get predictions
        normal_predictions = self.predict(normal_data)
        anomaly_predictions = self.predict(anomaly_data)
        
        # Get anomaly scores
        normal_scores = self.get_anomaly_scores(normal_data)
        anomaly_scores = self.get_anomaly_scores(anomaly_data)
        
        # Calculate metrics
        normal_accuracy = np.mean(normal_predictions == 1)
        anomaly_accuracy = np.mean(anomaly_predictions == -1)
        
        total_correct = np.sum(normal_predictions == 1) + np.sum(anomaly_predictions == -1)
        total_samples = len(normal_data) + len(anomaly_data)
        overall_accuracy = total_correct / total_samples
        
        results = {
            'normal_accuracy': normal_accuracy,
            'anomaly_accuracy': anomaly_accuracy,
            'overall_accuracy': overall_accuracy,
            'normal_scores': normal_scores,
            'anomaly_scores': anomaly_scores,
            'normal_predictions': normal_predictions,
            'anomaly_predictions': anomaly_predictions,
            'n_normal': len(normal_data),
            'n_anomaly': len(anomaly_data),
            'n_support_vectors': self.model.named_steps['ocsvm'].support_vectors_.shape[0]
        }
        
        if anomaly_classes is not None:
            results['anomaly_classes'] = anomaly_classes
            
        return results


class EnsembleAnomalyDetector:
    """
    Ensemble anomaly detector combining multiple detection methods.
    
    This class can combine different anomaly detection approaches (e.g., OCSVM with different
    parameters, Isolation Forest, etc.) to create a more robust detection system.
    """
    
    def __init__(self, detectors=None):
        """
        Initialize ensemble detector.
        
        Args:
            detectors (list): List of detector instances to ensemble
        """
        self.detectors = detectors or []
        self._is_fitted = False
        
    def add_detector(self, detector):
        """Add a detector to the ensemble."""
        self.detectors.append(detector)
        
    def fit(self, normal_data, verbose=True):
        """
        Fit all detectors in the ensemble.
        
        Args:
            normal_data (np.array): Normal training data
            verbose (bool): Whether to print training information
        """
        if verbose:
            print(f"Training ensemble with {len(self.detectors)} detectors...")
            
        for i, detector in enumerate(self.detectors):
            if verbose:
                print(f"Training detector {i+1}/{len(self.detectors)}...")
            detector.fit(normal_data, verbose=False)
            
        self._is_fitted = True
        
        if verbose:
            print("✅ Ensemble training complete")
        
        return self
    
    def predict(self, data, method='majority'):
        """
        Predict using ensemble of detectors.
        
        Args:
            data (np.array): Data to predict
            method (str): Ensemble method ('majority', 'unanimous', 'any')
                         - 'majority': Majority vote
                         - 'unanimous': All detectors must agree on anomaly
                         - 'any': Any detector flags as anomaly
            
        Returns:
            np.array: Predictions (1=normal, -1=anomaly)
        """
        if not self._is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions.")
            
        predictions = np.array([detector.predict(data) for detector in self.detectors])
        
        if method == 'majority':
            # Majority vote (convert -1/1 to 0/1, take mean, convert back)
            votes = (predictions + 1) / 2  # Convert -1,1 to 0,1
            majority = np.mean(votes, axis=0) >= 0.5
            return np.where(majority, 1, -1)
            
        elif method == 'unanimous':
            # All must agree on anomaly
            all_anomaly = np.all(predictions == -1, axis=0)
            return np.where(all_anomaly, -1, 1)
            
        elif method == 'any':
            # Any detector flags as anomaly
            any_anomaly = np.any(predictions == -1, axis=0)
            return np.where(any_anomaly, -1, 1)
            
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    
    def get_anomaly_scores(self, data, method='mean'):
        """
        Get ensemble anomaly scores.
        
        Args:
            data (np.array): Data to score
            method (str): Score combination method ('mean', 'max', 'min')
            
        Returns:
            np.array: Combined anomaly scores
        """
        if not self._is_fitted:
            raise ValueError("Ensemble must be fitted before computing scores.")
            
        scores = np.array([detector.get_anomaly_scores(data) for detector in self.detectors])
        
        if method == 'mean':
            return np.mean(scores, axis=0)
        elif method == 'max':
            return np.max(scores, axis=0)
        elif method == 'min':
            return np.min(scores, axis=0)
        else:
            raise ValueError(f"Unknown score combination method: {method}")


def create_distance_detector_variants():
    """
    Create a set of distance-based detector variants with different parameters.
    
    Returns:
        list: List of DistanceAnomalyDetector instances with different configurations
    """
    variants = [
        DistanceAnomalyDetector(nu=0.01, gamma='scale', kernel='rbf'),  # Conservative
        DistanceAnomalyDetector(nu=0.05, gamma='scale', kernel='rbf'),  # Balanced
        DistanceAnomalyDetector(nu=0.1, gamma='scale', kernel='rbf'),   # Liberal
        DistanceAnomalyDetector(nu=0.05, gamma='auto', kernel='rbf'),   # Different gamma
    ]
    
    return variants
