"""
Data Preparation Utilities for Unsupervised Learning

This module provides utilities for preparing time series data for LSTM autoencoders,
including sampling, validation, and conversion functions.
"""

import numpy as np
import random
import time


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
        if anomaly_classes is not None and len(anomaly_windows) > self.max_anomaly_samples:
            print(f"📊 Balanced sampling {self.max_anomaly_samples} anomaly windows from {len(anomaly_windows)} available...")
            
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
                sampled_anomaly_classes.extend([anomaly_classes[i] for i in cls_sampled])
                
                print(f"   • Class {cls}: {len(cls_sampled)} samples selected from {len(cls_indices)} available")
            
            sampled_anomaly_windows = [anomaly_windows[i] for i in sampled_indices]
            
        else:
            print(f"📊 Using all {len(anomaly_windows)} anomaly windows...")
            sampled_anomaly_windows = anomaly_windows
            sampled_anomaly_classes = anomaly_classes if anomaly_classes is not None else [None] * len(anomaly_windows)

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

    def prepare_full_pipeline(self, normal_windows, anomaly_windows, anomaly_classes=None):
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
        if sampled_anomaly_classes is not None and len(sampled_anomaly_classes) != anomaly_scaled.shape[0]:
            print(f"   ⚠️ Adjusting anomaly classes length from {len(sampled_anomaly_classes)} to {anomaly_scaled.shape[0]}")
            sampled_anomaly_classes = sampled_anomaly_classes[:anomaly_scaled.shape[0]]

        return normal_scaled, anomaly_scaled, data_info, sampled_anomaly_classes
