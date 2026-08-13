"""
Anomaly Detection Utilities

This module provides utilities for anomaly detection using reconstruction errors,
threshold determination, and evaluation metrics.
"""

import numpy as np


class AnomalyDetector:
    """Anomaly detector using reconstruction error thresholding."""

    def __init__(self):
        self.threshold = None
        self.normal_errors = None
        self.anomaly_errors = None

    def compute_reconstruction_errors(self, autoencoder, normal_data, anomaly_data):
        """
        Compute reconstruction errors for normal and anomaly data.

        Args:
            autoencoder: Trained autoencoder model
            normal_data (np.array): Normal data
            anomaly_data (np.array): Anomaly data

        Returns:
            tuple: (normal_errors, anomaly_errors)
        """
        print("📊 Computing Reconstruction Errors")
        print("=" * 40)

        print("🔵 Computing reconstruction errors for normal data...")
        self.normal_errors = autoencoder.get_reconstruction_errors(normal_data)

        print("🔴 Computing reconstruction errors for anomaly data...")
        self.anomaly_errors = autoencoder.get_reconstruction_errors(anomaly_data)

        print("✅ Reconstruction errors computed:")
        print(f"   • Normal errors: {len(self.normal_errors)} samples")
        print(f"   • Anomaly errors: {len(self.anomaly_errors)} samples")
        print(
            f"   • Normal error range: [{np.min(self.normal_errors):.6f}, {np.max(self.normal_errors):.6f}]"
        )
        print(
            f"   • Anomaly error range: [{np.min(self.anomaly_errors):.6f}, {np.max(self.anomaly_errors):.6f}]"
        )

        return self.normal_errors, self.anomaly_errors

    def determine_threshold(self, method="percentile", percentile=95):
        """
        Determine anomaly threshold based on normal data reconstruction errors.

        Args:
            method (str): Method for threshold determination ('percentile' or 'std')
            percentile (float): Percentile for threshold (if method='percentile')

        Returns:
            float: Computed threshold
        """
        if self.normal_errors is None:
            raise ValueError(
                "Normal errors not computed. Call compute_reconstruction_errors first."
            )

        print(f"🎯 Determining Anomaly Threshold ({method})")
        print("=" * 40)

        if method == "percentile":
            self.threshold = np.percentile(self.normal_errors, percentile)
            print(
                f"📈 Threshold set at {percentile}th percentile: {self.threshold:.6f}"
            )
        elif method == "std":
            mean_error = np.mean(self.normal_errors)
            std_error = np.std(self.normal_errors)
            self.threshold = mean_error + 2 * std_error
            print(f"📈 Threshold set at mean + 2*std: {self.threshold:.6f}")
        else:
            raise ValueError("Method must be 'percentile' or 'std'")

        return self.threshold

    def evaluate_detection(self):
        """
        Evaluate anomaly detection performance.

        Returns:
            dict: Evaluation metrics
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call determine_threshold first.")

        print("📊 Evaluating Anomaly Detection Performance")
        print("=" * 45)

        # Classify samples based on threshold
        normal_predictions = (self.normal_errors > self.threshold).astype(int)
        anomaly_predictions = (self.anomaly_errors > self.threshold).astype(int)

        # Calculate metrics
        normal_correct = np.sum(
            normal_predictions == 0
        )  # Should be classified as normal (0)
        normal_total = len(normal_predictions)
        normal_accuracy = normal_correct / normal_total

        anomaly_correct = np.sum(
            anomaly_predictions == 1
        )  # Should be classified as anomaly (1)
        anomaly_total = len(anomaly_predictions)
        anomaly_accuracy = anomaly_correct / anomaly_total

        overall_accuracy = (normal_correct + anomaly_correct) / (
            normal_total + anomaly_total
        )

        metrics = {
            "threshold": self.threshold,
            "normal_accuracy": normal_accuracy,
            "anomaly_accuracy": anomaly_accuracy,
            "overall_accuracy": overall_accuracy,
            "normal_false_positives": normal_total - normal_correct,
            "anomaly_false_negatives": anomaly_total - anomaly_correct,
        }

        print("🎯 Detection Results:")
        print(f"   • Threshold: {self.threshold:.6f}")
        print(
            f"   • Normal data accuracy: {normal_accuracy:.3f} ({normal_correct}/{normal_total})"
        )
        print(
            f"   • Anomaly data accuracy: {anomaly_accuracy:.3f} ({anomaly_correct}/{anomaly_total})"
        )
        print(f"   • Overall accuracy: {overall_accuracy:.3f}")
        print(
            f"   • False positives (normal as anomaly): {normal_total - normal_correct}"
        )
        print(
            f"   • False negatives (anomaly as normal): {anomaly_total - anomaly_correct}"
        )

        return metrics
