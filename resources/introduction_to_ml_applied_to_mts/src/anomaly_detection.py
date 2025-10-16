"""
Anomaly Detection Utilities

This module provides utilities for anomaly detection using reconstruction errors,
threshold determination, and evaluation metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.manifold import TSNE
import seaborn as sns


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

        print(f"✅ Reconstruction errors computed:")
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

        print(f"🎯 Detection Results:")
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


class PerClassAnomalyVisualizer:
    """Visualization utilities for per-class anomaly detection analysis."""

    def __init__(self):
        self.default_colors = [
            "red",
            "orange",
            "brown",
            "pink",
            "purple",
            "olive",
            "cyan",
            "magenta",
        ]

    def create_comprehensive_analysis(
        self, normal_errors, anomaly_errors, per_class_metrics, threshold
    ):
        """
        Create comprehensive per-class anomaly detection visualization.

        Args:
            normal_errors: Array of normal reconstruction errors
            anomaly_errors: Array of anomaly reconstruction errors
            per_class_metrics: Dictionary of per-class metrics
            threshold: Detection threshold
        """
        print("📈 Creating Comprehensive Per-Class Visualization")
        print("=" * 55)

        # Prepare ROC data
        from sklearn.metrics import roc_curve, auc

        y_true = np.concatenate(
            [np.zeros(len(normal_errors)), np.ones(len(anomaly_errors))]
        )
        y_scores = np.concatenate([normal_errors, anomaly_errors])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        # 1. Overall error distributions
        self._plot_error_distributions(
            axes[0, 0], normal_errors, anomaly_errors, per_class_metrics, threshold
        )

        # 2. Per-class accuracy bar chart
        self._plot_class_accuracy(axes[0, 1], per_class_metrics)

        # 3. ROC curve
        self._plot_roc_curve(axes[0, 2], fpr, tpr, roc_auc)

        # 4. Error amplification comparison
        self._plot_error_amplification(axes[1, 0], normal_errors, per_class_metrics)

        # 5. Sample count distribution
        self._plot_sample_distribution(axes[1, 1], per_class_metrics)

        # 6. Error box plots by class
        self._plot_error_boxplots(
            axes[1, 2], normal_errors, anomaly_errors, per_class_metrics
        )

        plt.tight_layout()
        plt.show()

        return fig

    def _plot_error_distributions(
        self, ax, normal_errors, anomaly_errors, per_class_metrics, threshold
    ):
        """Plot error distributions for normal and per-class anomalies."""
        ax.hist(
            normal_errors,
            bins=30,
            alpha=0.6,
            label="Normal",
            color="blue",
            density=True,
        )

        for cls, metrics in per_class_metrics.items():
            # Get indices for this class (assuming they're stored)
            if "indices" in metrics:
                cls_errors = [anomaly_errors[i] for i in metrics["indices"]]
                ax.hist(
                    cls_errors,
                    bins=20,
                    alpha=0.6,
                    label=f"Class {cls}",
                    color=metrics["color"],
                    density=True,
                )

        ax.axvline(
            threshold,
            color="black",
            linestyle="--",
            label=f"Threshold: {threshold:.4f}",
        )
        ax.set_xlabel("Reconstruction Error")
        ax.set_ylabel("Density")
        ax.set_title("Error Distribution by Class")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_class_accuracy(self, ax, per_class_metrics):
        """Plot per-class detection accuracy."""
        classes = list(per_class_metrics.keys())
        accuracies = [metrics["accuracy"] for metrics in per_class_metrics.values()]
        colors = [metrics["color"] for metrics in per_class_metrics.values()]

        bars = ax.bar(classes, accuracies, color=colors, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Anomaly Class")
        ax.set_ylabel("Detection Accuracy")
        ax.set_title("Detection Accuracy by Class")
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    def _plot_roc_curve(self, ax, fpr, tpr, roc_auc):
        """Plot ROC curve."""
        ax.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
        )
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    def _plot_error_amplification(self, ax, normal_errors, per_class_metrics):
        """Plot error amplification factors."""
        normal_mean = np.mean(normal_errors)
        classes = list(per_class_metrics.keys())
        amplifications = [
            metrics["mean_error"] / normal_mean
            for metrics in per_class_metrics.values()
        ]
        colors = [metrics["color"] for metrics in per_class_metrics.values()]

        bars = ax.bar(
            classes, amplifications, color=colors, alpha=0.7, edgecolor="black"
        )
        ax.axhline(1.0, color="blue", linestyle="--", label="Normal baseline")
        ax.set_xlabel("Anomaly Class")
        ax.set_ylabel("Error Amplification Factor")
        ax.set_title("Error Amplification vs Normal")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bar, amp in zip(bars, amplifications):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{amp:.1f}x",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    def _plot_sample_distribution(self, ax, per_class_metrics):
        """Plot sample count distribution."""
        classes = list(per_class_metrics.keys())
        counts = [metrics["count"] for metrics in per_class_metrics.values()]
        colors = [metrics["color"] for metrics in per_class_metrics.values()]

        bars = ax.bar(classes, counts, color=colors, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Anomaly Class")
        ax.set_ylabel("Sample Count")
        ax.set_title("Sample Distribution by Class")
        ax.grid(True, alpha=0.3)

        # Add count labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(counts) * 0.01,
                f"{count}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    def _plot_error_boxplots(
        self, ax, normal_errors, anomaly_errors, per_class_metrics
    ):
        """Plot box plots of errors by class."""
        all_errors = [normal_errors]
        labels = ["Normal"]
        colors = ["blue"]

        for cls, metrics in per_class_metrics.items():
            if "indices" in metrics:
                cls_errors = [anomaly_errors[i] for i in metrics["indices"]]
                all_errors.append(cls_errors)
                labels.append(f"Class {cls}")
                colors.append(metrics["color"])

        bp = ax.boxplot(all_errors, labels=labels, patch_artist=True)

        # Color the boxes
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xlabel("Class")
        ax.set_ylabel("Reconstruction Error")
        ax.set_title("Error Distribution Box Plots")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
