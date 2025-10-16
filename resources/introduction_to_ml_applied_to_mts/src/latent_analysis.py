"""
Latent Space Analysis for Autoencoder Models

This module provides utilities for analyzing and visualizing autoencoder latent spaces,
including t-SNE visualization, per-class separation analysis, and correlation studies.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


class LatentSpaceAnalyzer:
    """Analyzer for autoencoder latent space with per-class visualization."""

    def __init__(self):
        self.encoder = None
        self.normal_latent = None
        self.anomaly_latent = None
        self.tsne_embedding = None

    def extract_latent_representations(
        self,
        autoencoder,
        normal_scaled,
        anomaly_scaled,
        max_normal_samples=800,
        max_anomaly_samples=400,
    ):
        """
        Extract latent representations from autoencoder.

        Args:
            autoencoder: Trained autoencoder model
            normal_scaled: Normal data samples
            anomaly_scaled: Anomaly data samples
            max_normal_samples: Maximum normal samples to process
            max_anomaly_samples: Maximum anomaly samples to process

        Returns:
            tuple: (normal_latent, anomaly_latent, encoder_model)
        """
        print("Extracting Latent Representations")
        print("=" * 35)

        # Create encoder model
        encoder_input = autoencoder.model.input
        encoder_output = autoencoder.model.get_layer("latent").output
        self.encoder = tf.keras.Model(encoder_input, encoder_output, name="encoder")

        # Sample data for efficiency
        normal_sample_size = min(max_normal_samples, len(normal_scaled))
        anomaly_sample_size = min(max_anomaly_samples, len(anomaly_scaled))

        print(
            f"Processing {normal_sample_size} normal and {anomaly_sample_size} anomaly samples..."
        )

        # Extract latent representations
        self.normal_latent = self.encoder.predict(
            normal_scaled[:normal_sample_size], verbose=0
        )
        self.anomaly_latent = self.encoder.predict(
            anomaly_scaled[:anomaly_sample_size], verbose=0
        )

        print(f"✅ Latent extraction complete:")
        print(f"   • Normal latent shape: {self.normal_latent.shape}")
        print(f"   • Anomaly latent shape: {self.anomaly_latent.shape}")
        print(f"   • Latent dimension: {self.normal_latent.shape[1]}")

        return self.normal_latent, self.anomaly_latent, self.encoder

    def compute_latent_statistics(
        self, per_class_metrics, anomaly_classes, max_samples=400
    ):
        """
        Compute per-class latent space statistics.

        Args:
            per_class_metrics: Dictionary of per-class metrics
            anomaly_classes: List of anomaly class labels
            max_samples: Maximum samples used for latent extraction

        Returns:
            dict: Per-class latent statistics
        """
        print("Computing Per-Class Latent Statistics")
        print("=" * 40)

        normal_mean = np.mean(self.normal_latent, axis=0)
        anomaly_mean = np.mean(self.anomaly_latent, axis=0)
        latent_separation = np.linalg.norm(normal_mean - anomaly_mean)

        print(f"Overall Statistics:")
        print(f"   • Normal latent mean magnitude: {np.linalg.norm(normal_mean):.3f}")
        print(f"   • Anomaly latent mean magnitude: {np.linalg.norm(anomaly_mean):.3f}")
        print(f"   • Mean separation distance: {latent_separation:.3f}")

        # Per-class statistics
        class_latent_stats = {}
        anomaly_class_subset = anomaly_classes[:max_samples]

        print(f"\nPer-Class Statistics:")
        for cls in sorted(per_class_metrics.keys()):
            # Find indices for this class in our sample
            cls_indices = [
                i
                for i, label in enumerate(anomaly_class_subset)
                if str(int(label) if isinstance(label, (np.integer, int)) else label)
                == cls
            ]

            if cls_indices:
                cls_latent = self.anomaly_latent[cls_indices]
                cls_mean = np.mean(cls_latent, axis=0)
                cls_separation = np.linalg.norm(normal_mean - cls_mean)

                class_latent_stats[cls] = {
                    "mean": cls_mean,
                    "separation": cls_separation,
                    "latent_data": cls_latent,
                    "sample_count": len(cls_latent),
                }

                print(
                    f"   • Class {cls}: separation = {cls_separation:.3f}, samples = {len(cls_latent)}"
                )

        return class_latent_stats

    def compute_tsne_embedding(self, perplexity=30, random_state=42):
        """
        Compute t-SNE embedding of latent representations.

        Args:
            perplexity: t-SNE perplexity parameter
            random_state: Random state for reproducibility

        Returns:
            tuple: (normal_tsne, anomaly_tsne, full_embedding)
        """
        print("Computing t-SNE Embedding")
        print("=" * 25)

        # Combine and standardize latent representations
        all_latent = np.vstack([self.normal_latent, self.anomaly_latent])
        scaler = StandardScaler()
        all_latent_scaled = scaler.fit_transform(all_latent)

        # Apply t-SNE
        tsne_perplexity = min(perplexity, len(all_latent) // 4)
        print(f"Running t-SNE with perplexity={tsne_perplexity}...")

        tsne = TSNE(
            n_components=2,
            perplexity=tsne_perplexity,
            random_state=random_state,
            n_iter=1000,
            learning_rate="auto",
            init="pca",
        )

        self.tsne_embedding = tsne.fit_transform(all_latent_scaled)

        # Split results
        normal_tsne = self.tsne_embedding[: len(self.normal_latent)]
        anomaly_tsne = self.tsne_embedding[len(self.normal_latent) :]

        print(f"✅ t-SNE embedding complete")
        return normal_tsne, anomaly_tsne, self.tsne_embedding

    def create_comprehensive_latent_visualization(
        self,
        per_class_metrics,
        class_latent_stats,
        normal_tsne,
        anomaly_tsne,
        anomaly_classes,
    ):
        """
        Create comprehensive latent space visualization.

        Args:
            per_class_metrics: Dictionary of per-class metrics
            class_latent_stats: Per-class latent statistics
            normal_tsne: t-SNE embedding for normal data
            anomaly_tsne: t-SNE embedding for anomaly data
            anomaly_classes: Anomaly class labels

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        print("Creating Comprehensive Latent Visualization")
        print("=" * 45)

        fig, axes = plt.subplots(2, 4, figsize=(22, 12))

        # 1. t-SNE visualization with per-class colors
        self._plot_tsne_perclass(
            axes[0, 0], normal_tsne, anomaly_tsne, per_class_metrics, anomaly_classes
        )

        # 2. Complete latent space (first 2 dimensions)
        self._plot_complete_latent_space(
            axes[0, 1], class_latent_stats, per_class_metrics
        )

        # 3. Latent dimension variance analysis
        self._plot_latent_variance(axes[0, 2])

        # 4. Per-class separation in latent space
        self._plot_class_separation(axes[0, 3], class_latent_stats, per_class_metrics)

        # 5. Mean latent values per dimension comparison
        self._plot_mean_latent_comparison(
            axes[1, 0], class_latent_stats, per_class_metrics
        )

        # 6. Latent space quality vs detection accuracy
        self._plot_separation_vs_accuracy(
            axes[1, 1], class_latent_stats, per_class_metrics
        )

        # 7. Most discriminative dimension distribution
        self._plot_discriminative_dimension(
            axes[1, 2], class_latent_stats, per_class_metrics
        )

        # 8. t-SNE vs Detection Performance correlation
        self._plot_tsne_vs_performance(
            axes[1, 3], normal_tsne, anomaly_tsne, per_class_metrics, anomaly_classes
        )

        plt.tight_layout()
        plt.show()

        return fig

    def _plot_tsne_perclass(
        self, ax, normal_tsne, anomaly_tsne, per_class_metrics, anomaly_classes
    ):
        """Plot t-SNE with per-class colors."""
        # Plot normal data
        ax.scatter(
            normal_tsne[:, 0],
            normal_tsne[:, 1],
            alpha=0.6,
            label="Normal",
            color="blue",
            s=25,
            edgecolors="white",
            linewidth=0.5,
        )

        # Plot each anomaly class
        legend_elements = [mpatches.Patch(color="blue", label="Normal")]

        for cls in sorted(per_class_metrics.keys()):
            cls_mask = np.array(
                [
                    str(int(label) if isinstance(label, (np.integer, int)) else label)
                    == cls
                    for label in anomaly_classes[: len(anomaly_tsne)]
                ]
            )
            if np.any(cls_mask):
                cls_tsne = anomaly_tsne[cls_mask]
                color = per_class_metrics[cls]["color"]

                ax.scatter(
                    cls_tsne[:, 0],
                    cls_tsne[:, 1],
                    alpha=0.7,
                    color=color,
                    s=30,
                    edgecolors="white",
                    linewidth=0.5,
                )

                legend_elements.append(
                    mpatches.Patch(color=color, label=f"Class {cls}")
                )

        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_title("t-SNE: Latent Space (Per-Class)")
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

    def _plot_complete_latent_space(self, ax, class_latent_stats, per_class_metrics):
        """Plot complete latent space (first 2 dimensions)."""
        if self.normal_latent.shape[1] >= 2:
            ax.scatter(
                self.normal_latent[:, 0],
                self.normal_latent[:, 1],
                alpha=0.6,
                label="Normal",
                color="blue",
                s=20,
            )

            for cls in sorted(per_class_metrics.keys()):
                if cls in class_latent_stats:
                    cls_latent = class_latent_stats[cls]["latent_data"]
                    color = per_class_metrics[cls]["color"]
                    ax.scatter(
                        cls_latent[:, 0],
                        cls_latent[:, 1],
                        alpha=0.7,
                        color=color,
                        s=25,
                        label=f"Class {cls}",
                    )

            ax.set_xlabel("Latent Dimension 1")
            ax.set_ylabel("Latent Dimension 2")
            ax.set_title("Complete Latent Space (First 2 Dims)")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)

    def _plot_latent_variance(self, ax):
        """Plot variance per latent dimension."""
        all_latent = np.vstack([self.normal_latent, self.anomaly_latent])
        latent_variances = np.var(all_latent, axis=0)
        ax.bar(range(len(latent_variances)), latent_variances, alpha=0.7, color="green")
        ax.set_xlabel("Latent Dimension")
        ax.set_ylabel("Variance")
        ax.set_title("Variance per Latent Dimension")
        ax.grid(True, alpha=0.3)

    def _plot_class_separation(self, ax, class_latent_stats, per_class_metrics):
        """Plot per-class separation in latent space."""
        classes = sorted(class_latent_stats.keys())
        separations = [class_latent_stats[cls]["separation"] for cls in classes]
        colors = [per_class_metrics[cls]["color"] for cls in classes]

        bars = ax.bar(classes, separations, color=colors, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Anomaly Class")
        ax.set_ylabel("Separation from Normal")
        ax.set_title("Latent Space Separation by Class")
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bar, sep in zip(bars, separations):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(separations) * 0.01,
                f"{sep:.2f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    def _plot_mean_latent_comparison(self, ax, class_latent_stats, per_class_metrics):
        """Plot mean latent values per dimension comparison."""
        normal_mean = np.mean(self.normal_latent, axis=0)
        dims = range(self.normal_latent.shape[1])

        ax.plot(
            dims,
            normal_mean,
            "o-",
            color="blue",
            label="Normal Mean",
            linewidth=2,
            markersize=4,
        )

        for cls in sorted(class_latent_stats.keys()):
            if cls in class_latent_stats:
                cls_mean = class_latent_stats[cls]["mean"]
                color = per_class_metrics[cls]["color"]
                ax.plot(
                    dims,
                    cls_mean,
                    "s-",
                    color=color,
                    label=f"Class {cls} Mean",
                    linewidth=1.5,
                    markersize=3,
                )

        ax.set_xlabel("Latent Dimension")
        ax.set_ylabel("Mean Value")
        ax.set_title("Mean Values per Dimension")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

    def _plot_separation_vs_accuracy(self, ax, class_latent_stats, per_class_metrics):
        """Plot latent space quality vs detection accuracy."""
        classes = sorted(class_latent_stats.keys())
        separations = [class_latent_stats[cls]["separation"] for cls in classes]
        accuracies = [per_class_metrics[cls]["accuracy"] for cls in classes]
        colors = [per_class_metrics[cls]["color"] for cls in classes]

        ax.scatter(
            separations, accuracies, c=colors, s=100, alpha=0.7, edgecolors="black"
        )

        # Add class labels
        for i, cls in enumerate(classes):
            ax.annotate(
                f"Class {cls}",
                (separations[i], accuracies[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

        ax.set_xlabel("Latent Space Separation")
        ax.set_ylabel("Detection Accuracy")
        ax.set_title("Separation vs Detection Performance")
        ax.grid(True, alpha=0.3)

    def _plot_discriminative_dimension(self, ax, class_latent_stats, per_class_metrics):
        """Plot distribution of most discriminative dimension."""
        normal_mean = np.mean(self.normal_latent, axis=0)
        anomaly_mean = np.mean(self.anomaly_latent, axis=0)

        # Find most discriminative dimension
        dim_separations = np.abs(normal_mean - anomaly_mean)
        most_disc_dim = np.argmax(dim_separations)

        ax.hist(
            self.normal_latent[:, most_disc_dim],
            bins=20,
            alpha=0.5,
            label="Normal",
            color="blue",
            density=True,
        )

        for cls in sorted(class_latent_stats.keys()):
            if cls in class_latent_stats:
                cls_latent = class_latent_stats[cls]["latent_data"]
                color = per_class_metrics[cls]["color"]
                ax.hist(
                    cls_latent[:, most_disc_dim],
                    bins=15,
                    alpha=0.6,
                    label=f"Class {cls}",
                    color=color,
                    density=True,
                )

        ax.set_xlabel(f"Latent Value (Dim {most_disc_dim})")
        ax.set_ylabel("Density")
        ax.set_title(f"Most Discriminative Dimension ({most_disc_dim})")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

    def _plot_tsne_vs_performance(
        self, ax, normal_tsne, anomaly_tsne, per_class_metrics, anomaly_classes
    ):
        """Plot t-SNE vs detection performance correlation."""
        # Calculate t-SNE cluster separation for each class
        normal_center = np.mean(normal_tsne, axis=0)
        classes = sorted(per_class_metrics.keys())
        tsne_separations = []
        accuracies = [per_class_metrics[cls]["accuracy"] for cls in classes]
        colors = [per_class_metrics[cls]["color"] for cls in classes]

        for cls in classes:
            cls_mask = np.array(
                [
                    str(int(label) if isinstance(label, (np.integer, int)) else label)
                    == cls
                    for label in anomaly_classes[: len(anomaly_tsne)]
                ]
            )
            if np.any(cls_mask):
                cls_tsne_points = anomaly_tsne[cls_mask]
                cls_center = np.mean(cls_tsne_points, axis=0)
                tsne_sep = np.linalg.norm(normal_center - cls_center)
                tsne_separations.append(tsne_sep)
            else:
                tsne_separations.append(0)

        ax.scatter(
            tsne_separations, accuracies, c=colors, s=100, alpha=0.7, edgecolors="black"
        )

        for i, cls in enumerate(classes):
            ax.annotate(
                f"Class {cls}",
                (tsne_separations[i], accuracies[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

        ax.set_xlabel("t-SNE Separation from Normal")
        ax.set_ylabel("Detection Accuracy")
        ax.set_title("t-SNE Separation vs Detection")
        ax.grid(True, alpha=0.3)
