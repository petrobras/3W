"""
Visualization Module for 3W Dataset

Plots that the 3W Toolkit does not offer: the dataset overview of notebook 1,
the t-SNE/UMAP projections of notebook 2 and the anomaly detection panels of
notebook 3. Plots of time series and correlations use the toolkit directly.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ThreeWToolkit.metrics import roc_auc_score


def plot_dataset_overview(counts: pd.DataFrame) -> None:
    """Files available and loaded per class, and the real/non-real mix loaded.

    Args:
        counts: one row per class folder, with the columns `real_available`,
            `other_available`, `real_loaded` and `other_loaded`.
    """
    fig, (ax_bars, ax_pie) = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(len(counts))
    ax_bars.bar(
        x - 0.2, counts.real_available + counts.other_available, 0.4, label="Available"
    )
    ax_bars.bar(x + 0.2, counts.real_loaded, 0.4, label="Loaded (real)")
    ax_bars.bar(
        x + 0.2,
        counts.other_loaded,
        0.4,
        bottom=counts.real_loaded,
        label="Loaded (simulated/drawn)",
    )
    ax_bars.set_xticks(x, counts.index)
    ax_bars.set(
        title="Files per class: available vs loaded", xlabel="Class", ylabel="Files"
    )
    ax_bars.legend()
    ax_bars.grid(True, alpha=0.3)
    loaded = [counts.real_loaded.sum(), counts.other_loaded.sum()]
    ax_pie.pie(
        loaded, labels=["Real", "Simulated/drawn"], autopct="%1.1f%%", startangle=90
    )
    ax_pie.set_title("Provenance of the loaded files")
    plt.tight_layout()
    plt.show()


class DimensionalityReductionVisualizer:
    """
    A class for dimensionality reduction visualization of the 3W dataset.

    Provides methods for t-SNE and UMAP analysis with multiple configurations.
    """

    def __init__(self):
        """Initialize the dimensionality reduction visualizer."""
        self.class_colors = self._generate_class_colors()

    def _generate_class_colors(self):
        """Generate distinct, vibrant colors for each class."""
        # Define high-contrast, vibrant colors for better visibility
        vibrant_colors = [
            "#FF0000",  # Bright Red (Class 0)
            "#00FF00",  # Bright Green (Class 1)
            "#0000FF",  # Bright Blue (Class 2)
            "#FF8000",  # Bright Orange (Class 3)
            "#8000FF",  # Bright Purple (Class 4)
            "#00FFFF",  # Bright Cyan (Class 5)
            "#FF0080",  # Bright Magenta (Class 6)
            "#80FF00",  # Bright Lime (Class 7)
            "#0080FF",  # Bright Sky Blue (Class 8)
            "#FFFF00",  # Bright Yellow (Class 9)
        ]

        return {i: vibrant_colors[i % len(vibrant_colors)] for i in range(10)}

    def load_windowed_data(self, persistence, config, fold_number=None):
        """
        Load windowed data from processed files.

        Args:
            persistence: DataPersistence instance
            config: Configuration module
            fold_number (int, optional): Specific fold to load. If None, loads first available.

        Returns:
            tuple: (test_dfs, test_classes, metadata)
        """
        print("📊 Loading windowed data...")

        # Check windowed directory
        windowed_dir = os.path.join(persistence.cv_splits_dir, "windowed")
        print(f"   • Windowed directory: {windowed_dir}")

        if not os.path.exists(windowed_dir):
            raise FileNotFoundError(
                "Run '1_data_treatment.ipynb' first to generate windowed data"
            )

        # Find fold directories
        fold_dirs = [
            d
            for d in os.listdir(windowed_dir)
            if d.startswith("fold_") and os.path.isdir(os.path.join(windowed_dir, d))
        ]
        fold_dirs.sort()

        if not fold_dirs:
            raise FileNotFoundError("No fold directories found in windowed data.")

        # Select fold
        if fold_number is not None:
            target_fold = f"fold_{fold_number}"
            if target_fold not in fold_dirs:
                raise ValueError(
                    f"Fold {fold_number} not found. Available: {fold_dirs}"
                )
            first_fold_dir = target_fold
        else:
            first_fold_dir = fold_dirs[0]

        fold_path = os.path.join(windowed_dir, first_fold_dir)
        fold_num = first_fold_dir.replace("fold_", "")

        print(f"   • Loading from {first_fold_dir}")

        # Load data with fallback options
        data_files = {
            "pickle": os.path.join(fold_path, f"test_windowed.{config.SAVE_FORMAT}"),
            "parquet": os.path.join(fold_path, "test_windowed.parquet"),
        }

        test_dfs, test_classes = None, None
        loaded_format = None

        for format_name, file_path in data_files.items():
            if os.path.exists(file_path):
                try:
                    if format_name == "pickle":
                        test_dfs, test_classes = persistence._load_dataframes(
                            file_path, config.SAVE_FORMAT
                        )
                    else:
                        test_dfs, test_classes = persistence._load_from_parquet(
                            file_path
                        )

                    loaded_format = format_name
                    break

                except Exception as e:
                    print(f"   ⚠️ Failed to load {format_name}: {str(e)[:50]}...")
                    continue

        if test_dfs is None:
            raise FileNotFoundError(
                f"No compatible test data file found in: {fold_path}"
            )

        metadata = {
            "fold_number": fold_num,
            "format": loaded_format,
            "windows_count": len(test_dfs),
            "fold_path": fold_path,
        }

        print(f"   ✅ Loaded {len(test_dfs)} windows from fold {fold_num}")

        return test_dfs, test_classes, metadata

    def extract_and_map_classes(self, test_dfs, test_classes=None):
        """Extract and map class labels from windowed data."""
        print("🏷️ Processing class labels...")

        # Extract classes
        window_classes = []
        for i, window_df in enumerate(test_dfs):
            if "class" in window_df.columns:
                window_class = window_df["class"].iloc[-1]
                # if window_df["class"].nunique() > 1:
                #     print(f"   ⚠️ Warning: Inconsistent classes in window {i}")
                window_classes.append(window_class)
            else:
                if test_classes and i < len(test_classes):
                    window_classes.append(test_classes[i])
                else:
                    window_classes.append(0)

        print(
            f"🏷️ Original class distribution: {dict(zip(*np.unique(window_classes, return_counts=True)))}"
        )

        # Map transient classes
        transient_mapping = {101: 1, 102: 2, 105: 5, 106: 6, 107: 7, 108: 8, 109: 9}
        mapped_classes = []
        transient_count = 0

        for cls in window_classes:
            if cls in transient_mapping:
                mapped_classes.append(transient_mapping[cls])
                transient_count += 1
            else:
                mapped_classes.append(cls)

        print(f"🔄 Mapped {transient_count} points with transient classes")
        print(
            f"🏷️ Mapped class distribution: {dict(zip(*np.unique(mapped_classes, return_counts=True)))}"
        )

        return mapped_classes

    def intelligent_sampling_for_visualization(self, test_dfs, mapped_classes, config):
        """Perform intelligent sampling for visualization."""
        print("🎯 Performing intelligent sampling...")

        # Normal operation plus the selected anomaly classes
        target_classes = [0] + list(config.SELECTED_ANOMALY_CLASSES)
        max_samples_per_class = config.VISUALIZATION_MAX_SAMPLES

        selected_indices = []
        selected_classes = []
        sampling_summary = {}

        np.random.seed(getattr(config, "VISUALIZATION_RANDOM_SEED", 42))

        for target_class in target_classes:
            class_indices = [
                i for i, cls in enumerate(mapped_classes) if cls == target_class
            ]

            if len(class_indices) > 0:
                n_samples = min(max_samples_per_class, len(class_indices))
                sampled_indices = np.random.choice(
                    class_indices, size=n_samples, replace=False
                )
                selected_indices.extend(sampled_indices)
                selected_classes.extend([target_class] * len(sampled_indices))
                sampling_summary[target_class] = {
                    "available": len(class_indices),
                    "sampled": n_samples,
                }

        # Filter data
        selected_test_dfs = [test_dfs[i] for i in selected_indices]

        print(f"   • Selected {len(selected_indices)} windows")
        print(f"   • Sampling summary: {sampling_summary}")

        return selected_test_dfs, selected_classes, sampling_summary

    def prepare_features_for_visualization(self, selected_test_dfs, selected_classes):
        """Extract and standardize features for dimensionality reduction."""
        print("🔄 Preparing features for visualization...")

        # Extract features
        flattened_windows = []
        feature_columns = None

        for i, window_df in enumerate(selected_test_dfs):
            if feature_columns is None:
                feature_columns = [col for col in window_df.columns if col != "class"]
                print(
                    f"   • Features: {len(feature_columns)} sensors × {window_df.shape[0]} timesteps"
                )

            try:
                flattened = window_df[feature_columns].values.flatten()
                flattened_windows.append(flattened)
            except Exception as e:
                print(f"   ⚠️ Error processing window {i}: {e}")
                continue

        if not flattened_windows:
            raise ValueError("No valid windows processed")

        # Convert to arrays
        X = np.array(flattened_windows)
        y_labels = np.array(selected_classes[: len(flattened_windows)])

        # Validate shapes
        if X.shape[0] != y_labels.shape[0]:
            print("⚠️ Shape mismatch, fixing...")
            min_samples = min(X.shape[0], y_labels.shape[0])
            X = X[:min_samples]
            y_labels = y_labels[:min_samples]

        # Standardize features
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        print(f"   • Feature matrix: {X_scaled.shape}")
        print(f"   • Memory usage: {X_scaled.nbytes / 1024 / 1024:.2f} MB")

        return X_scaled, y_labels, scaler

    def run_tsne_analysis(self, X_scaled, y_labels, configs=None):
        """Run t-SNE with multiple configurations."""
        from sklearn.manifold import TSNE

        if configs is None:
            # Use default configs from config module if available
            import src.config as config

            configs = getattr(
                config,
                "TSNE_CONFIGS",
                [
                    {"perplexity": 30, "learning_rate": 200, "title": "Standard t-SNE"},
                    {"perplexity": 10, "learning_rate": 100, "title": "Low Perplexity"},
                    {
                        "perplexity": 50,
                        "learning_rate": 300,
                        "title": "High Perplexity",
                    },
                    {"perplexity": 30, "learning_rate": 500, "title": "Fast Learning"},
                ],
            )

        print("🔮 Running t-SNE analysis...")
        results = []

        for i, config in enumerate(configs):
            print(
                f"   • t-SNE #{i + 1}: perplexity={config['perplexity']}, lr={config['learning_rate']}..."
            )

            try:
                start_time = time.time()

                tsne = TSNE(
                    n_components=2,
                    perplexity=min(config["perplexity"], X_scaled.shape[0] - 1),
                    learning_rate=config["learning_rate"],
                    random_state=42,
                    max_iter=1000,
                    init="pca",
                )

                X_tsne = tsne.fit_transform(X_scaled)
                elapsed_time = time.time() - start_time

                results.append(
                    {
                        "embedding": X_tsne,
                        "config": config,
                        "time": elapsed_time,
                        "reducer": tsne,
                    }
                )

                print(f"     ✅ Completed in {elapsed_time:.1f}s")

            except Exception as e:
                print(f"     ❌ Failed: {e}")
                results.append(
                    {"embedding": None, "config": config, "time": None, "error": str(e)}
                )

        return results

    def run_umap_analysis(self, X_scaled, y_labels, configs=None):
        """Run UMAP with multiple configurations."""
        try:
            import umap
        except ImportError:
            print(
                "❌ UMAP not available. Install with: uv pip install -e '../../.[resources]'"
            )
            return None

        if configs is None:
            # Use default configs from config module if available
            import src.config as config

            configs = getattr(
                config,
                "UMAP_CONFIGS",
                [
                    {
                        "n_neighbors": 15,
                        "min_dist": 0.1,
                        "metric": "euclidean",
                        "title": "Standard UMAP",
                    },
                    {
                        "n_neighbors": 5,
                        "min_dist": 0.0,
                        "metric": "euclidean",
                        "title": "Tight Clusters",
                    },
                    {
                        "n_neighbors": 50,
                        "min_dist": 0.5,
                        "metric": "cosine",
                        "title": "Global Structure",
                    },
                    {
                        "n_neighbors": 30,
                        "min_dist": 0.25,
                        "metric": "manhattan",
                        "title": "Robust Config",
                    },
                ],
            )

        print("🚀 Running UMAP analysis...")
        results = []

        for i, config in enumerate(configs):
            print(
                f"   • UMAP #{i + 1}: neighbors={config['n_neighbors']}, dist={config['min_dist']}..."
            )

            try:
                start_time = time.time()

                # Setting random_state (reproducibility) makes UMAP run single-threaded
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="n_jobs value 1")
                    reducer = umap.UMAP(
                        n_components=2,
                        n_neighbors=min(config["n_neighbors"], X_scaled.shape[0] - 1),
                        min_dist=config["min_dist"],
                        metric=config["metric"],
                        random_state=42,
                        n_epochs=200,
                    )

                    X_umap = reducer.fit_transform(X_scaled)
                elapsed_time = time.time() - start_time

                results.append(
                    {
                        "embedding": X_umap,
                        "config": config,
                        "time": elapsed_time,
                        "reducer": reducer,
                    }
                )

                print(f"     ✅ Completed in {elapsed_time:.1f}s")

            except Exception as e:
                print(f"     ❌ Failed: {e}")
                results.append(
                    {"embedding": None, "config": config, "time": None, "error": str(e)}
                )

        return results

    def plot_dimensionality_reduction_results(
        self, results, y_labels, method_name="Dimensionality Reduction"
    ):
        """Plot results from dimensionality reduction analysis."""
        from sklearn.metrics import pairwise_distances

        n_configs = len(results)
        cols = 2
        rows = (n_configs + 1) // 2

        fig, axes = plt.subplots(rows, cols, figsize=(16, 8 * rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()

        unique_classes = np.unique(y_labels)

        for idx, result in enumerate(results):
            if idx >= len(axes):
                break

            ax = axes[idx]

            if result["embedding"] is not None:
                embedding = result["embedding"]
                config = result["config"]

                # Calculate separation quality
                embedding_distances = pairwise_distances(embedding)
                avg_intra_class = 0
                avg_inter_class = 0

                for class_label in unique_classes:
                    class_mask = y_labels == class_label
                    if np.sum(class_mask) > 1:
                        intra_distances = embedding_distances[class_mask][:, class_mask]
                        avg_intra_class += np.mean(intra_distances)

                        inter_distances = embedding_distances[class_mask][
                            :, ~class_mask
                        ]
                        if inter_distances.size > 0:
                            avg_inter_class += np.mean(inter_distances)

                separation_ratio = avg_inter_class / max(avg_intra_class, 1e-10)

                # Plot each class with improved visibility
                for class_label in unique_classes:
                    mask = y_labels == class_label
                    class_points = embedding[mask]

                    ax.scatter(
                        class_points[:, 0],
                        class_points[:, 1],
                        c=[self.class_colors[class_label]],
                        label=f"Class {class_label} (n={np.sum(mask)})",
                        alpha=0.85,  # Increased alpha for better visibility
                        s=80,  # Larger markers for better visibility
                        edgecolors="black",  # Black edges for better contrast
                        linewidth=0.8,  # Thicker edges
                    )

                # Styling
                ax.set_title(
                    f"{config['title']}\n(Separation: {separation_ratio:.2f})",
                    fontsize=11,
                    fontweight="bold",
                )
                ax.set_xlabel(f"{method_name} Component 1")
                ax.set_ylabel(f"{method_name} Component 2")
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
                ax.grid(True, alpha=0.3)

                # Performance annotation
                if result["time"]:
                    ax.text(
                        0.02,
                        0.98,
                        f"{result['time']:.1f}s",
                        transform=ax.transAxes,
                        fontsize=9,
                        verticalalignment="top",
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="white", alpha=0.8
                        ),
                    )
            else:
                # Handle failed reductions
                ax.text(
                    0.5,
                    0.5,
                    f"{method_name} Failed\n{result.get('error', 'Unknown error')}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    fontsize=12,
                    color="red",
                )
                ax.set_title(f"{result['config']['title']} - Failed", fontsize=11)

        # Hide unused subplots
        for idx in range(len(results), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(
            f"{method_name} Parameter Comparison - 3W Dataset",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        plt.show()

        return fig

    def print_dimensionality_reduction_summary(self, results, method_name):
        """Print performance summary for reduction results."""
        print(f"\n📊 {method_name} Analysis Results:")

        successful_results = [r for r in results if r["embedding"] is not None]

        if successful_results:
            fastest_config = min(successful_results, key=lambda x: x["time"])

            print("⚡ Configuration Performance:")
            for i, result in enumerate(results):
                if result["embedding"] is not None:
                    status = (
                        " ⚡ Fastest"
                        if result["time"] == fastest_config["time"]
                        else ""
                    )
                    print(f"   • Config {i + 1}: {result['time']:.1f}s{status}")
                else:
                    print(f"   • Config {i + 1}: Failed")
        else:
            print("   ❌ All configurations failed")


class AnomalyDetectionVisualizer:
    """
    Visualizer for anomaly detection results using autoencoders.
    """

    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        self.colors = {
            "normal": "#1f77b4",  # blue
            "anomaly": "#d62728",  # red
            "threshold": "#2ca02c",  # green
            "threshold_alt": "#ff7f0e",  # orange
        }

    def plot_per_fold_performance(self, fold_results):
        """
        Plot per-fold performance results.

        Args:
            fold_results (dict): Dictionary with fold results
        """
        valid_results = [r for r in fold_results.values() if r is not None]

        if not valid_results:
            print("No valid results to plot")
            return

        # Calculate statistics
        avg_normal_acc = np.mean([r["normal_accuracy"] for r in valid_results])
        avg_anomaly_acc = np.mean([r["anomaly_accuracy"] for r in valid_results])
        avg_overall_acc = np.mean([r["overall_accuracy"] for r in valid_results])

        std_normal_acc = np.std([r["normal_accuracy"] for r in valid_results])
        std_anomaly_acc = np.std([r["anomaly_accuracy"] for r in valid_results])
        std_overall_acc = np.std([r["overall_accuracy"] for r in valid_results])

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Per-fold performance
        fold_names = list(fold_results.keys())
        normal_accs = [
            fold_results[f]["normal_accuracy"] if fold_results[f] else 0
            for f in fold_names
        ]
        anomaly_accs = [
            fold_results[f]["anomaly_accuracy"] if fold_results[f] else 0
            for f in fold_names
        ]

        x = np.arange(len(fold_names))
        width = 0.35

        ax1.bar(
            x - width / 2,
            normal_accs,
            width,
            label="Normal",
            alpha=0.7,
            color=self.colors["normal"],
        )
        ax1.bar(
            x + width / 2,
            anomaly_accs,
            width,
            label="Anomaly",
            alpha=0.7,
            color=self.colors["anomaly"],
        )
        ax1.set_xlabel("Fold")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Per-Fold Detection Accuracy")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f.replace("fold_", "F") for f in fold_names])
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Overall statistics
        categories = ["Normal", "Anomaly", "Overall"]
        means = [avg_normal_acc, avg_anomaly_acc, avg_overall_acc]
        stds = [std_normal_acc, std_anomaly_acc, std_overall_acc]

        ax2.bar(
            categories,
            means,
            yerr=stds,
            capsize=5,
            alpha=0.7,
            color=[self.colors["normal"], self.colors["anomaly"], "green"],
        )
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Average Performance ± Std")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_reconstruction_errors(
        self,
        normal_errors,
        anomaly_errors,
        anomaly_classes=None,
        threshold_percentile=95,
        threshold_std_multiplier=2,
    ):
        """
        Plot reconstruction error distributions with thresholds.

        Args:
            normal_errors (np.array): Reconstruction errors for normal data
            anomaly_errors (np.array): Reconstruction errors for anomaly data
            anomaly_classes (list): Class labels for anomaly data
            threshold_percentile (int): Percentile for threshold calculation
            threshold_std_multiplier (float): Standard deviation multiplier for threshold
        """
        # Calculate thresholds
        threshold_pct = np.percentile(normal_errors, threshold_percentile)
        threshold_std = np.mean(normal_errors) + threshold_std_multiplier * np.std(
            normal_errors
        )

        # Create figure
        if anomaly_classes is not None and len(set(anomaly_classes)) > 1:
            fig, axes = plt.subplots(2, 2, figsize=self.figsize)
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes = [axes[0], axes[1]]

        # Plot 1: Overall distribution
        ax = axes[0]
        ax.hist(
            normal_errors,
            bins=50,
            alpha=0.7,
            label="Normal",
            color=self.colors["normal"],
            density=True,
        )
        ax.hist(
            anomaly_errors,
            bins=50,
            alpha=0.7,
            label="Anomaly",
            color=self.colors["anomaly"],
            density=True,
        )
        ax.axvline(
            threshold_pct,
            color=self.colors["threshold"],
            linestyle="--",
            label=f"{threshold_percentile}th percentile",
        )
        ax.axvline(
            threshold_std,
            color=self.colors["threshold_alt"],
            linestyle="--",
            label=f"μ + {threshold_std_multiplier}σ",
        )
        ax.set_xlabel("Reconstruction Error")
        ax.set_ylabel("Density")
        ax.set_title("Reconstruction Error Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Box plot comparison
        ax = axes[1]
        data_to_plot = [normal_errors, anomaly_errors]
        labels = ["Normal", "Anomaly"]
        colors = [self.colors["normal"], self.colors["anomaly"]]

        box_plot = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
        for patch, color in zip(box_plot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.axhline(
            threshold_pct,
            color=self.colors["threshold"],
            linestyle="--",
            label=f"{threshold_percentile}th percentile",
        )
        ax.axhline(
            threshold_std,
            color=self.colors["threshold_alt"],
            linestyle="--",
            label=f"μ + {threshold_std_multiplier}σ",
        )
        ax.set_ylabel("Reconstruction Error")
        ax.set_title("Error Distribution by Data Type")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Per-class distributions (if applicable)
        if (
            anomaly_classes is not None
            and len(set(anomaly_classes)) > 1
            and len(axes) > 2
        ):
            ax = axes[2]
            unique_classes = sorted(set(anomaly_classes))

            # Normal data
            ax.hist(
                normal_errors,
                bins=30,
                alpha=0.5,
                label="Normal",
                color=self.colors["normal"],
                density=True,
            )

            # Each anomaly class
            class_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_classes)))
            for i, cls in enumerate(unique_classes):
                if cls != 0:  # Skip normal class
                    cls_indices = [j for j, c in enumerate(anomaly_classes) if c == cls]
                    cls_errors = anomaly_errors[cls_indices]
                    ax.hist(
                        cls_errors,
                        bins=30,
                        alpha=0.7,
                        label=f"Class {cls}",
                        color=class_colors[i],
                        density=True,
                    )

            ax.axvline(
                threshold_pct,
                color=self.colors["threshold"],
                linestyle="--",
                label=f"{threshold_percentile}th percentile",
            )
            ax.axvline(
                threshold_std,
                color=self.colors["threshold_alt"],
                linestyle="--",
                label=f"μ + {threshold_std_multiplier}σ",
            )
            ax.set_xlabel("Reconstruction Error")
            ax.set_ylabel("Density")
            ax.set_title("Per-Class Error Distribution")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot 4: Class performance summary
            if len(axes) > 3:
                ax = axes[3]
                class_stats = []
                class_names = []

                for cls in unique_classes:
                    if cls != 0:
                        cls_indices = [
                            j for j, c in enumerate(anomaly_classes) if c == cls
                        ]
                        cls_errors = anomaly_errors[cls_indices]
                        detection_rate_pct = np.mean(cls_errors > threshold_pct)
                        detection_rate_std = np.mean(cls_errors > threshold_std)
                        class_stats.append([detection_rate_pct, detection_rate_std])
                        class_names.append(f"Class {cls}")

                class_stats = np.array(class_stats)
                x = np.arange(len(class_names))
                width = 0.35

                ax.bar(
                    x - width / 2,
                    class_stats[:, 0],
                    width,
                    label=f"{threshold_percentile}th percentile",
                    alpha=0.7,
                    color=self.colors["threshold"],
                )
                ax.bar(
                    x + width / 2,
                    class_stats[:, 1],
                    width,
                    label=f"μ + {threshold_std_multiplier}σ",
                    alpha=0.7,
                    color=self.colors["threshold_alt"],
                )

                ax.set_xlabel("Anomaly Class")
                ax.set_ylabel("Detection Rate")
                ax.set_title("Detection Rate by Class and Threshold")
                ax.set_xticks(x)
                ax.set_xticklabels(class_names)
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print threshold information
        print("Threshold Analysis:")
        print(f"  • {threshold_percentile}th percentile: {threshold_pct:.6f}")
        print(f"  • μ + {threshold_std_multiplier}σ: {threshold_std:.6f}")

    def plot_latent_space_tsne(
        self,
        autoencoder,
        normal_data,
        anomaly_data,
        anomaly_classes=None,
        max_samples=1000,
        perplexity=30,
        random_state=42,
    ):
        """
        Plot t-SNE visualization of latent space.

        Args:
            autoencoder: Trained autoencoder model
            normal_data (np.array): Normal data samples
            anomaly_data (np.array): Anomaly data samples
            anomaly_classes (list): Class labels for anomaly data
            max_samples (int): Maximum samples to use for t-SNE
            perplexity (int): t-SNE perplexity parameter
            random_state (int): Random state for reproducibility
        """
        from sklearn.manifold import TSNE

        # Sample data if too large
        if len(normal_data) > max_samples // 2:
            normal_indices = np.random.choice(
                len(normal_data), max_samples // 2, replace=False
            )
            normal_sample = normal_data[normal_indices]
        else:
            normal_sample = normal_data

        if len(anomaly_data) > max_samples // 2:
            anomaly_indices = np.random.choice(
                len(anomaly_data), max_samples // 2, replace=False
            )
            anomaly_sample = anomaly_data[anomaly_indices]
            if anomaly_classes is not None:
                anomaly_classes_sample = [anomaly_classes[i] for i in anomaly_indices]
            else:
                anomaly_classes_sample = None
        else:
            anomaly_sample = anomaly_data
            anomaly_classes_sample = anomaly_classes

        # Get latent representations using the encoder method
        normal_latent = autoencoder.encode(normal_sample, verbose=0)
        anomaly_latent = autoencoder.encode(anomaly_sample, verbose=0)

        # Combine data
        all_latent = np.vstack([normal_latent, anomaly_latent])
        labels = ["Normal"] * len(normal_latent) + ["Anomaly"] * len(anomaly_latent)

        # Compute t-SNE
        print(f"Computing t-SNE embedding for {len(all_latent)} samples...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        latent_2d = tsne.fit_transform(all_latent)

        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Normal vs Anomaly
        ax = axes[0]
        normal_mask = np.array(labels) == "Normal"
        anomaly_mask = np.array(labels) == "Anomaly"

        ax.scatter(
            latent_2d[normal_mask, 0],
            latent_2d[normal_mask, 1],
            c=self.colors["normal"],
            alpha=0.6,
            label="Normal",
            s=20,
        )
        ax.scatter(
            latent_2d[anomaly_mask, 0],
            latent_2d[anomaly_mask, 1],
            c=self.colors["anomaly"],
            alpha=0.6,
            label="Anomaly",
            s=20,
        )
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title("Latent Space: Normal vs Anomaly")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: By anomaly class
        ax = axes[1]
        ax.scatter(
            latent_2d[normal_mask, 0],
            latent_2d[normal_mask, 1],
            c=self.colors["normal"],
            alpha=0.6,
            label="Normal",
            s=20,
        )

        if anomaly_classes_sample is not None:
            unique_classes = sorted(set(anomaly_classes_sample))
            class_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_classes)))

            for i, cls in enumerate(unique_classes):
                if cls != 0:  # Skip normal class
                    cls_mask = np.array(anomaly_classes_sample) == cls
                    anomaly_start_idx = len(normal_latent)
                    global_mask = np.zeros(len(latent_2d), dtype=bool)
                    global_mask[anomaly_start_idx:][cls_mask] = True

                    ax.scatter(
                        latent_2d[global_mask, 0],
                        latent_2d[global_mask, 1],
                        c=[class_colors[i]],
                        alpha=0.7,
                        label=f"Class {cls}",
                        s=20,
                    )
        else:
            ax.scatter(
                latent_2d[anomaly_mask, 0],
                latent_2d[anomaly_mask, 1],
                c=self.colors["anomaly"],
                alpha=0.6,
                label="Anomaly",
                s=20,
            )

        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title("Latent Space: By Anomaly Class")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_roc_analysis(self, normal_errors, anomaly_errors, anomaly_classes=None):
        """
        Plot ROC (Receiver Operating Characteristic) curve analysis.

        Args:
            normal_errors (np.array): Reconstruction errors for normal data
            anomaly_errors (np.array): Reconstruction errors for anomaly data
            anomaly_classes (list): Class labels for anomaly data
        """
        from sklearn.metrics import roc_curve, auc, precision_recall_curve

        # Prepare data for ROC analysis
        # Normal = 0 (negative class), Anomaly = 1 (positive class)
        y_true = np.concatenate(
            [np.zeros(len(normal_errors)), np.ones(len(anomaly_errors))]
        )
        y_scores = np.concatenate([normal_errors, anomaly_errors])

        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)

        # Compute Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)

        # Create comprehensive ROC analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: ROC Curve
        ax = axes[0, 0]
        ax.plot(
            fpr,
            tpr,
            color=self.colors["anomaly"],
            lw=2,
            label=f"ROC Curve (AUC = {roc_auc:.3f})",
        )
        ax.plot(
            [0, 1],
            [0, 1],
            color="gray",
            lw=1,
            linestyle="--",
            alpha=0.8,
            label="Random Classifier",
        )
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity)")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        # Add optimal threshold point
        # Find threshold that maximizes TPR - FPR (Youden's index)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        optimal_tpr = tpr[optimal_idx]

        ax.plot(
            optimal_fpr,
            optimal_tpr,
            "ro",
            markersize=8,
            label=f"Optimal Point (threshold={optimal_threshold:.4f})",
        )
        ax.legend(loc="lower right")

        # Plot 2: Precision-Recall Curve
        ax = axes[0, 1]
        ax.plot(
            recall,
            precision,
            color=self.colors["threshold"],
            lw=2,
            label=f"PR Curve (AUC = {pr_auc:.3f})",
        )
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall (Sensitivity)")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

        # Plot 3: Threshold Analysis
        ax = axes[1, 0]
        # Sample thresholds for visualization
        threshold_range = np.linspace(np.min(y_scores), np.max(y_scores), 100)
        tpr_vals = []
        fpr_vals = []

        for thresh in threshold_range:
            predictions = y_scores > thresh
            tp = np.sum((predictions == 1) & (y_true == 1))
            fp = np.sum((predictions == 1) & (y_true == 0))
            tn = np.sum((predictions == 0) & (y_true == 0))
            fn = np.sum((predictions == 0) & (y_true == 1))

            tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0

            tpr_vals.append(tpr_val)
            fpr_vals.append(fpr_val)

        ax.plot(
            threshold_range,
            tpr_vals,
            color=self.colors["anomaly"],
            label="True Positive Rate",
            lw=2,
        )
        ax.plot(
            threshold_range,
            fpr_vals,
            color=self.colors["normal"],
            label="False Positive Rate",
            lw=2,
        )
        ax.axvline(
            optimal_threshold,
            color="red",
            linestyle="--",
            alpha=0.8,
            label="Optimal Threshold",
        )
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Rate")
        ax.set_title("TPR and FPR vs Threshold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Performance metrics summary
        ax = axes[1, 1]

        # Calculate metrics at different common thresholds
        percentiles = [90, 95, 99]
        threshold_methods = {}

        # Percentile-based thresholds
        for p in percentiles:
            thresh = np.percentile(normal_errors, p)
            predictions = y_scores > thresh
            tp = np.sum((predictions == 1) & (y_true == 1))
            fp = np.sum((predictions == 1) & (y_true == 0))
            tn = np.sum((predictions == 0) & (y_true == 0))
            fn = np.sum((predictions == 0) & (y_true == 1))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0

            threshold_methods[f"{p}th percentile"] = {
                "threshold": thresh,
                "tpr": tpr,
                "fpr": fpr,
                "precision": precision,
            }

        # Statistical threshold
        stat_thresh = np.mean(normal_errors) + 2 * np.std(normal_errors)
        predictions = y_scores > stat_thresh
        tp = np.sum((predictions == 1) & (y_true == 1))
        fp = np.sum((predictions == 1) & (y_true == 0))
        tn = np.sum((predictions == 0) & (y_true == 0))
        fn = np.sum((predictions == 0) & (y_true == 1))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        threshold_methods["μ + 2σ"] = {
            "threshold": stat_thresh,
            "tpr": tpr,
            "fpr": fpr,
            "precision": precision,
        }

        # Optimal threshold
        threshold_methods["Optimal (Youden)"] = {
            "threshold": optimal_threshold,
            "tpr": optimal_tpr,
            "fpr": optimal_fpr,
            "precision": precision,  # Using the same precision for simplicity
        }

        # Create comparison plot
        methods = list(threshold_methods.keys())
        tpr_values = [threshold_methods[m]["tpr"] for m in methods]
        fpr_values = [threshold_methods[m]["fpr"] for m in methods]

        x = np.arange(len(methods))
        width = 0.35

        ax.bar(
            x - width / 2,
            tpr_values,
            width,
            label="True Positive Rate",
            alpha=0.7,
            color=self.colors["anomaly"],
        )
        ax.bar(
            x + width / 2,
            fpr_values,
            width,
            label="False Positive Rate",
            alpha=0.7,
            color=self.colors["normal"],
        )

        ax.set_xlabel("Threshold Method")
        ax.set_ylabel("Rate")
        ax.set_title("Performance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print detailed results
        print("\n📊 ROC Analysis Results:")
        print(f"   • ROC AUC: {roc_auc:.3f}")
        print(f"   • PR AUC: {pr_auc:.3f}")
        print(f"   • Optimal threshold (Youden's): {optimal_threshold:.6f}")
        print(
            f"   • At optimal threshold - TPR: {optimal_tpr:.3f}, FPR: {optimal_fpr:.3f}"
        )

        print("\n📋 Threshold Method Comparison:")
        print(
            f"{'Method':<15} {'Threshold':<12} {'TPR':<8} {'FPR':<8} {'Precision':<8}"
        )
        print("-" * 55)
        for method, metrics in threshold_methods.items():
            print(
                f"{method:<15} {metrics['threshold']:<12.6f} {metrics['tpr']:<8.3f} "
                f"{metrics['fpr']:<8.3f} {metrics['precision']:<8.3f}"
            )

        # Performance interpretation
        print("\n🎯 Performance Interpretation:")
        if roc_auc >= 0.9:
            print("   • Excellent discrimination (AUC ≥ 0.9)")
        elif roc_auc >= 0.8:
            print("   • Good discrimination (0.8 ≤ AUC < 0.9)")
        elif roc_auc >= 0.7:
            print("   • Fair discrimination (0.7 ≤ AUC < 0.8)")
        else:
            print("   • Poor discrimination (AUC < 0.7)")

        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "optimal_threshold": optimal_threshold,
            "optimal_tpr": optimal_tpr,
            "optimal_fpr": optimal_fpr,
            "threshold_methods": threshold_methods,
        }

    def plot_svm_analysis(
        self, normal_scores, anomaly_scores, anomaly_classes=None, fold_name="Best Fold"
    ):
        """
        Plot comprehensive SVM anomaly score analysis.

        Args:
            normal_scores (np.array): SVM anomaly scores for normal data
                (``-decision_function``: positive = outside the boundary = anomaly)
            anomaly_scores (np.array): SVM anomaly scores for anomaly data
            anomaly_classes (list): Class labels for anomaly data
            fold_name (str): Name of the fold being analyzed
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import roc_curve

        print(f"🏆 Best Performing Fold (Distance-Based): {fold_name}")

        # The scores are anomaly scores (minus the OCSVM decision function):
        # Negative scores = Normal (inside the boundary)
        # Positive scores = Anomaly (outside the boundary)

        # Create proper SVM visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Score distributions
        ax = axes[0, 0]
        ax.hist(
            normal_scores,
            alpha=0.7,
            bins=30,
            label="Normal",
            color="blue",
            density=True,
        )
        ax.hist(
            anomaly_scores,
            alpha=0.7,
            bins=30,
            label="Anomaly",
            color="red",
            density=True,
        )
        ax.axvline(
            0,
            color="black",
            linestyle="--",
            alpha=0.8,
            label="Decision Boundary (score=0)",
        )
        ax.set_xlabel("SVM Anomaly Score (-decision function)")
        ax.set_ylabel("Density")
        ax.set_title("Distribution of SVM Decision Scores")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Box plots
        ax = axes[0, 1]
        data_to_plot = [normal_scores, anomaly_scores]
        box_plot = ax.boxplot(
            data_to_plot, tick_labels=["Normal", "Anomaly"], patch_artist=True
        )
        box_plot["boxes"][0].set_facecolor("blue")
        box_plot["boxes"][1].set_facecolor("red")
        ax.axhline(
            0, color="black", linestyle="--", alpha=0.8, label="Decision Boundary"
        )
        ax.set_ylabel("SVM Anomaly Score")
        ax.set_title("Score Distribution Summary")
        ax.grid(True, alpha=0.3)

        # Plot 3: ROC Curve
        # For ROC: Normal=0, Anomaly=1 (standard convention)
        y_true = np.concatenate(
            [np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))]
        )

        # Higher score = more anomalous, which is what the ROC curve expects
        y_scores = np.concatenate([normal_scores, anomaly_scores])

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)
        print(f"   • ROC AUC: {roc_auc:.3f}")
        ax = axes[1, 0]
        ax.plot(fpr, tpr, color="red", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
        ax.plot(
            [0, 1],
            [0, 1],
            color="gray",
            lw=1,
            linestyle="--",
            alpha=0.8,
            label="Random",
        )
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve (SVM Anomaly Score)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        # Plot 4: Performance metrics
        ax = axes[1, 1]

        # Calculate performance at decision boundary (score = 0)
        normal_correct = np.sum(normal_scores < 0)
        anomaly_correct = np.sum(anomaly_scores > 0)

        # Show some key statistics
        metrics_text = f"""SVM Performance Summary:
        
Decision Boundary: score = 0
        
Normal Data:
• Correctly classified: {normal_correct}/{len(normal_scores)} ({normal_correct / len(normal_scores) * 100:.1f}%)
• Mean score: {normal_scores.mean():.3f}
• Std score: {normal_scores.std():.3f}

Anomaly Data:
• Correctly classified: {anomaly_correct}/{len(anomaly_scores)} ({anomaly_correct / len(anomaly_scores) * 100:.1f}%)
• Mean score: {anomaly_scores.mean():.3f}
• Std score: {anomaly_scores.std():.3f}

Overall Accuracy: {(normal_correct + anomaly_correct) / (len(normal_scores) + len(anomaly_scores)) * 100:.1f}%
ROC AUC: {roc_auc:.3f}"""

        ax.text(
            0.05,
            0.95,
            metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        plt.tight_layout()
        plt.show()

        return {
            "roc_auc": roc_auc,
            "normal_accuracy": normal_correct / len(normal_scores),
            "anomaly_accuracy": anomaly_correct / len(anomaly_scores),
            "overall_accuracy": (normal_correct + anomaly_correct)
            / (len(normal_scores) + len(anomaly_scores)),
        }
