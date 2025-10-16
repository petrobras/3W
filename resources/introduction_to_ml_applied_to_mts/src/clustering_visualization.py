"""
Clustering Visualization Utilities
=================================

This module provides visualization functions for clustering analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class ClusteringVisualizer:
    """Create visualizations for clustering analysis"""

    def __init__(self, figsize=(18, 12)):
        self.figsize = figsize
        plt.style.use("default")

    def plot_data_overview(self, X_original, X_scaled, y_labels, pca_model, X_pca_2d):
        """Create data overview visualization"""
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)

        # 1. Class distribution
        unique_classes, class_counts = np.unique(y_labels, return_counts=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_classes)))

        axes[0, 0].bar(unique_classes, class_counts, color=colors)
        axes[0, 0].set_title("Class Distribution", fontweight="bold")
        axes[0, 0].set_xlabel("Class Label")
        axes[0, 0].set_ylabel("Number of Samples")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. PCA explained variance
        axes[0, 1].plot(
            range(1, min(21, len(pca_model.explained_variance_ratio_) + 1)),
            pca_model.explained_variance_ratio_[:20],
            "bo-",
            linewidth=2,
            markersize=6,
        )
        axes[0, 1].set_title(
            "PCA Explained Variance (First 20 Components)", fontweight="bold"
        )
        axes[0, 1].set_xlabel("Principal Component")
        axes[0, 1].set_ylabel("Explained Variance Ratio")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Cumulative explained variance
        cumulative_variance = np.cumsum(pca_model.explained_variance_ratio_)
        axes[0, 2].plot(
            range(1, min(101, len(cumulative_variance) + 1)),
            cumulative_variance[:100],
            "ro-",
            linewidth=2,
            markersize=4,
        )
        axes[0, 2].axhline(y=0.90, color="g", linestyle="--", label="90%")
        axes[0, 2].axhline(y=0.95, color="orange", linestyle="--", label="95%")
        axes[0, 2].set_title("Cumulative Explained Variance", fontweight="bold")
        axes[0, 2].set_xlabel("Number of Components")
        axes[0, 2].set_ylabel("Cumulative Variance Explained")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. PCA 2D visualization
        for i, class_label in enumerate(unique_classes):
            mask = y_labels == class_label
            axes[1, 0].scatter(
                X_pca_2d[mask, 0],
                X_pca_2d[mask, 1],
                c=[colors[i]],
                label=f"Class {class_label}",
                alpha=0.7,
                s=50,
            )
        axes[1, 0].set_title("PCA: First Two Components", fontweight="bold")
        axes[1, 0].set_xlabel("PC1")
        axes[1, 0].set_ylabel("PC2")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Feature distribution (sample)
        feature_sample = X_original[:, : min(20, X_original.shape[1])]
        axes[1, 1].boxplot(feature_sample.T)
        axes[1, 1].set_title("Feature Distribution (Sample)", fontweight="bold")
        axes[1, 1].set_xlabel("Feature Index")
        axes[1, 1].set_ylabel("Feature Value")
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Scaled feature distribution
        scaled_sample = X_scaled[:, : min(20, X_scaled.shape[1])]
        axes[1, 2].boxplot(scaled_sample.T)
        axes[1, 2].set_title("Scaled Feature Distribution (Sample)", fontweight="bold")
        axes[1, 2].set_xlabel("Feature Index")
        axes[1, 2].set_ylabel("Scaled Feature Value")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_kmeans_analysis(
        self,
        k_range,
        results,
        optimal_k_scaled,
        optimal_k_pca,
        X_pca_2d,
        optimal_labels,
        centroids_2d,
        y_true,
        n_clusters,
    ):
        """Create K-means analysis visualization"""
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)

        # Plot 1: Elbow Method - Scaled Data (or PCA Data if scaled not available)
        if optimal_k_scaled is not None and "scaled" in results:
            # Original scaled data analysis
            axes[0, 0].plot(
                k_range, results["scaled"]["wcss"], "bo-", linewidth=2, markersize=8
            )
            axes[0, 0].set_title("Elbow Method: Scaled Data", fontweight="bold")
            axes[0, 0].set_xlabel("Number of Clusters (K)")
            axes[0, 0].set_ylabel("WCSS")
            axes[0, 0].grid(True, alpha=0.3)

            # Mark optimal K
            optimal_idx = optimal_k_scaled - 2
            axes[0, 0].scatter(
                [optimal_k_scaled],
                [results["scaled"]["wcss"][optimal_idx]],
                color="red",
                s=100,
                zorder=5,
                label=f"Optimal K={optimal_k_scaled}",
            )
            axes[0, 0].legend()
        elif optimal_k_pca is not None and "scaled" in results:
            # PCA data analysis when no scaled optimal_k
            axes[0, 0].plot(
                k_range, results["scaled"]["wcss"], "go-", linewidth=2, markersize=8
            )
            axes[0, 0].set_title("Elbow Method: PCA 50D Data", fontweight="bold")
            axes[0, 0].set_xlabel("Number of Clusters (K)")
            axes[0, 0].set_ylabel("WCSS")
            axes[0, 0].grid(True, alpha=0.3)

            # Mark optimal K
            optimal_idx = optimal_k_pca - 2
            axes[0, 0].scatter(
                [optimal_k_pca],
                [results["scaled"]["wcss"][optimal_idx]],
                color="red",
                s=100,
                zorder=5,
                label=f"Optimal K={optimal_k_pca}",
            )
            axes[0, 0].legend()
        else:
            axes[0, 0].text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=axes[0, 0].transAxes,
            )
            axes[0, 0].set_title("Elbow Method", fontweight="bold")

        # Plot 2: Clustering Method Used
        if optimal_k_pca is not None and "pca_50" in results:
            # Legacy: show PCA comparison
            axes[0, 1].plot(
                k_range, results["pca_50"]["wcss"], "go-", linewidth=2, markersize=8
            )
            axes[0, 1].set_title("Elbow Method: PCA 50D Data", fontweight="bold")
            axes[0, 1].set_xlabel("Number of Clusters (K)")
            axes[0, 1].set_ylabel("WCSS")
            axes[0, 1].grid(True, alpha=0.3)

            optimal_idx_pca = optimal_k_pca - 2
            axes[0, 1].scatter(
                [optimal_k_pca],
                [results["pca_50"]["wcss"][optimal_idx_pca]],
                color="red",
                s=100,
                zorder=5,
                label=f"Optimal K={optimal_k_pca}",
            )
            axes[0, 1].legend()
        else:
            # New: emphasize scaled data usage
            axes[0, 1].bar(
                [
                    "Scaled Data\n(Used for Clustering)",
                    "PCA 50D\n(Not Used)",
                    "PCA 2D\n(Visualization Only)",
                ],
                [1, 0, 0.5],
                color=["green", "lightgray", "lightblue"],
                alpha=0.7,
            )
            axes[0, 1].set_title("Clustering Data Usage", fontweight="bold")
            axes[0, 1].set_ylabel("Usage Level")
            axes[0, 1].set_ylim(0, 1.2)
            axes[0, 1].grid(True, alpha=0.3, axis="y")

        # Plot 3: Silhouette Scores
        if optimal_k_scaled is not None and "scaled" in results:
            # Show scaled data analysis
            axes[0, 2].plot(
                k_range,
                results["scaled"]["silhouette"],
                "bo-",
                linewidth=2,
                label="Scaled Data (Used)",
                markersize=8,
            )
            if optimal_k_pca is not None and "pca_50" in results:
                axes[0, 2].plot(
                    k_range,
                    results["pca_50"]["silhouette"],
                    "go-",
                    linewidth=2,
                    label="PCA 50D (Not Used)",
                    alpha=0.5,
                )
        elif optimal_k_pca is not None and "scaled" in results:
            # Show PCA data analysis only
            axes[0, 2].plot(
                k_range,
                results["scaled"]["silhouette"],
                "go-",
                linewidth=2,
                label="PCA 50D Data (Used)",
                markersize=8,
            )

        axes[0, 2].set_title("Silhouette Score Analysis", fontweight="bold")
        axes[0, 2].set_xlabel("Number of Clusters (K)")
        axes[0, 2].set_ylabel("Silhouette Score")
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()

        # Plot 4: Optimal K-means Clustering (2D PCA)
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        for i in range(n_clusters):
            mask = optimal_labels == i
            axes[1, 0].scatter(
                X_pca_2d[mask, 0],
                X_pca_2d[mask, 1],
                c=[colors[i]],
                label=f"Cluster {i}",
                alpha=0.7,
                s=50,
            )

        # Plot centroids
        if centroids_2d is not None:
            axes[1, 0].scatter(
                centroids_2d[:, 0],
                centroids_2d[:, 1],
                c="red",
                marker="x",
                s=200,
                linewidths=3,
                label="Centroids",
            )

        axes[1, 0].set_title(f"Optimal K-means: K={n_clusters}", fontweight="bold")
        axes[1, 0].set_xlabel("PC1")
        axes[1, 0].set_ylabel("PC2")
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 5: True Classes
        unique_classes = np.unique(y_true)
        class_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
        for i, class_label in enumerate(unique_classes):
            mask = y_true == class_label
            axes[1, 1].scatter(
                X_pca_2d[mask, 0],
                X_pca_2d[mask, 1],
                c=[class_colors[i]],
                label=f"Class {class_label}",
                alpha=0.7,
                s=50,
            )
        axes[1, 1].set_title("True Classes (Ground Truth)", fontweight="bold")
        axes[1, 1].set_xlabel("PC1")
        axes[1, 1].set_ylabel("PC2")
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        axes[1, 1].grid(True, alpha=0.3)

        # Plot 6: Confusion Matrix
        # Convert labels to consistent numeric format to avoid string/number mix error
        y_true_numeric = np.array([int(str(label)) for label in y_true])
        optimal_labels_numeric = np.array([int(label) for label in optimal_labels])

        # Get unique values for proper matrix sizing
        unique_true_classes = sorted(np.unique(y_true_numeric))
        unique_clusters = sorted(np.unique(optimal_labels_numeric))

        # Create confusion matrix with proper sizing
        cluster_class_matrix = confusion_matrix(
            y_true_numeric,
            optimal_labels_numeric,
            labels=list(range(max(max(unique_true_classes), max(unique_clusters)) + 1)),
        )

        # Trim matrix to only the relevant classes and clusters
        cluster_class_matrix = cluster_class_matrix[
            np.ix_(unique_true_classes, unique_clusters)
        ]

        im = axes[1, 2].imshow(cluster_class_matrix, cmap="Blues", aspect="auto")
        axes[1, 2].set_title("Cluster vs True Class Matrix", fontweight="bold")
        axes[1, 2].set_xlabel("Predicted Cluster")
        axes[1, 2].set_ylabel("True Class")

        # Set proper tick labels
        axes[1, 2].set_xticks(range(len(unique_clusters)))
        axes[1, 2].set_xticklabels([f"C{c}" for c in unique_clusters])
        axes[1, 2].set_yticks(range(len(unique_true_classes)))
        axes[1, 2].set_yticklabels([f"Class {c}" for c in unique_true_classes])

        # Add text annotations
        for i in range(len(unique_true_classes)):
            for j in range(len(unique_clusters)):
                text = axes[1, 2].text(
                    j,
                    i,
                    cluster_class_matrix[i, j],
                    ha="center",
                    va="center",
                    color=(
                        "black"
                        if cluster_class_matrix[i, j] < cluster_class_matrix.max() / 2
                        else "white"
                    ),
                )

        plt.colorbar(im, ax=axes[1, 2])
        plt.tight_layout()
        plt.show()

    def plot_clustering_comparison(
        self,
        X_pca_2d,
        kmeans_labels,
        ms_labels,
        dbscan_results,
        y_true,
        methods,
        n_clusters_found,
        silhouette_scores,
    ):
        """Create clustering methods comparison visualization"""
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)

        # K-means results
        n_clusters_kmeans = len(np.unique(kmeans_labels))
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters_kmeans))

        for i in range(n_clusters_kmeans):
            mask = kmeans_labels == i
            axes[0, 0].scatter(
                X_pca_2d[mask, 0],
                X_pca_2d[mask, 1],
                c=[colors[i]],
                label=f"Cluster {i}",
                alpha=0.7,
                s=30,
            )
        axes[0, 0].set_title(
            f"K-means: {n_clusters_kmeans} Clusters", fontweight="bold"
        )
        axes[0, 0].set_xlabel("PC1")
        axes[0, 0].set_ylabel("PC2")
        axes[0, 0].grid(True, alpha=0.3)

        # Mean Shift results
        n_clusters_ms = len(np.unique(ms_labels))
        colors_ms = plt.cm.viridis(np.linspace(0, 1, n_clusters_ms))

        for i in range(n_clusters_ms):
            mask = ms_labels == i
            axes[0, 1].scatter(
                X_pca_2d[mask, 0],
                X_pca_2d[mask, 1],
                c=[colors_ms[i]],
                label=f"Cluster {i}",
                alpha=0.7,
                s=30,
            )
        axes[0, 1].set_title(f"Mean Shift: {n_clusters_ms} Clusters", fontweight="bold")
        axes[0, 1].set_xlabel("PC1")
        axes[0, 1].set_ylabel("PC2")
        axes[0, 1].grid(True, alpha=0.3)

        # DBSCAN results
        if dbscan_results is not None:
            dbscan_labels = dbscan_results["labels"]
            unique_labels = set(dbscan_labels)
            colors_db = plt.cm.plasma(np.linspace(0, 1, len(unique_labels)))

            for k, col in zip(unique_labels, colors_db):
                if k == -1:
                    mask = dbscan_labels == k
                    axes[0, 2].scatter(
                        X_pca_2d[mask, 0],
                        X_pca_2d[mask, 1],
                        c="black",
                        marker="x",
                        alpha=0.5,
                        s=20,
                        label="Noise",
                    )
                else:
                    mask = dbscan_labels == k
                    axes[0, 2].scatter(
                        X_pca_2d[mask, 0],
                        X_pca_2d[mask, 1],
                        c=[col],
                        label=f"Cluster {k}",
                        alpha=0.7,
                        s=30,
                    )
            axes[0, 2].set_title(
                f"DBSCAN: {dbscan_results['n_clusters']} Clusters + Noise",
                fontweight="bold",
            )
        else:
            axes[0, 2].text(
                0.5,
                0.5,
                "DBSCAN\nNo Valid\nClustering",
                ha="center",
                va="center",
                transform=axes[0, 2].transAxes,
                fontsize=14,
                bbox=dict(boxstyle="round", facecolor="lightgray"),
            )
            axes[0, 2].set_title("DBSCAN: No Valid Clustering", fontweight="bold")

        axes[0, 2].set_xlabel("PC1")
        axes[0, 2].set_ylabel("PC2")
        axes[0, 2].grid(True, alpha=0.3)

        # True Classes
        unique_classes = np.unique(y_true)
        class_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
        for i, class_label in enumerate(unique_classes):
            mask = y_true == class_label
            axes[1, 0].scatter(
                X_pca_2d[mask, 0],
                X_pca_2d[mask, 1],
                c=[class_colors[i]],
                label=f"Class {class_label}",
                alpha=0.7,
                s=30,
            )
        axes[1, 0].set_title("True Classes (Ground Truth)", fontweight="bold")
        axes[1, 0].set_xlabel("PC1")
        axes[1, 0].set_ylabel("PC2")
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)

        # Clusters found comparison
        true_n_classes = len(unique_classes)
        x_pos = np.arange(len(methods))
        bars = axes[1, 1].bar(
            x_pos, n_clusters_found, alpha=0.7, color=["blue", "green", "red"]
        )
        axes[1, 1].axhline(
            y=true_n_classes,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"True Classes ({true_n_classes})",
        )
        axes[1, 1].set_title("Clusters Found by Method", fontweight="bold")
        axes[1, 1].set_ylabel("Number of Clusters")
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(methods)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Performance metrics comparison
        metrics = ["Silhouette Score"]
        x = np.arange(len(methods))
        bars = axes[1, 2].bar(
            x, silhouette_scores, color=["blue", "green", "red"], alpha=0.7
        )
        axes[1, 2].set_title("Performance Metrics Comparison", fontweight="bold")
        axes[1, 2].set_ylabel("Silhouette Score")
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(methods)
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_ylim(0, max(silhouette_scores) * 1.1)

        # Add value labels
        for bar, score in zip(bars, silhouette_scores):
            axes[1, 2].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.show()

    def plot_cluster_interpretation(
        self,
        cluster_interpretations,
        cluster_class_matrix,
        cluster_feature_stats,
        X_pca_2d,
        kmeans_labels,
        centroids_2d,
        silhouette_scores,
        methods,
    ):
        """Create cluster interpretation visualization"""
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)

        n_clusters = len(cluster_interpretations)
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))

        # Plot 1: Cluster Size Distribution
        cluster_sizes = [cluster_interpretations[i]["size"] for i in range(n_clusters)]
        cluster_labels = [
            f'Cluster {i}\n({cluster_interpretations[i]["size"]} samples)'
            for i in range(n_clusters)
        ]

        axes[0, 0].pie(
            cluster_sizes,
            labels=cluster_labels,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        axes[0, 0].set_title("Cluster Size Distribution", fontweight="bold")

        # Plot 2: Cluster Purity
        cluster_purities = [
            cluster_interpretations[i]["dominant_percentage"] for i in range(n_clusters)
        ]
        cluster_names = [f"Cluster {i}" for i in range(n_clusters)]

        bars = axes[0, 1].bar(cluster_names, cluster_purities, color=colors, alpha=0.7)
        axes[0, 1].set_title("Cluster Purity (% Dominant Class)", fontweight="bold")
        axes[0, 1].set_ylabel("Percentage (%)")
        axes[0, 1].set_ylim(0, 100)
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Add value labels
        for bar, purity in zip(bars, cluster_purities):
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{purity:.1f}%",
                ha="center",
                va="bottom",
            )

        # Plot 3: Cluster vs Class Heatmap
        sns.heatmap(
            cluster_class_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[f"Cluster {i}" for i in range(n_clusters)],
            yticklabels=[f"Class {i}" for i in range(len(cluster_class_matrix))],
            ax=axes[0, 2],
        )
        axes[0, 2].set_title("Cluster vs Class Correspondence", fontweight="bold")
        axes[0, 2].set_xlabel("Predicted Cluster")
        axes[0, 2].set_ylabel("True Class")

        # Plot 4: Feature Variability by Cluster
        cluster_variabilities = [
            np.mean(cluster_feature_stats[i]["std"]) for i in range(n_clusters)
        ]
        bars = axes[1, 0].bar(
            cluster_names, cluster_variabilities, color=colors, alpha=0.7
        )
        axes[1, 0].set_title("Feature Variability by Cluster", fontweight="bold")
        axes[1, 0].set_ylabel("Mean Standard Deviation")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Plot 5: Clusters in 2D PCA Space
        for i in range(n_clusters):
            mask = kmeans_labels == i
            axes[1, 1].scatter(
                X_pca_2d[mask, 0],
                X_pca_2d[mask, 1],
                c=[colors[i]],
                label=f"Cluster {i}",
                alpha=0.6,
                s=30,
            )

        # Add cluster centers
        if centroids_2d is not None:
            axes[1, 1].scatter(
                centroids_2d[:, 0],
                centroids_2d[:, 1],
                c="red",
                marker="X",
                s=200,
                edgecolors="black",
                linewidth=2,
                label="Centroids",
            )

        axes[1, 1].set_title("Clusters in 2D PCA Space", fontweight="bold")
        axes[1, 1].set_xlabel("PC1")
        axes[1, 1].set_ylabel("PC2")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Plot 6: Performance Summary
        x = np.arange(len(methods))
        bars = axes[1, 2].bar(
            x, silhouette_scores, color=["blue", "green", "red"], alpha=0.7
        )
        axes[1, 2].set_title("Clustering Method Performance", fontweight="bold")
        axes[1, 2].set_ylabel("Silhouette Score")
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(methods)
        axes[1, 2].set_ylim(0, max(silhouette_scores) * 1.1)

        # Add value labels
        for bar, score in zip(bars, silhouette_scores):
            axes[1, 2].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.show()

    def plot_meanshift_analysis(self, meanshift_results, clustering_data):
        """
        Create comprehensive Mean Shift clustering visualization & metrics.

        Parameters:
        -----------
        meanshift_results : dict
            Mean Shift results containing both 'pca' and 'normalized' results
        clustering_data : dict
            Dictionary containing original data, labels, and preprocessed data
        """
        from sklearn.metrics import (
            silhouette_score,
            adjusted_rand_score,
            normalized_mutual_info_score,
            homogeneity_score,
            completeness_score,
            v_measure_score,
            confusion_matrix,
        )

        print(
            "📊 Creating Mean Shift Clustering Analysis & Comparison with Ground Truth"
        )
        print("=" * 70)

        # Extract data
        y_true = clustering_data["y_labels"]
        X_pca_2d = clustering_data["X_pca_2d"]
        X_pca_50 = clustering_data["X_pca_50"]
        X_normalized = clustering_data["X_normalized"]

        # Convert ground truth to numeric for consistent plotting
        y_true_numeric = np.array(
            [
                int(str(label)) if isinstance(label, (str, np.str_)) else label
                for label in y_true
            ]
        )

        # Check available data types in meanshift_results
        available_data_types = list(meanshift_results.keys())
        print(f"🔍 Available Mean Shift results: {available_data_types}")

        # Create figure with subplots for both data types
        n_data_types = len(available_data_types)
        fig, axes = plt.subplots(n_data_types, 6, figsize=(24, 8 * n_data_types))

        # Ensure axes is 2D even with single data type
        if n_data_types == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(
            f"Mean Shift Clustering Analysis vs Ground Truth ({n_data_types} Data Types)",
            fontsize=16,
            fontweight="bold",
        )

        for data_idx, data_type in enumerate(available_data_types):
            ms_result = meanshift_results[data_type]
            ms_labels = ms_result["labels"]
            n_clusters_ms = ms_result["n_clusters"]

            # Select appropriate data for metrics calculation
            X_data = X_pca_50 if data_type == "pca" else X_normalized

            print(f"\n🔬 Computing metrics for {data_type.upper()} data...")

            # Calculate comprehensive metrics
            # Handle case where only 1 cluster is found (silhouette score requires >= 2 clusters)
            if n_clusters_ms <= 1:
                print(
                    f"   ⚠️ Only {n_clusters_ms} cluster(s) found - silhouette score not applicable"
                )
                silhouette_avg = -1.0  # Invalid/not applicable
            else:
                silhouette_avg = silhouette_score(X_data, ms_labels)

            ari_score = adjusted_rand_score(y_true_numeric, ms_labels)
            nmi_score = normalized_mutual_info_score(y_true_numeric, ms_labels)
            homogeneity = homogeneity_score(y_true_numeric, ms_labels)
            completeness = completeness_score(y_true_numeric, ms_labels)
            v_measure = v_measure_score(y_true_numeric, ms_labels)

            print(f"📈 Mean Shift ({data_type.upper()}) Quality Metrics:")
            print(f"   • Clusters Found: {n_clusters_ms}")
            print(f"   • Ground Truth Classes: {len(np.unique(y_true_numeric))}")
            if silhouette_avg >= 0:
                print(f"   • Silhouette Score: {silhouette_avg:.3f}")
            else:
                print(f"   • Silhouette Score: N/A (single cluster)")
            print(f"   • Adjusted Rand Index: {ari_score:.3f}")
            print(f"   • Normalized Mutual Info: {nmi_score:.3f}")
            print(f"   • Homogeneity: {homogeneity:.3f}")
            print(f"   • Completeness: {completeness:.3f}")
            print(f"   • V-Measure: {v_measure:.3f}")

            # 1. Mean Shift Clustering Results
            scatter1 = axes[data_idx, 0].scatter(
                X_pca_2d[:, 0],
                X_pca_2d[:, 1],
                c=ms_labels,
                cmap="tab10",
                alpha=0.7,
                s=50,
            )

            # Compute cluster centers manually
            unique_clusters = np.unique(ms_labels)
            valid_clusters = unique_clusters[unique_clusters >= 0]

            if len(valid_clusters) > 0:
                centers_2d = []
                for cluster_id in valid_clusters:
                    cluster_mask = ms_labels == cluster_id
                    cluster_center = np.mean(X_pca_2d[cluster_mask], axis=0)
                    centers_2d.append(cluster_center)

                centers_2d = np.array(centers_2d)
                axes[data_idx, 0].scatter(
                    centers_2d[:, 0],
                    centers_2d[:, 1],
                    c="red",
                    marker="x",
                    s=200,
                    linewidths=3,
                )

            axes[data_idx, 0].set_title(
                f"Mean Shift ({data_type.upper()})\\n{n_clusters_ms} clusters",
                fontweight="bold",
            )
            axes[data_idx, 0].set_xlabel("PCA Component 1")
            axes[data_idx, 0].set_ylabel("PCA Component 2")
            axes[data_idx, 0].grid(True, alpha=0.3)

            # 2. Ground Truth Classes (only for first row)
            if data_idx == 0:
                scatter2 = axes[data_idx, 1].scatter(
                    X_pca_2d[:, 0],
                    X_pca_2d[:, 1],
                    c=y_true_numeric,
                    cmap="tab10",
                    alpha=0.7,
                    s=50,
                )
                axes[data_idx, 1].set_title(
                    f"Ground Truth\\n{len(np.unique(y_true_numeric))} classes",
                    fontweight="bold",
                )
                axes[data_idx, 1].set_xlabel("PCA Component 1")
                axes[data_idx, 1].set_ylabel("PCA Component 2")
                axes[data_idx, 1].grid(True, alpha=0.3)
            else:
                # Copy ground truth for visual comparison
                scatter2 = axes[data_idx, 1].scatter(
                    X_pca_2d[:, 0],
                    X_pca_2d[:, 1],
                    c=y_true_numeric,
                    cmap="tab10",
                    alpha=0.7,
                    s=50,
                )
                axes[data_idx, 1].set_title(
                    f"Ground Truth\\n(reference)", fontweight="bold"
                )
                axes[data_idx, 1].set_xlabel("PCA Component 1")
                axes[data_idx, 1].set_ylabel("PCA Component 2")
                axes[data_idx, 1].grid(True, alpha=0.3)

            # 3. Cluster Size Distribution
            unique_clusters, cluster_counts = np.unique(ms_labels, return_counts=True)
            bars = axes[data_idx, 2].bar(
                unique_clusters,
                cluster_counts,
                color=plt.cm.tab10(np.linspace(0, 1, len(unique_clusters))),
            )
            axes[data_idx, 2].set_title(
                f"Cluster Sizes ({data_type.upper()})", fontweight="bold"
            )
            axes[data_idx, 2].set_xlabel("Cluster ID")
            axes[data_idx, 2].set_ylabel("Number of Samples")
            axes[data_idx, 2].grid(True, alpha=0.3)

            for bar, count in zip(bars, cluster_counts):
                axes[data_idx, 2].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    str(count),
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            # 4. Confusion Matrix
            cm = confusion_matrix(y_true_numeric, ms_labels)
            cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

            im = axes[data_idx, 3].imshow(
                cm_normalized, interpolation="nearest", cmap="Blues"
            )
            axes[data_idx, 3].set_title(
                f"Confusion Matrix ({data_type.upper()})", fontweight="bold"
            )
            axes[data_idx, 3].set_xlabel("Mean Shift Cluster")
            axes[data_idx, 3].set_ylabel("Ground Truth Class")

            thresh = cm_normalized.max() / 2.0
            for i in range(cm_normalized.shape[0]):
                for j in range(cm_normalized.shape[1]):
                    axes[data_idx, 3].text(
                        j,
                        i,
                        f"{cm_normalized[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white" if cm_normalized[i, j] > thresh else "black",
                    )

            # 5. Quality Metrics Summary
            metrics_data = {
                "Metric": [
                    "Silhouette",
                    "ARI",
                    "NMI",
                    "Homogeneity",
                    "Completeness",
                    "V-Measure",
                ],
                "Score": [
                    max(0, silhouette_avg),
                    ari_score,
                    nmi_score,
                    homogeneity,
                    completeness,
                    v_measure,
                ],
            }

            # Handle silhouette score display (show as 0 if N/A)
            display_scores = []
            for i, (metric, score) in enumerate(
                zip(metrics_data["Metric"], metrics_data["Score"])
            ):
                if metric == "Silhouette" and silhouette_avg < 0:
                    display_scores.append(0.0)  # Show as 0 for visualization
                else:
                    display_scores.append(score)

            colors = [
                (
                    "lightgray"
                    if metrics_data["Metric"][i] == "Silhouette" and silhouette_avg < 0
                    else "skyblue" if score >= 0.5 else "lightcoral"
                )
                for i, score in enumerate(display_scores)
            ]
            bars = axes[data_idx, 4].bar(
                metrics_data["Metric"], display_scores, color=colors
            )
            axes[data_idx, 4].set_title(
                f"Quality Metrics ({data_type.upper()})", fontweight="bold"
            )
            axes[data_idx, 4].set_ylabel("Score")
            axes[data_idx, 4].set_ylim(0, 1)
            axes[data_idx, 4].tick_params(axis="x", rotation=45)
            axes[data_idx, 4].grid(True, alpha=0.3)

            for bar, score, metric in zip(bars, display_scores, metrics_data["Metric"]):
                if metric == "Silhouette" and silhouette_avg < 0:
                    axes[data_idx, 4].text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        "N/A",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )
                else:
                    axes[data_idx, 4].text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{score:.3f}",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

            # 6. Class-Cluster Distribution Heatmap
            class_cluster_counts = np.zeros(
                (len(np.unique(y_true_numeric)), len(unique_clusters))
            )
            for i, true_class in enumerate(np.unique(y_true_numeric)):
                for j, cluster in enumerate(unique_clusters):
                    mask = (y_true_numeric == true_class) & (ms_labels == cluster)
                    class_cluster_counts[i, j] = np.sum(mask)

            im2 = axes[data_idx, 5].imshow(
                class_cluster_counts, cmap="YlOrRd", interpolation="nearest"
            )
            axes[data_idx, 5].set_title(
                f"Class-Cluster Distribution ({data_type.upper()})", fontweight="bold"
            )
            axes[data_idx, 5].set_xlabel("Mean Shift Cluster")
            axes[data_idx, 5].set_ylabel("Ground Truth Class")
            axes[data_idx, 5].set_xticks(range(len(unique_clusters)))
            axes[data_idx, 5].set_xticklabels(unique_clusters)
            axes[data_idx, 5].set_yticks(range(len(np.unique(y_true_numeric))))
            axes[data_idx, 5].set_yticklabels(np.unique(y_true_numeric))

            for i in range(class_cluster_counts.shape[0]):
                for j in range(class_cluster_counts.shape[1]):
                    axes[data_idx, 5].text(
                        j,
                        i,
                        f"{int(class_cluster_counts[i, j])}",
                        ha="center",
                        va="center",
                        fontweight="bold",
                        color=(
                            "white"
                            if class_cluster_counts[i, j]
                            > class_cluster_counts.max() / 2
                            else "black"
                        ),
                    )

        plt.tight_layout()
        plt.show()

        # Summary interpretation for both data types
        print("\n🎯 Mean Shift Clustering Comparison Summary:")
        print("=" * 60)

        for data_type in available_data_types:
            ms_result = meanshift_results[data_type]
            ms_labels = ms_result["labels"]
            n_clusters_ms = ms_result["n_clusters"]

            X_data = X_pca_50 if data_type == "pca" else X_normalized

            # Handle silhouette score calculation
            if n_clusters_ms <= 1:
                silhouette_avg = -1.0  # N/A
                silhouette_display = "N/A (single cluster)"
            else:
                silhouette_avg = silhouette_score(X_data, ms_labels)
                silhouette_display = f"{silhouette_avg:.3f}"

            ari_score = adjusted_rand_score(y_true_numeric, ms_labels)

            print(f"\n📊 {data_type.upper()} Data Performance:")
            print(f"   • Clusters found: {n_clusters_ms}")
            print(f"   • Silhouette score: {silhouette_display}")
            print(f"   • Agreement with GT (ARI): {ari_score:.3f}")

            if ari_score > 0.5:
                print(f"   ✅ Good agreement with ground truth")
            elif ari_score > 0.2:
                print(f"   ⚠️ Moderate agreement with ground truth")
            else:
                print(f"   ❌ Poor agreement with ground truth")

            if silhouette_avg >= 0:
                if silhouette_avg > 0.5:
                    print(f"   ✅ Well-separated clusters")
                elif silhouette_avg > 0.2:
                    print(f"   ⚠️ Moderately separated clusters")
                else:
                    print(f"   ❌ Poorly separated clusters")
            else:
                print(f"   ⚠️ Cluster separation not applicable (single cluster)")

    def plot_dbscan_analysis(
        self, dbscan_results_norm, dbscan_results_pca, clustering_data
    ):
        """
        Create comprehensive DBSCAN clustering visualization & metrics.

        Parameters:
        -----------
        dbscan_results_norm : dict
            DBSCAN results for normalized data
        dbscan_results_pca : dict
            DBSCAN results for PCA 50D data
        clustering_data : dict
            Dictionary containing original data, labels, and preprocessed data
        """
        from sklearn.metrics import (
            silhouette_score,
            adjusted_rand_score,
            normalized_mutual_info_score,
            homogeneity_score,
            completeness_score,
            v_measure_score,
            confusion_matrix,
        )

        print("📊 Creating DBSCAN Clustering Analysis & Comparison with Ground Truth")
        print("=" * 70)

        # Extract data
        y_true = clustering_data["y_labels"]
        X_pca_2d = clustering_data["X_pca_2d"]
        X_pca_50 = clustering_data["X_pca_50"]
        X_normalized = clustering_data["X_normalized"]

        # Convert ground truth to numeric for consistent plotting
        y_true_numeric = np.array(
            [
                int(str(label)) if isinstance(label, (str, np.str_)) else label
                for label in y_true
            ]
        )

        # Organize DBSCAN results
        dbscan_results = {"normalized": dbscan_results_norm, "pca": dbscan_results_pca}

        # Filter out None results
        available_results = {k: v for k, v in dbscan_results.items() if v is not None}
        available_data_types = list(available_results.keys())

        print(f"🔍 Available DBSCAN results: {available_data_types}")

        if not available_data_types:
            print("❌ No DBSCAN results available for visualization")
            return

        # Create figure with subplots for both data types
        n_data_types = len(available_data_types)
        fig, axes = plt.subplots(n_data_types, 6, figsize=(24, 8 * n_data_types))

        # Ensure axes is 2D even with single data type
        if n_data_types == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(
            f"DBSCAN Clustering Analysis vs Ground Truth ({n_data_types} Data Types)",
            fontsize=16,
            fontweight="bold",
        )

        for data_idx, data_type in enumerate(available_data_types):
            dbscan_result = available_results[data_type]

            # Extract DBSCAN results
            dbscan_labels = dbscan_result["labels"]
            n_clusters_dbscan = dbscan_result["n_clusters"]
            n_noise = dbscan_result["n_noise"]
            eps = dbscan_result["params"]["eps"]
            min_samples = dbscan_result["params"]["min_samples"]

            # Select appropriate data for metrics calculation
            X_data = X_pca_50 if data_type == "pca" else X_normalized

            print(f"\n🔬 Computing metrics for {data_type.upper()} data...")
            print(f"   • Parameters: eps={eps:.4f}, min_samples={min_samples}")
            print(f"   • Clusters found: {n_clusters_dbscan}")
            print(f"   • Noise points: {n_noise}")

            # Calculate comprehensive metrics
            # Handle case where only noise or single cluster is found
            if n_clusters_dbscan <= 1:
                print(
                    f"   ⚠️ Only {n_clusters_dbscan} cluster(s) found - silhouette score not applicable"
                )
                silhouette_avg = -1.0  # Invalid/not applicable
            else:
                # For DBSCAN, exclude noise points (label -1) from silhouette calculation
                non_noise_mask = dbscan_labels != -1
                if np.sum(non_noise_mask) < 2:
                    silhouette_avg = -1.0
                else:
                    silhouette_avg = silhouette_score(
                        X_data[non_noise_mask], dbscan_labels[non_noise_mask]
                    )

            ari_score = adjusted_rand_score(y_true_numeric, dbscan_labels)
            nmi_score = normalized_mutual_info_score(y_true_numeric, dbscan_labels)
            homogeneity = homogeneity_score(y_true_numeric, dbscan_labels)
            completeness = completeness_score(y_true_numeric, dbscan_labels)
            v_measure = v_measure_score(y_true_numeric, dbscan_labels)

            print(f"📈 DBSCAN ({data_type.upper()}) Quality Metrics:")
            print(f"   • Clusters Found: {n_clusters_dbscan}")
            print(f"   • Noise Points: {n_noise}")
            print(f"   • Ground Truth Classes: {len(np.unique(y_true_numeric))}")
            if silhouette_avg >= 0:
                print(f"   • Silhouette Score: {silhouette_avg:.3f}")
            else:
                print(f"   • Silhouette Score: N/A (insufficient clusters)")
            print(f"   • Adjusted Rand Index: {ari_score:.3f}")
            print(f"   • Normalized Mutual Info: {nmi_score:.3f}")
            print(f"   • Homogeneity: {homogeneity:.3f}")
            print(f"   • Completeness: {completeness:.3f}")
            print(f"   • V-Measure: {v_measure:.3f}")

            # 1. DBSCAN Clustering Results
            scatter1 = axes[data_idx, 0].scatter(
                X_pca_2d[:, 0],
                X_pca_2d[:, 1],
                c=dbscan_labels,
                cmap="tab10",
                alpha=0.7,
                s=50,
            )

            # Compute cluster centers manually (excluding noise points)
            unique_clusters = np.unique(dbscan_labels)
            valid_clusters = unique_clusters[unique_clusters >= 0]  # Exclude noise (-1)

            if len(valid_clusters) > 0:
                centers_2d = []
                for cluster_id in valid_clusters:
                    cluster_mask = dbscan_labels == cluster_id
                    cluster_center = np.mean(X_pca_2d[cluster_mask], axis=0)
                    centers_2d.append(cluster_center)

                centers_2d = np.array(centers_2d)
                axes[data_idx, 0].scatter(
                    centers_2d[:, 0],
                    centers_2d[:, 1],
                    c="red",
                    marker="x",
                    s=200,
                    linewidths=3,
                )

            axes[data_idx, 0].set_title(
                f"DBSCAN ({data_type.upper()})\\n{n_clusters_dbscan} clusters, {n_noise} noise",
                fontweight="bold",
            )
            axes[data_idx, 0].set_xlabel("PCA Component 1")
            axes[data_idx, 0].set_ylabel("PCA Component 2")
            axes[data_idx, 0].grid(True, alpha=0.3)

            # 2. Ground Truth Classes (only for first row)
            if data_idx == 0:
                scatter2 = axes[data_idx, 1].scatter(
                    X_pca_2d[:, 0],
                    X_pca_2d[:, 1],
                    c=y_true_numeric,
                    cmap="tab10",
                    alpha=0.7,
                    s=50,
                )
                axes[data_idx, 1].set_title(
                    f"Ground Truth\\n{len(np.unique(y_true_numeric))} classes",
                    fontweight="bold",
                )
                axes[data_idx, 1].set_xlabel("PCA Component 1")
                axes[data_idx, 1].set_ylabel("PCA Component 2")
                axes[data_idx, 1].grid(True, alpha=0.3)
            else:
                # Copy ground truth for visual comparison
                scatter2 = axes[data_idx, 1].scatter(
                    X_pca_2d[:, 0],
                    X_pca_2d[:, 1],
                    c=y_true_numeric,
                    cmap="tab10",
                    alpha=0.7,
                    s=50,
                )
                axes[data_idx, 1].set_title(
                    f"Ground Truth\\n(reference)", fontweight="bold"
                )
                axes[data_idx, 1].set_xlabel("PCA Component 1")
                axes[data_idx, 1].set_ylabel("PCA Component 2")
                axes[data_idx, 1].grid(True, alpha=0.3)

            # 3. Cluster Size Distribution (including noise)
            unique_labels, label_counts = np.unique(dbscan_labels, return_counts=True)

            # Separate noise from clusters for better visualization
            noise_mask = unique_labels == -1
            cluster_labels = unique_labels[~noise_mask]
            cluster_counts = label_counts[~noise_mask]
            noise_count = label_counts[noise_mask][0] if np.any(noise_mask) else 0

            # Plot clusters
            if len(cluster_labels) > 0:
                bars = axes[data_idx, 2].bar(
                    cluster_labels,
                    cluster_counts,
                    color=plt.cm.tab10(np.linspace(0, 1, len(cluster_labels))),
                )

            # Add noise bar if present
            if noise_count > 0:
                noise_bar = axes[data_idx, 2].bar(
                    [-1], [noise_count], color="gray", alpha=0.7, label="Noise"
                )

            axes[data_idx, 2].set_title(
                f"Cluster/Noise Distribution ({data_type.upper()})", fontweight="bold"
            )
            axes[data_idx, 2].set_xlabel("Cluster ID (-1 = Noise)")
            axes[data_idx, 2].set_ylabel("Number of Samples")
            axes[data_idx, 2].grid(True, alpha=0.3)

            # Add value labels on bars
            if len(cluster_labels) > 0:
                for i, (label, count) in enumerate(zip(cluster_labels, cluster_counts)):
                    axes[data_idx, 2].text(
                        label,
                        count + 1,
                        str(count),
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )
            if noise_count > 0:
                axes[data_idx, 2].text(
                    -1,
                    noise_count + 1,
                    str(noise_count),
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            # 4. Confusion Matrix
            cm = confusion_matrix(y_true_numeric, dbscan_labels)
            cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

            im = axes[data_idx, 3].imshow(
                cm_normalized, interpolation="nearest", cmap="Blues"
            )
            axes[data_idx, 3].set_title(
                f"Confusion Matrix ({data_type.upper()})", fontweight="bold"
            )
            axes[data_idx, 3].set_xlabel("DBSCAN Cluster")
            axes[data_idx, 3].set_ylabel("Ground Truth Class")

            thresh = cm_normalized.max() / 2.0
            for i in range(cm_normalized.shape[0]):
                for j in range(cm_normalized.shape[1]):
                    axes[data_idx, 3].text(
                        j,
                        i,
                        f"{cm_normalized[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white" if cm_normalized[i, j] > thresh else "black",
                    )

            # 5. Quality Metrics Summary
            metrics_data = {
                "Metric": [
                    "Silhouette",
                    "ARI",
                    "NMI",
                    "Homogeneity",
                    "Completeness",
                    "V-Measure",
                ],
                "Score": [
                    max(0, silhouette_avg),
                    ari_score,
                    nmi_score,
                    homogeneity,
                    completeness,
                    v_measure,
                ],
            }

            # Handle silhouette score display (show as 0 if N/A)
            display_scores = []
            for i, (metric, score) in enumerate(
                zip(metrics_data["Metric"], metrics_data["Score"])
            ):
                if metric == "Silhouette" and silhouette_avg < 0:
                    display_scores.append(0.0)  # Show as 0 for visualization
                else:
                    display_scores.append(score)

            colors = [
                (
                    "lightgray"
                    if metrics_data["Metric"][i] == "Silhouette" and silhouette_avg < 0
                    else "skyblue" if score >= 0.5 else "lightcoral"
                )
                for i, score in enumerate(display_scores)
            ]
            bars = axes[data_idx, 4].bar(
                metrics_data["Metric"], display_scores, color=colors
            )
            axes[data_idx, 4].set_title(
                f"Quality Metrics ({data_type.upper()})", fontweight="bold"
            )
            axes[data_idx, 4].set_ylabel("Score")
            axes[data_idx, 4].set_ylim(0, 1)
            axes[data_idx, 4].tick_params(axis="x", rotation=45)
            axes[data_idx, 4].grid(True, alpha=0.3)

            for bar, score, metric in zip(bars, display_scores, metrics_data["Metric"]):
                if metric == "Silhouette" and silhouette_avg < 0:
                    axes[data_idx, 4].text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        "N/A",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )
                else:
                    axes[data_idx, 4].text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{score:.3f}",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

            # 6. Class-Cluster Distribution Heatmap
            class_cluster_counts = np.zeros(
                (len(np.unique(y_true_numeric)), len(unique_labels))
            )
            for i, true_class in enumerate(np.unique(y_true_numeric)):
                for j, cluster in enumerate(unique_labels):
                    mask = (y_true_numeric == true_class) & (dbscan_labels == cluster)
                    class_cluster_counts[i, j] = np.sum(mask)

            im2 = axes[data_idx, 5].imshow(
                class_cluster_counts, cmap="YlOrRd", interpolation="nearest"
            )
            axes[data_idx, 5].set_title(
                f"Class-Cluster Distribution ({data_type.upper()})", fontweight="bold"
            )
            axes[data_idx, 5].set_xlabel("DBSCAN Cluster")
            axes[data_idx, 5].set_ylabel("Ground Truth Class")
            axes[data_idx, 5].set_xticks(range(len(unique_labels)))
            axes[data_idx, 5].set_xticklabels(unique_labels)
            axes[data_idx, 5].set_yticks(range(len(np.unique(y_true_numeric))))
            axes[data_idx, 5].set_yticklabels(np.unique(y_true_numeric))

            for i in range(class_cluster_counts.shape[0]):
                for j in range(class_cluster_counts.shape[1]):
                    axes[data_idx, 5].text(
                        j,
                        i,
                        f"{int(class_cluster_counts[i, j])}",
                        ha="center",
                        va="center",
                        fontweight="bold",
                        color=(
                            "white"
                            if class_cluster_counts[i, j]
                            > class_cluster_counts.max() / 2
                            else "black"
                        ),
                    )

        plt.tight_layout()
        plt.show()

        # Summary interpretation for both data types
        print("\n🎯 DBSCAN Clustering Comparison Summary:")
        print("=" * 60)

        for data_type in available_data_types:
            dbscan_result = available_results[data_type]
            dbscan_labels = dbscan_result["labels"]
            n_clusters_dbscan = dbscan_result["n_clusters"]
            n_noise = dbscan_result["n_noise"]

            X_data = X_pca_50 if data_type == "pca" else X_normalized

            # Handle silhouette score calculation
            if n_clusters_dbscan <= 1:
                silhouette_avg = -1.0  # N/A
                silhouette_display = "N/A (insufficient clusters)"
            else:
                non_noise_mask = dbscan_labels != -1
                if np.sum(non_noise_mask) < 2:
                    silhouette_avg = -1.0
                    silhouette_display = "N/A (insufficient non-noise points)"
                else:
                    silhouette_avg = silhouette_score(
                        X_data[non_noise_mask], dbscan_labels[non_noise_mask]
                    )
                    silhouette_display = f"{silhouette_avg:.3f}"

            ari_score = adjusted_rand_score(y_true_numeric, dbscan_labels)

            print(f"\n📊 {data_type.upper()} Data Performance:")
            print(f"   • Clusters found: {n_clusters_dbscan}")
            print(
                f"   • Noise points: {n_noise} ({n_noise/len(dbscan_labels)*100:.1f}%)"
            )
            print(f"   • Silhouette score: {silhouette_display}")
            print(f"   • Agreement with GT (ARI): {ari_score:.3f}")

            if ari_score > 0.5:
                print(f"   ✅ Good agreement with ground truth")
            elif ari_score > 0.2:
                print(f"   ⚠️ Moderate agreement with ground truth")
            else:
                print(f"   ❌ Poor agreement with ground truth")

            if silhouette_avg >= 0:
                if silhouette_avg > 0.5:
                    print(f"   ✅ Well-separated clusters")
                elif silhouette_avg > 0.2:
                    print(f"   ⚠️ Moderately separated clusters")
                else:
                    print(f"   ❌ Poorly separated clusters")
            else:
                print(f"   ⚠️ Cluster separation not applicable")

            # DBSCAN-specific insights
            if n_noise > len(dbscan_labels) * 0.1:  # More than 10% noise
                print(f"   ⚠️ High noise ratio - consider adjusting eps or min_samples")
            elif n_noise == 0:
                print(f"   📝 No noise detected - data well-clustered or eps too large")

    def clustering_analysis_suite(
        self,
        clustering_data,
        kmeans_results=None,
        advanced_results=None,
        accuracy_results=None,
    ):
        """
        Run a complete suite of clustering analysis visualizations.

        Parameters:
        -----------
        clustering_data : dict
            Dictionary containing data and labels for visualization
        kmeans_results : dict, optional
            K-means analysis results from run_kmeans_complete_analysis
        advanced_results : dict, optional
            Results from advanced clustering methods
        accuracy_results : dict, optional
            Accuracy evaluation results from evaluate_clustering_accuracy

        Returns:
        --------
        None (displays plots)
        """
        print("🎨 Running Complete Clustering Analysis Visualization Suite")
        print("=" * 65)

        visualizations_count = 0

        # 1. Data Overview (if we have the necessary data)
        if all(
            key in clustering_data
            for key in ["X_original", "X_scaled", "y_labels", "pca_model", "X_pca_2d"]
        ):
            print("📊 1. Generating Data Overview...")
            self.plot_data_overview(
                clustering_data["X_original"],
                clustering_data["X_scaled"],
                clustering_data["y_labels"],
                clustering_data["pca_model"],
                clustering_data["X_pca_2d"],
            )
            visualizations_count += 1

        # 2. K-means Analysis (if results provided)
        if kmeans_results:
            print("🎯 2. Generating K-means Analysis Plots...")
            try:
                if "optimization_results" in kmeans_results:
                    results = kmeans_results["optimization_results"]
                    optimal_k = kmeans_results["optimal_k"]
                    analysis_type = kmeans_results.get("analysis_type", "scaled")

                    if analysis_type == "pca50":
                        self.plot_kmeans_analysis(
                            results={"pca50": results.get("pca50", results)},
                            optimal_k_pca50=optimal_k,
                            clustering_data=clustering_data,
                            use_pca_only=True,
                        )
                    else:
                        self.plot_kmeans_analysis(
                            results=results,
                            optimal_k_scaled=optimal_k,
                            clustering_data=clustering_data,
                            use_scaled_only=True,
                        )
                    visualizations_count += 1
            except Exception as e:
                print(f"   ⚠️ K-means visualization error: {str(e)}")

        # 3. Advanced Methods Comparison (if available)
        if advanced_results and hasattr(self, "_plot_multi_algorithm_comparison"):
            print("🔬 3. Generating Advanced Methods Comparison...")
            try:
                # Check if this is AdvancedClusteringVisualizer
                if hasattr(self, "_plot_multi_algorithm_comparison"):
                    self._plot_multi_algorithm_comparison(
                        clustering_data, advanced_results
                    )
                visualizations_count += 1
            except Exception as e:
                print(f"   ⚠️ Advanced methods visualization error: {str(e)}")

        # 4. Performance Summary
        if kmeans_results or advanced_results:
            print("📈 4. Generating Performance Summary...")
            try:
                # Collect performance data
                methods = []
                silhouette_scores = []

                if kmeans_results and "silhouette_score" in kmeans_results:
                    analysis_type = kmeans_results.get("analysis_type", "scaled")
                    methods.append(f"K-means ({analysis_type})")
                    silhouette_scores.append(kmeans_results["silhouette_score"])

                if advanced_results:
                    for method_name, result in advanced_results.items():
                        if (
                            result
                            and isinstance(result, dict)
                            and "silhouette" in result
                        ):
                            methods.append(method_name.replace("_", " ").title())
                            silhouette_scores.append(result["silhouette"])

                if methods and silhouette_scores:
                    # Create performance comparison plot
                    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                    bars = ax.bar(
                        methods,
                        silhouette_scores,
                        color=plt.cm.Set3(np.linspace(0, 1, len(methods))),
                    )
                    ax.set_title(
                        "Clustering Methods Performance Comparison", fontweight="bold"
                    )
                    ax.set_ylabel("Silhouette Score")
                    ax.set_ylim(0, max(silhouette_scores) * 1.1)

                    # Add value labels
                    for bar, score in zip(bars, silhouette_scores):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.005,
                            f"{score:.3f}",
                            ha="center",
                            va="bottom",
                        )

                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    plt.show()
                    visualizations_count += 1

            except Exception as e:
                print(f"   ⚠️ Performance summary error: {str(e)}")

        # 5. Accuracy Summary (if available)
        if accuracy_results:
            print("🎯 5. Generating Accuracy Summary...")
            try:
                methods = list(accuracy_results.keys())
                accuracies = [
                    result["accuracy"] for result in accuracy_results.values()
                ]

                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                bars = ax.bar(
                    methods,
                    accuracies,
                    color=plt.cm.Pastel1(np.linspace(0, 1, len(methods))),
                )
                ax.set_title(
                    "Clustering Accuracy Against Ground Truth", fontweight="bold"
                )
                ax.set_ylabel("Accuracy")
                ax.set_ylim(0, 1)

                # Add value labels
                for bar, acc in zip(bars, accuracies):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{acc:.1%}",
                        ha="center",
                        va="bottom",
                    )

                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.show()
                visualizations_count += 1

            except Exception as e:
                print(f"   ⚠️ Accuracy summary error: {str(e)}")

        # Summary
        print(f"\n🎉 Visualization Suite Complete!")
        print(f"   📊 Generated {visualizations_count} visualization(s)")
        print("-" * 65)


class AdvancedClusteringVisualizer:
    """Create advanced visualizations for clustering analysis"""

    def __init__(self, figsize=(18, 12)):
        self.figsize = figsize
        plt.style.use("default")

    def create_advanced_visualizations(
        self, clustering_data, advanced_results, original_results=None
    ):
        """
        Create enhanced interactive visualizations for clustering analysis

        Parameters:
        -----------
        clustering_data : dict
            Dictionary containing clustering data including X_pca_2d, y_labels, pca_model
        advanced_results : dict
            Results from advanced clustering suite
        original_results : dict, optional
            Original clustering results for comparison
        """
        print("🎨 Creating Advanced Visualizations")
        print("=" * 40)

        # Set style for better plots
        sns.set_palette("husl")

        # Create visualizations
        self._plot_multi_algorithm_comparison(clustering_data, advanced_results)

        print(f"\n🎨 Advanced Visualizations Complete!")

    def _plot_multi_algorithm_comparison(self, clustering_data, advanced_results):
        """Create multi-algorithm comparison dashboard"""
        print("1. Multi-Algorithm Comparison Dashboard")

        # Prepare data
        X_pca_2d = clustering_data["X_pca_2d"]
        y_true = clustering_data["y_labels"]

        # Create subplots for comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "Advanced Clustering Methods Comparison", fontsize=16, fontweight="bold"
        )

        algorithms = ["enhanced_kmeans", "gmm", "hierarchical", "optimized_dbscan"]
        algorithm_names = [
            "Enhanced K-means",
            "Gaussian Mixture",
            "Hierarchical",
            "Optimized DBSCAN",
        ]

        plot_idx = 0
        for alg, name in zip(algorithms, algorithm_names):
            if alg in advanced_results and advanced_results[alg] is not None:
                row, col = plot_idx // 3, plot_idx % 3
                labels = advanced_results[alg]["labels"]

                # Convert labels to numeric for matplotlib color mapping
                labels_numeric = np.array(
                    [
                        int(str(label)) if isinstance(label, (str, np.str_)) else label
                        for label in labels
                    ]
                )

                # Create scatter plot
                scatter = axes[row, col].scatter(
                    X_pca_2d[:, 0],
                    X_pca_2d[:, 1],
                    c=labels_numeric,
                    cmap="tab10",
                    alpha=0.7,
                    s=30,
                )
                axes[row, col].set_title(
                    f'{name}\n(Silhouette: {advanced_results[alg]["silhouette"]:.3f})'
                )
                axes[row, col].set_xlabel("PCA Component 1")
                axes[row, col].set_ylabel("PCA Component 2")
                axes[row, col].grid(True, alpha=0.3)

                # Add cluster centers for all centroid-based methods
                if "model" in advanced_results[alg]:
                    model = advanced_results[alg]["model"]
                    if hasattr(model, "cluster_centers_"):
                        # Centroids are in 50D PCA space, take first 2 components for visualization
                        centroids_2d = model.cluster_centers_[:, :2]
                        axes[row, col].scatter(
                            centroids_2d[:, 0],
                            centroids_2d[:, 1],
                            c="red",
                            marker="x",
                            s=200,
                            linewidths=3,
                            label="Centroids",
                        )
                    elif hasattr(model, "means_"):
                        # For GMM models, use means instead of cluster_centers_
                        centroids_2d = model.means_[:, :2]
                        axes[row, col].scatter(
                            centroids_2d[:, 0],
                            centroids_2d[:, 1],
                            c="red",
                            marker="x",
                            s=200,
                            linewidths=3,
                            label="Centroids",
                        )
                    elif hasattr(model, "children_"):
                        # For hierarchical clustering, calculate cluster centers manually
                        unique_labels = np.unique(labels_numeric)
                        centroids_2d = []
                        for label in unique_labels:
                            if label >= 0:  # Exclude noise points
                                mask = labels_numeric == label
                                center = np.mean(X_pca_2d[mask], axis=0)
                                centroids_2d.append(center)
                        if centroids_2d:
                            centroids_2d = np.array(centroids_2d)
                            axes[row, col].scatter(
                                centroids_2d[:, 0],
                                centroids_2d[:, 1],
                                c="red",
                                marker="x",
                                s=200,
                                linewidths=3,
                                label="Centroids",
                            )

                    # Add legend if centroids were plotted
                    if len(axes[row, col].collections) > 1:
                        axes[row, col].legend()

                plot_idx += 1

        # Ground truth comparison
        if plot_idx < 6:
            row, col = plot_idx // 3, plot_idx % 3
            # Convert y_true to numeric for matplotlib color mapping
            y_true_numeric = np.array(
                [
                    int(str(label)) if isinstance(label, (str, np.str_)) else label
                    for label in y_true
                ]
            )
            scatter = axes[row, col].scatter(
                X_pca_2d[:, 0],
                X_pca_2d[:, 1],
                c=y_true_numeric,
                cmap="tab10",
                alpha=0.7,
                s=30,
            )
            axes[row, col].set_title("Ground Truth Classes")
            axes[row, col].set_xlabel("PCA Component 1")
            axes[row, col].set_ylabel("PCA Component 2")
            axes[row, col].grid(True, alpha=0.3)
            plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, 6):
            row, col = i // 3, i % 3
            axes[row, col].axis("off")

        plt.tight_layout()
        plt.show()
