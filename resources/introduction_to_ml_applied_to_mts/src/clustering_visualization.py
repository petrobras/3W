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
        plt.style.use('default')
    
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
            'bo-', linewidth=2, markersize=6
        )
        axes[0, 1].set_title("PCA Explained Variance (First 20 Components)", fontweight="bold")
        axes[0, 1].set_xlabel("Principal Component")
        axes[0, 1].set_ylabel("Explained Variance Ratio")
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Cumulative explained variance
        cumulative_variance = np.cumsum(pca_model.explained_variance_ratio_)
        axes[0, 2].plot(
            range(1, min(101, len(cumulative_variance) + 1)),
            cumulative_variance[:100],
            'ro-', linewidth=2, markersize=4
        )
        axes[0, 2].axhline(y=0.90, color='g', linestyle='--', label='90%')
        axes[0, 2].axhline(y=0.95, color='orange', linestyle='--', label='95%')
        axes[0, 2].set_title("Cumulative Explained Variance", fontweight="bold")
        axes[0, 2].set_xlabel("Number of Components")
        axes[0, 2].set_ylabel("Cumulative Variance Explained")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. PCA 2D visualization
        for i, class_label in enumerate(unique_classes):
            mask = y_labels == class_label
            axes[1, 0].scatter(
                X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                c=[colors[i]], label=f'Class {class_label}',
                alpha=0.7, s=50
            )
        axes[1, 0].set_title("PCA: First Two Components", fontweight="bold")
        axes[1, 0].set_xlabel("PC1")
        axes[1, 0].set_ylabel("PC2")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Feature distribution (sample)
        feature_sample = X_original[:, :min(20, X_original.shape[1])]
        axes[1, 1].boxplot(feature_sample.T)
        axes[1, 1].set_title("Feature Distribution (Sample)", fontweight="bold")
        axes[1, 1].set_xlabel("Feature Index")
        axes[1, 1].set_ylabel("Feature Value")
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Scaled feature distribution
        scaled_sample = X_scaled[:, :min(20, X_scaled.shape[1])]
        axes[1, 2].boxplot(scaled_sample.T)
        axes[1, 2].set_title("Scaled Feature Distribution (Sample)", fontweight="bold")
        axes[1, 2].set_xlabel("Feature Index")
        axes[1, 2].set_ylabel("Scaled Feature Value")
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_kmeans_analysis(self, k_range, results, optimal_k_scaled, optimal_k_pca, 
                           X_pca_2d, optimal_labels, centroids_2d, y_true, n_clusters):
        """Create K-means analysis visualization"""
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        
        # Plot 1: Elbow Method - Scaled Data
        axes[0, 0].plot(k_range, results['scaled']['wcss'], 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_title("Elbow Method: Scaled Data", fontweight="bold")
        axes[0, 0].set_xlabel("Number of Clusters (K)")
        axes[0, 0].set_ylabel("WCSS")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Mark optimal K
        optimal_idx = optimal_k_scaled - 2
        axes[0, 0].scatter([optimal_k_scaled], [results['scaled']['wcss'][optimal_idx]],
                          color='red', s=100, zorder=5, label=f'Optimal K={optimal_k_scaled}')
        axes[0, 0].legend()
        
        # Plot 2: Elbow Method - PCA Data
        axes[0, 1].plot(k_range, results['pca_50']['wcss'], 'go-', linewidth=2, markersize=8)
        axes[0, 1].set_title("Elbow Method: PCA 50D Data", fontweight="bold")
        axes[0, 1].set_xlabel("Number of Clusters (K)")
        axes[0, 1].set_ylabel("WCSS")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Mark optimal K
        optimal_idx_pca = optimal_k_pca - 2
        axes[0, 1].scatter([optimal_k_pca], [results['pca_50']['wcss'][optimal_idx_pca]],
                          color='red', s=100, zorder=5, label=f'Optimal K={optimal_k_pca}')
        axes[0, 1].legend()
        
        # Plot 3: Silhouette Scores Comparison
        axes[0, 2].plot(k_range, results['scaled']['silhouette'], 'bo-', linewidth=2, label='Scaled Data')
        axes[0, 2].plot(k_range, results['pca_50']['silhouette'], 'go-', linewidth=2, label='PCA 50D')
        axes[0, 2].set_title("Silhouette Score Comparison", fontweight="bold")
        axes[0, 2].set_xlabel("Number of Clusters (K)")
        axes[0, 2].set_ylabel("Silhouette Score")
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()
        
        # Plot 4: Optimal K-means Clustering (2D PCA)
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        for i in range(n_clusters):
            mask = optimal_labels == i
            axes[1, 0].scatter(
                X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                c=[colors[i]], label=f'Cluster {i}',
                alpha=0.7, s=50
            )
        
        # Plot centroids
        if centroids_2d is not None:
            axes[1, 0].scatter(
                centroids_2d[:, 0], centroids_2d[:, 1],
                c='red', marker='x', s=200, linewidths=3, label='Centroids'
            )
        
        axes[1, 0].set_title(f"Optimal K-means: K={n_clusters}", fontweight="bold")
        axes[1, 0].set_xlabel("PC1")
        axes[1, 0].set_ylabel("PC2")
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: True Classes
        unique_classes = np.unique(y_true)
        class_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
        for i, class_label in enumerate(unique_classes):
            mask = y_true == class_label
            axes[1, 1].scatter(
                X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                c=[class_colors[i]], label=f'Class {class_label}',
                alpha=0.7, s=50
            )
        axes[1, 1].set_title("True Classes (Ground Truth)", fontweight="bold")
        axes[1, 1].set_xlabel("PC1")
        axes[1, 1].set_ylabel("PC2")
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
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
            labels=list(range(max(max(unique_true_classes), max(unique_clusters)) + 1))
        )
        
        # Trim matrix to only the relevant classes and clusters
        cluster_class_matrix = cluster_class_matrix[
            np.ix_(unique_true_classes, unique_clusters)
        ]
        
        im = axes[1, 2].imshow(cluster_class_matrix, cmap='Blues', aspect='auto')
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
                    j, i, cluster_class_matrix[i, j],
                    ha="center", va="center",
                    color=("black" if cluster_class_matrix[i, j] < cluster_class_matrix.max() / 2 else "white")
                )
        
        plt.colorbar(im, ax=axes[1, 2])
        plt.tight_layout()
        plt.show()
    
    def plot_clustering_comparison(self, X_pca_2d, kmeans_labels, ms_labels, dbscan_results, 
                                 y_true, methods, n_clusters_found, silhouette_scores):
        """Create clustering methods comparison visualization"""
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        
        # K-means results
        n_clusters_kmeans = len(np.unique(kmeans_labels))
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters_kmeans))
        
        for i in range(n_clusters_kmeans):
            mask = kmeans_labels == i
            axes[0, 0].scatter(
                X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                c=[colors[i]], label=f'Cluster {i}',
                alpha=0.7, s=30
            )
        axes[0, 0].set_title(f"K-means: {n_clusters_kmeans} Clusters", fontweight="bold")
        axes[0, 0].set_xlabel("PC1")
        axes[0, 0].set_ylabel("PC2")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Mean Shift results
        n_clusters_ms = len(np.unique(ms_labels))
        colors_ms = plt.cm.viridis(np.linspace(0, 1, n_clusters_ms))
        
        for i in range(n_clusters_ms):
            mask = ms_labels == i
            axes[0, 1].scatter(
                X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                c=[colors_ms[i]], label=f'Cluster {i}',
                alpha=0.7, s=30
            )
        axes[0, 1].set_title(f"Mean Shift: {n_clusters_ms} Clusters", fontweight="bold")
        axes[0, 1].set_xlabel("PC1")
        axes[0, 1].set_ylabel("PC2")
        axes[0, 1].grid(True, alpha=0.3)
        
        # DBSCAN results
        if dbscan_results is not None:
            dbscan_labels = dbscan_results['labels']
            unique_labels = set(dbscan_labels)
            colors_db = plt.cm.plasma(np.linspace(0, 1, len(unique_labels)))
            
            for k, col in zip(unique_labels, colors_db):
                if k == -1:
                    mask = dbscan_labels == k
                    axes[0, 2].scatter(
                        X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                        c='black', marker='x', alpha=0.5, s=20, label='Noise'
                    )
                else:
                    mask = dbscan_labels == k
                    axes[0, 2].scatter(
                        X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                        c=[col], label=f'Cluster {k}',
                        alpha=0.7, s=30
                    )
            axes[0, 2].set_title(f"DBSCAN: {dbscan_results['n_clusters']} Clusters + Noise", fontweight="bold")
        else:
            axes[0, 2].text(0.5, 0.5, 'DBSCAN\nNo Valid\nClustering', 
                           ha='center', va='center', transform=axes[0, 2].transAxes,
                           fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgray'))
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
                X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                c=[class_colors[i]], label=f'Class {class_label}',
                alpha=0.7, s=30
            )
        axes[1, 0].set_title("True Classes (Ground Truth)", fontweight="bold")
        axes[1, 0].set_xlabel("PC1")
        axes[1, 0].set_ylabel("PC2")
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Clusters found comparison
        true_n_classes = len(unique_classes)
        x_pos = np.arange(len(methods))
        bars = axes[1, 1].bar(x_pos, n_clusters_found, alpha=0.7, color=['blue', 'green', 'red'])
        axes[1, 1].axhline(y=true_n_classes, color='orange', linestyle='--', 
                          linewidth=2, label=f'True Classes ({true_n_classes})')
        axes[1, 1].set_title("Clusters Found by Method", fontweight="bold")
        axes[1, 1].set_ylabel("Number of Clusters")
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(methods)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Performance metrics comparison
        metrics = ["Silhouette Score"]
        x = np.arange(len(methods))
        bars = axes[1, 2].bar(x, silhouette_scores, color=['blue', 'green', 'red'], alpha=0.7)
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
                f'{score:.3f}',
                ha='center', va='bottom'
            )
        
        plt.tight_layout()
        plt.show()
    
    def plot_cluster_interpretation(self, cluster_interpretations, cluster_class_matrix, 
                                  cluster_feature_stats, X_pca_2d, kmeans_labels, 
                                  centroids_2d, silhouette_scores, methods):
        """Create cluster interpretation visualization"""
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        
        n_clusters = len(cluster_interpretations)
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        
        # Plot 1: Cluster Size Distribution
        cluster_sizes = [cluster_interpretations[i]['size'] for i in range(n_clusters)]
        cluster_labels = [f'Cluster {i}\n({cluster_interpretations[i]["size"]} samples)' 
                         for i in range(n_clusters)]
        
        axes[0, 0].pie(cluster_sizes, labels=cluster_labels, autopct='%1.1f%%',
                      colors=colors, startangle=90)
        axes[0, 0].set_title("Cluster Size Distribution", fontweight="bold")
        
        # Plot 2: Cluster Purity
        cluster_purities = [cluster_interpretations[i]['dominant_percentage'] for i in range(n_clusters)]
        cluster_names = [f'Cluster {i}' for i in range(n_clusters)]
        
        bars = axes[0, 1].bar(cluster_names, cluster_purities, color=colors, alpha=0.7)
        axes[0, 1].set_title("Cluster Purity (% Dominant Class)", fontweight="bold")
        axes[0, 1].set_ylabel("Percentage (%)")
        axes[0, 1].set_ylim(0, 100)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, purity in zip(bars, cluster_purities):
            axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                           f'{purity:.1f}%', ha='center', va='bottom')
        
        # Plot 3: Cluster vs Class Heatmap
        sns.heatmap(cluster_class_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[f'Cluster {i}' for i in range(n_clusters)],
                   yticklabels=[f'Class {i}' for i in range(len(cluster_class_matrix))],
                   ax=axes[0, 2])
        axes[0, 2].set_title("Cluster vs Class Correspondence", fontweight="bold")
        axes[0, 2].set_xlabel("Predicted Cluster")
        axes[0, 2].set_ylabel("True Class")
        
        # Plot 4: Feature Variability by Cluster
        cluster_variabilities = [np.mean(cluster_feature_stats[i]['std']) for i in range(n_clusters)]
        bars = axes[1, 0].bar(cluster_names, cluster_variabilities, color=colors, alpha=0.7)
        axes[1, 0].set_title("Feature Variability by Cluster", fontweight="bold")
        axes[1, 0].set_ylabel("Mean Standard Deviation")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 5: Clusters in 2D PCA Space
        for i in range(n_clusters):
            mask = kmeans_labels == i
            axes[1, 1].scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                             c=[colors[i]], label=f'Cluster {i}', alpha=0.6, s=30)
        
        # Add cluster centers
        if centroids_2d is not None:
            axes[1, 1].scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                             c='red', marker='X', s=200, edgecolors='black',
                             linewidth=2, label='Centroids')
        
        axes[1, 1].set_title("Clusters in 2D PCA Space", fontweight="bold")
        axes[1, 1].set_xlabel("PC1")
        axes[1, 1].set_ylabel("PC2")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Performance Summary
        x = np.arange(len(methods))
        bars = axes[1, 2].bar(x, silhouette_scores, color=['blue', 'green', 'red'], alpha=0.7)
        axes[1, 2].set_title("Clustering Method Performance", fontweight="bold")
        axes[1, 2].set_ylabel("Silhouette Score")
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(methods)
        axes[1, 2].set_ylim(0, max(silhouette_scores) * 1.1)
        
        # Add value labels
        for bar, score in zip(bars, silhouette_scores):
            axes[1, 2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()