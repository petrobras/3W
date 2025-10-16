"""
Clustering Utilities for 3W Dataset Analysis
============================================

This module provides utility functions for clustering analysis on the 3W dataset,
including data preparation, clustering algorithms, and evaluation metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans, MeanShift, DBSCAN, estimate_bandwidth
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    adjusted_rand_score,
    confusion_matrix,
)
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate
import warnings

warnings.filterwarnings("ignore")


class ClusteringDataLoader:
    """Load and prepare 3W dataset for clustering analysis"""

    def __init__(self, target_length=500, max_files_per_class=20, enable_sampling=True):
        self.target_length = target_length
        self.max_files_per_class = max_files_per_class
        self.enable_sampling = enable_sampling

    def load_complete_data(self, persistence, complete_data_dir):
        """Load complete 3W dataset from pickle/parquet files"""
        print("Loading Complete 3W Dataset for Clustering Analysis")
        print("=" * 55)

        # Get fold directories
        fold_dirs = [
            d
            for d in os.listdir(complete_data_dir)
            if d.startswith("fold_") and os.path.isdir(os.path.join(complete_data_dir, d))
        ]
        fold_dirs.sort()

        if not fold_dirs:
            raise FileNotFoundError("No fold directories found")

        # Use the first fold for clustering analysis
        selected_fold = fold_dirs[0]
        fold_path = os.path.join(complete_data_dir, selected_fold)
        print(f"Using {selected_fold} for clustering analysis")

        # Data containers
        all_time_series = []
        all_class_labels = []
        all_file_info = []

        # Load both train and test data
        datasets_to_load = ["train_data", "test_data"]

        for dataset_name in datasets_to_load:
            print(f"\nLoading {dataset_name}...")

            pickle_file = os.path.join(fold_path, f"{dataset_name}.pickle")
            parquet_file = os.path.join(fold_path, f"{dataset_name}.parquet")

            if os.path.exists(pickle_file):
                complete_dfs, complete_classes = persistence._load_dataframes(
                    pickle_file, "pickle"
                )
            elif os.path.exists(parquet_file):
                complete_dfs, complete_classes = persistence._load_from_parquet(parquet_file)
            else:
                continue

            if complete_dfs is None or len(complete_dfs) == 0:
                continue

            # Sample files if needed
            if self.enable_sampling and len(complete_dfs) > self.max_files_per_class * 10:
                n_samples = min(self.max_files_per_class * 10, len(complete_dfs))
                selected_indices = np.random.choice(
                    len(complete_dfs), n_samples, replace=False
                )
                complete_dfs = [complete_dfs[i] for i in selected_indices]
                complete_classes = [complete_classes[i] for i in selected_indices]

            # Process each time series
            processed_data = self._process_time_series(
                complete_dfs, complete_classes, dataset_name
            )
            all_time_series.extend(processed_data["time_series"])
            all_class_labels.extend(processed_data["labels"])
            all_file_info.extend(processed_data["file_info"])

        # Convert to numpy arrays
        X_original = np.array(all_time_series)
        y_labels = np.array(all_class_labels)

        return {
            "X_original": X_original,
            "y_labels": y_labels,
            "file_info": all_file_info,
            "fold_used": selected_fold,
        }

    def _process_time_series(self, complete_dfs, complete_classes, dataset_name):
        """Process individual time series data"""
        time_series = []
        labels = []
        file_info = []

        for i, (df, class_label) in enumerate(zip(complete_dfs, complete_classes)):
            try:
                # Drop the 'class' column if it exists
                if "class" in df.columns:
                    df = df.drop("class", axis=1)

                # Use only numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    continue

                df_numeric = df[numeric_cols]
                original_length = len(df_numeric)

                if original_length < 10:
                    continue

                # Resize each column to target length
                resized_data = []
                valid_sensors = 0

                for col in df_numeric.columns:
                    series = df_numeric[col].values

                    if np.isnan(series).all():
                        continue

                    # Interpolate NaN values
                    if np.isnan(series).any():
                        series = pd.Series(series).interpolate().values

                    # Resize using interpolation
                    if original_length != self.target_length:
                        original_indices = np.linspace(0, 1, original_length)
                        target_indices = np.linspace(0, 1, self.target_length)
                        f_interp = interpolate.interp1d(
                            original_indices,
                            series,
                            kind="linear",
                            bounds_error=False,
                            fill_value="extrapolate",
                        )
                        resized_series = f_interp(target_indices)
                    else:
                        resized_series = series

                    resized_data.append(resized_series)
                    valid_sensors += 1

                if valid_sensors == 0:
                    continue

                # Flatten all sensor data into a single feature vector
                flattened_features = np.concatenate(resized_data)

                time_series.append(flattened_features)
                labels.append(class_label)
                file_info.append(
                    {
                        "dataset": dataset_name,
                        "index": i,
                        "original_length": original_length,
                        "sensors_used": valid_sensors,
                        "class": class_label,
                    }
                )

            except Exception as e:
                continue

        return {"time_series": time_series, "labels": labels, "file_info": file_info}


class ClusteringPreprocessor:
    """Handle data preprocessing for clustering analysis"""

    def __init__(self):
        self.scaler_standard = StandardScaler()
        self.scaler_minmax = MinMaxScaler()
        self.pca = PCA()

    def prepare_data(self, X_original):
        """Prepare data for clustering with scaling and dimensionality reduction"""
        print("Data Preparation for Clustering Analysis")
        print("=" * 45)

        # Clean data
        X_clean = self._clean_data(X_original)

        # Apply scaling
        X_scaled = self.scaler_standard.fit_transform(X_clean)
        X_normalized = self.scaler_minmax.fit_transform(X_clean)

        # Apply PCA
        X_pca = self.pca.fit_transform(X_scaled)

        # Calculate variance thresholds
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

        return {
            "X_original": X_clean,
            "X_scaled": X_scaled,
            "X_normalized": X_normalized,
            "X_pca_50": X_pca[:, :50],
            "X_pca_95": X_pca[:, :n_components_95],
            "X_pca_2d": X_pca[:, :2],
            "pca_model": self.pca,
            "scaler_standard": self.scaler_standard,
            "scaler_minmax": self.scaler_minmax,
            "variance_info": {
                "n_components_90": n_components_90,
                "n_components_95": n_components_95,
                "cumulative_variance": cumulative_variance,
            },
        }

    def _clean_data(self, X_original):
        """Clean data by handling NaN and infinite values"""
        nan_count = np.isnan(X_original).sum()
        inf_count = np.isinf(X_original).sum()

        if nan_count > 0 or inf_count > 0:
            print(f"Cleaning data: {nan_count} NaN, {inf_count} infinite values")
            X_clean = X_original.copy()

            # Replace inf with NaN first
            X_clean[np.isinf(X_clean)] = np.nan

            # Replace NaN with column medians
            for i in range(X_clean.shape[1]):
                col_data = X_clean[:, i]
                if np.isnan(col_data).any():
                    median_val = np.nanmedian(col_data)
                    X_clean[np.isnan(col_data), i] = median_val

            return X_clean

        return X_original


class KMeansAnalyzer:
    """K-means clustering analysis with parameter optimization"""

    def __init__(self, k_range=None):
        self.k_range = k_range or range(2, 16)

    def find_optimal_k(self, X_scaled, X_pca_50=None):
        """Find optimal K using elbow method and silhouette analysis on SCALED data only"""
        print("K-means: Finding Optimal K (using SCALED data for clustering)")
        print("=" * 60)

        results = {
            "scaled": {"wcss": [], "silhouette": [], "models": []},
        }

        for k in self.k_range:
            print(f"Testing K={k}...", end=" ")

            # K-means on scaled data ONLY (PCA only used for visualization)
            kmeans_scaled = KMeans(
                n_clusters=k, random_state=42, n_init=10, max_iter=300
            )
            labels_scaled = kmeans_scaled.fit_predict(X_scaled)
            results["scaled"]["wcss"].append(kmeans_scaled.inertia_)
            results["scaled"]["silhouette"].append(
                silhouette_score(X_scaled, labels_scaled)
            )
            results["scaled"]["models"].append(kmeans_scaled)

            print("OK")

        # Find optimal K using scaled data only
        optimal_k_scaled = self.k_range[np.argmax(results["scaled"]["silhouette"])]

        return results, optimal_k_scaled

    def get_best_model(self, results, optimal_k_scaled, X_scaled):
        """Get the best performing K-means model (always uses scaled data)"""
        best_data = X_scaled
        best_k = optimal_k_scaled
        best_model = results["scaled"]["models"][optimal_k_scaled - 2]
        data_type = "Scaled Data"

        return {
            "best_model": best_model,
            "best_k": best_k,
            "best_data_type": data_type,
            "best_data": best_data,
            "optimal_labels": best_model.labels_,
        }

    def run_kmeans_complete_analysis(self, X_scaled, X_pca_50=None, analysis_type="scaled", 
                                   clustering_data=None, visualizer=None):
        """
        Run complete K-means analysis including optimization, training, and evaluation.
        
        Parameters:
        -----------
        X_scaled : array-like
            Scaled feature data for clustering
        X_pca_50 : array-like, optional
            PCA-reduced data (50 components) for comparison analysis
        analysis_type : str, default "scaled"
            Type of analysis - "scaled" for scaled data only, "pca50" for PCA 50D analysis
        clustering_data : dict, optional
            Dictionary containing visualization data and labels
        visualizer : object, optional
            Clustering visualization object with plot_kmeans_analysis method
            
        Returns:
        --------
        dict : Complete analysis results including models, metrics, and labels
        """
        print(f"🚀 Running Complete K-means Analysis ({analysis_type.upper()})")
        print("=" * 60)
        
        # Determine which data to use for clustering
        if analysis_type == "pca50" and X_pca_50 is not None:
            clustering_X = X_pca_50
            data_desc = "PCA 50D"
        else:
            clustering_X = X_scaled
            data_desc = "Scaled"
            
        # Step 1: Find optimal K
        print(f"📊 Step 1: Finding optimal K using {data_desc} data...")
        if analysis_type == "pca50" and X_pca_50 is not None:
            # For PCA analysis, use custom optimization
            results = {"pca50": {"wcss": [], "silhouette": [], "models": []}}
            
            for k in self.k_range:
                print(f"Testing K={k}...", end=" ")
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
                labels = kmeans.fit_predict(clustering_X)
                results["pca50"]["wcss"].append(kmeans.inertia_)
                results["pca50"]["silhouette"].append(silhouette_score(clustering_X, labels))
                results["pca50"]["models"].append(kmeans)
                print("OK")
                
            optimal_k = self.k_range[np.argmax(results["pca50"]["silhouette"])]
            best_model = results["pca50"]["models"][optimal_k - 2]
        else:
            # Use existing scaled data optimization
            results, optimal_k = self.find_optimal_k(X_scaled, X_pca_50)
            best_model = results["scaled"]["models"][optimal_k - 2]
        
        print(f"   ✅ Optimal K found: {optimal_k}")
        
        # Step 2: Get final model and predictions
        print(f"🎯 Step 2: Training final model with K={optimal_k}...")
        final_labels = best_model.predict(clustering_X)
        
        # Calculate metrics
        silhouette_avg = silhouette_score(clustering_X, final_labels)
        inertia = best_model.inertia_
        n_clusters = len(np.unique(final_labels))
        
        print(f"   ✅ Model trained successfully")
        print(f"   📈 Silhouette Score: {silhouette_avg:.3f}")
        print(f"   📊 Inertia: {inertia:.2f}")
        print(f"   🎯 Clusters Found: {n_clusters}")
        
        # Step 3: Visualization (if visualizer and data provided)
        if visualizer and clustering_data:
            print(f"📊 Step 3: Generating visualizations...")
            try:
                if analysis_type == "pca50":
                    # For PCA analysis, use PCA-only visualization
                    visualizer.plot_kmeans_analysis(
                        results={"pca50": results["pca50"]},
                        optimal_k_pca50=optimal_k,
                        clustering_data=clustering_data,
                        use_pca_only=True
                    )
                else:
                    # For scaled analysis, use scaled-only visualization
                    visualizer.plot_kmeans_analysis(
                        results=results,
                        optimal_k_scaled=optimal_k,
                        clustering_data=clustering_data,
                        use_scaled_only=True
                    )
                print(f"   ✅ Visualizations generated successfully")
            except Exception as e:
                print(f"   ⚠️ Visualization warning: {str(e)}")
        
        # Return comprehensive results
        analysis_results = {
            "analysis_type": analysis_type,
            "data_description": data_desc,
            "optimal_k": optimal_k,
            "best_model": best_model,
            "labels": final_labels,
            "silhouette_score": silhouette_avg,
            "inertia": inertia,
            "n_clusters": n_clusters,
            "optimization_results": results,
            "clustering_data_shape": clustering_X.shape
        }
        
        print(f"🎉 Complete K-means analysis finished!")
        print("-" * 60)
        
        return analysis_results


class AdvancedClusteringAnalyzer:
    """Mean Shift and DBSCAN clustering analysis"""

    def run_mean_shift(self, X_pca_50, X_normalized):
        """Run Mean Shift clustering analysis"""
        print("Mean Shift Clustering")
        print("=" * 25)

        # Estimate bandwidth
        bandwidth_pca = estimate_bandwidth(
            X_pca_50, quantile=0.2, n_samples=500, random_state=42
        )
        bandwidth_norm = estimate_bandwidth(
            X_normalized, quantile=0.2, n_samples=500, random_state=42
        )

        # Mean Shift on PCA data
        ms_pca = MeanShift(bandwidth=bandwidth_pca, bin_seeding=True, max_iter=300)
        ms_labels_pca = ms_pca.fit_predict(X_pca_50)
        n_clusters_ms_pca = len(set(ms_labels_pca)) - (1 if -1 in ms_labels_pca else 0)

        # Mean Shift on normalized data (subset for speed)
        subset_size = min(1000, len(X_normalized))
        subset_indices = np.random.choice(len(X_normalized), subset_size, replace=False)
        X_norm_subset = X_normalized[subset_indices]

        ms_norm = MeanShift(
            bandwidth=bandwidth_norm * 10, bin_seeding=True, max_iter=200
        )
        ms_labels_norm = ms_norm.fit_predict(X_norm_subset)
        n_clusters_ms_norm = len(set(ms_labels_norm)) - (
            1 if -1 in ms_labels_norm else 0
        )

        return {
            "pca": {
                "labels": ms_labels_pca,
                "n_clusters": n_clusters_ms_pca,
                "model": ms_pca,
            },
            "normalized": {
                "labels": ms_labels_norm,
                "n_clusters": n_clusters_ms_norm,
                "model": ms_norm,
            },
        }

    def run_dbscan(self, X_pca_50):
        """Run DBSCAN clustering with parameter optimization"""
        print("DBSCAN Clustering")
        print("=" * 20)

        # Calculate k-distance for eps estimation
        k_distances = self._calculate_k_distance(X_pca_50, k=4)
        eps_estimate = np.percentile(k_distances, 90)

        # Test different parameters
        eps_values = [
            eps_estimate * 0.5,
            eps_estimate,
            eps_estimate * 1.5,
            eps_estimate * 2.0,
        ]
        min_samples_values = [3, 5, 8]

        best_dbscan = None
        best_silhouette = -1
        best_params = {}
        best_results = {}

        for eps in eps_values:
            for min_samples in min_samples_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X_pca_50)

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)

                if n_clusters > 1 and n_clusters < len(X_pca_50) - 1:
                    try:
                        silhouette = silhouette_score(X_pca_50, labels)
                        if silhouette > best_silhouette:
                            best_silhouette = silhouette
                            best_dbscan = dbscan
                            best_params = {"eps": eps, "min_samples": min_samples}
                            best_results = {
                                "model": dbscan,
                                "labels": labels,
                                "n_clusters": n_clusters,
                                "n_noise": n_noise,
                                "silhouette": silhouette,
                                "params": best_params,
                            }
                    except:
                        continue

        return best_results if best_dbscan is not None else None

    def _calculate_k_distance(self, X, k=4):
        """Calculate k-distance for DBSCAN eps estimation"""
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(X)
        distances, indices = neigh.kneighbors(X)
        distances = np.sort(distances[:, k - 1], axis=0)
        return distances


class ClusteringEvaluator:
    """Evaluate and interpret clustering results"""

    def evaluate_clustering(self, y_true, cluster_labels, X_data):
        """Calculate clustering evaluation metrics"""
        # Convert labels to consistent numeric format to avoid string/number mix error
        y_true_numeric = np.array([int(str(label)) for label in y_true])
        cluster_labels_numeric = np.array([int(label) for label in cluster_labels])

        silhouette = silhouette_score(X_data, cluster_labels_numeric)
        ari = adjusted_rand_score(y_true_numeric, cluster_labels_numeric)

        return {"silhouette_score": silhouette, "adjusted_rand_index": ari}

    def interpret_clusters(self, y_true, cluster_labels, X_original):
        """Interpret cluster meanings and characteristics"""
        # Convert labels to consistent numeric format to avoid string/number mix error
        y_true_numeric = np.array([int(str(label)) for label in y_true])
        cluster_labels_numeric = np.array([int(label) for label in cluster_labels])

        # Get unique values for proper matrix sizing
        unique_true_classes = sorted(np.unique(y_true_numeric))
        unique_clusters = sorted(np.unique(cluster_labels_numeric))

        n_clusters = len(unique_clusters)
        n_classes = len(unique_true_classes)

        # Create confusion matrix with proper labels parameter for consistent sizing
        cluster_class_matrix = confusion_matrix(
            y_true_numeric,
            cluster_labels_numeric,
            labels=list(range(max(max(unique_true_classes), max(unique_clusters)) + 1)),
        )

        # Trim matrix to only the relevant classes and clusters
        cluster_class_matrix = cluster_class_matrix[
            np.ix_(unique_true_classes, unique_clusters)
        ]

        interpretations = {}

        for cluster_id in unique_clusters:
            mask = cluster_labels_numeric == cluster_id
            cluster_classes = y_true_numeric[mask]

            if len(cluster_classes) > 0:
                unique_classes_in_cluster, counts = np.unique(
                    cluster_classes, return_counts=True
                )
                dominant_class = unique_classes_in_cluster[np.argmax(counts)]
                dominant_percentage = (max(counts) / len(cluster_classes)) * 100

                # Analyze feature characteristics
                cluster_data = X_original[mask]
                cluster_mean = np.mean(cluster_data, axis=0)
                cluster_std = np.std(cluster_data, axis=0)

                # Oil well operation interpretation
                operation_type, description = self._get_operation_interpretation(
                    dominant_class
                )

                interpretations[cluster_id] = {
                    "dominant_class": dominant_class,
                    "dominant_percentage": dominant_percentage,
                    "size": len(cluster_classes),
                    "class_distribution": dict(zip(unique_classes_in_cluster, counts)),
                    "operation_type": operation_type,
                    "description": description,
                    "feature_stats": {"mean": cluster_mean, "std": cluster_std},
                }

        return (
            interpretations,
            cluster_class_matrix,
            unique_true_classes,
            unique_clusters,
        )

    def _get_operation_interpretation(self, dominant_class):
        """Get oil well operation interpretation for a class"""
        if dominant_class == 0:
            return (
                "Normal Operation",
                "Stable oil well operation with standard sensor readings",
            )
        elif dominant_class in [1, 2, 3]:
            return (
                "Flow-related Issues",
                "Problems with flow rates, possibly low production or blockages",
            )
        elif dominant_class in [4, 5, 6]:
            return (
                "Pressure-related Issues",
                "Pressure anomalies, potentially equipment malfunction",
            )
        elif dominant_class in [7, 8, 9]:
            return (
                "Severe Faults",
                "Critical operational issues requiring immediate attention",
            )
        else:
            return "Mixed Operation", "Multiple operational states represented"

    def evaluate_kmeans_final_results(self, kmeans_results, clustering_data):
        """
        Comprehensive K-means final results evaluation and reporting
        
        Parameters:
        -----------
        kmeans_results : dict
            Results from K-means analysis containing model, labels, etc.
        clustering_data : dict
            Complete clustering data containing X_scaled, y_labels, pca_model, etc.
            
        Returns:
        --------
        dict : Updated kmeans_results with additional metrics and centroids_2d
        """
        print("K-means Clustering: Final Results and Evaluation")
        print("=" * 50)

        # Extract results from the new kmeans_results structure
        best_model = kmeans_results["best_model"]
        best_k = kmeans_results["optimal_k"]
        data_type = kmeans_results["data_description"]
        optimal_labels = kmeans_results["labels"]
        best_data = clustering_data["X_scaled"]  # We know it uses scaled data

        print(f"Applying optimal K-means clustering...")
        print(f"   • Method: {data_type}")
        print(f"   • Optimal K: {best_k}")
        print(f"   • PCA usage: Visualization only (2D plots)")

        # Calculate evaluation metrics
        y_true = clustering_data["y_labels"]
        metrics = self.evaluate_clustering(y_true, optimal_labels, best_data)

        print(f"\nK-means Performance Evaluation:")
        print(f"   • Silhouette Score: {metrics['silhouette_score']:.3f}")
        print(f"   • Adjusted Rand Index: {metrics['adjusted_rand_index']:.3f}")
        print(f"   • WCSS: {best_model.inertia_:.2e}")

        # Cluster size analysis
        unique_clusters, cluster_counts = np.unique(optimal_labels, return_counts=True)
        print(f"\nCluster Size Distribution:")
        for cluster, count in zip(unique_clusters, cluster_counts):
            percentage = (count / len(optimal_labels)) * 100
            print(f"   • Cluster {cluster}: {count} samples ({percentage:.1f}%)")

        # Transform centroids from scaled space to 2D PCA for visualization
        # Since clustering was done on scaled data, we need to project centroids to 2D PCA space
        pca_model = clustering_data["pca_model"]
        centroids_2d = pca_model.transform(best_model.cluster_centers_)[:, :2]

        # Update kmeans_results with additional metrics for compatibility
        updated_results = kmeans_results.copy()
        updated_results.update({
            "ari_score": metrics["adjusted_rand_index"],
            "elbow_analysis": kmeans_results.get("optimization_results", {}),
            "centroids_2d": centroids_2d,
            "optimal_k_scaled": best_k,
        })

        print(f"\nK-means clustering analysis complete!")
        print(f"✅ Clustering performed on: SCALED data ({clustering_data['X_scaled'].shape})")
        print(f"✅ PCA used for: 2D visualization only")
        
        return updated_results

    def evaluate_kmeans_pca_final_results(self, kmeans_optimization_pca, clustering_data, scaled_kmeans_results=None):
        """
        Comprehensive K-means PCA 50D final results evaluation and reporting
        
        Parameters:
        -----------
        kmeans_optimization_pca : dict
            Results from K-means PCA optimization containing best_model_info
        clustering_data : dict
            Complete clustering data containing X_pca_50, y_labels, etc.
        scaled_kmeans_results : dict, optional
            Previous scaled K-means results for comparison
            
        Returns:
        --------
        dict : Complete kmeans_results_pca with evaluation metrics
        """
        print("K-means Clustering: PCA 50D Final Results and Evaluation")
        print("=" * 55)

        # Extract optimization results (PCA 50D data)
        best_model_info_pca = kmeans_optimization_pca["best_model_info"]
        best_model_pca = best_model_info_pca["best_model"]
        best_k_pca = best_model_info_pca["best_k"]
        data_type_pca = best_model_info_pca["best_data_type"]
        optimal_labels_pca = best_model_info_pca["optimal_labels"]
        best_data_pca = best_model_info_pca["best_data"]

        print(f"Applying optimal K-means clustering (PCA 50D)...")
        print(f"   • Method: {data_type_pca} (PCA 50D data)")
        print(f"   • Optimal K: {best_k_pca}")
        print(f"   • Feature dimensions: {clustering_data['X_pca_50'].shape[1]} PCA components")

        # Calculate evaluation metrics
        y_true = clustering_data["y_labels"]
        metrics_pca = self.evaluate_clustering(y_true, optimal_labels_pca, best_data_pca)

        print(f"\nK-means Performance Evaluation (PCA 50D):")
        print(f"   • Silhouette Score: {metrics_pca['silhouette_score']:.3f}")
        print(f"   • Adjusted Rand Index: {metrics_pca['adjusted_rand_index']:.3f}")
        print(f"   • WCSS: {best_model_pca.inertia_:.2e}")

        # Cluster size analysis
        unique_clusters_pca, cluster_counts_pca = np.unique(optimal_labels_pca, return_counts=True)
        print(f"\nCluster Size Distribution (PCA 50D):")
        for cluster, count in zip(unique_clusters_pca, cluster_counts_pca):
            percentage = (count / len(optimal_labels_pca)) * 100
            print(f"   • Cluster {cluster}: {count} samples ({percentage:.1f}%)")

        # Transform centroids to 2D PCA for visualization
        # Since clustering was done on PCA 50D, take first 2 components for visualization
        centroids_2d_pca = best_model_pca.cluster_centers_[:, :2]

        # Store K-means PCA results
        kmeans_results_pca = {
            "best_model": best_model_pca,
            "best_k": best_k_pca,
            "best_data_type": data_type_pca,
            "optimal_labels": optimal_labels_pca,
            "silhouette_score": metrics_pca["silhouette_score"],
            "ari_score": metrics_pca["adjusted_rand_index"],
            "elbow_analysis": kmeans_optimization_pca["results"],
            "centroids_2d": centroids_2d_pca,
            "optimal_k_pca": kmeans_optimization_pca["optimal_k_pca"],
        }

        print(f"\nK-means clustering analysis (PCA 50D) complete!")
        print(f"✅ Clustering performed on: PCA 50D data ({clustering_data['X_pca_50'].shape})")
        print(f"✅ Centroids available in: 2D PCA space for visualization")

        # Compare with previous scaled data results if provided
        if scaled_kmeans_results is not None:
            print(f"\n📊 COMPARISON: Scaled Data vs PCA 50D Data")
            print("=" * 45)
            print(f"   • Scaled Data   - K: {scaled_kmeans_results['optimal_k']}, Silhouette: {scaled_kmeans_results['silhouette_score']:.3f}, ARI: {scaled_kmeans_results['ari_score']:.3f}")
            print(f"   • PCA 50D Data  - K: {best_k_pca}, Silhouette: {metrics_pca['silhouette_score']:.3f}, ARI: {metrics_pca['adjusted_rand_index']:.3f}")
            
            # Determine better approach
            if metrics_pca['silhouette_score'] > scaled_kmeans_results['silhouette_score']:
                print(f"   🏆 PCA 50D performs better (higher silhouette score)")
            elif metrics_pca['silhouette_score'] < scaled_kmeans_results['silhouette_score']:
                print(f"   🏆 Scaled Data performs better (higher silhouette score)")
            else:
                print(f"   🤝 Both methods perform similarly")

        return kmeans_results_pca


class AdvancedClusteringSuite:
    """Advanced clustering implementation with multiple algorithms and optimizations"""

    def __init__(self):
        self.results = {}

    def run_advanced_clustering_suite(self, X_data, y_true=None, max_k=10):
        """
        Run advanced clustering suite with multiple algorithms and optimizations

        Parameters:
        -----------
        X_data : array-like
            Data to cluster
        y_true : array-like, optional
            True labels for evaluation
        max_k : int
            Maximum number of clusters to test

        Returns:
        --------
        dict : Results from all clustering methods
        """
        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
        from sklearn.mixture import GaussianMixture
        from sklearn.neighbors import NearestNeighbors
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        import time

        results = {}

        print("🔬 Advanced Clustering Suite")
        print("=" * 40)

        # 1. OPTIMIZED K-MEANS WITH MULTIPLE INITIALIZATIONS
        print("1. Enhanced K-means with K-means++")
        best_kmeans = self._run_enhanced_kmeans(X_data, max_k)
        results["enhanced_kmeans"] = best_kmeans
        if best_kmeans:
            print(
                f"   ✓ Best K: {best_kmeans['k']}, Silhouette: {best_kmeans['silhouette']:.3f}"
            )

        # 2. GAUSSIAN MIXTURE MODELS
        print("2. Gaussian Mixture Models (GMM)")
        best_gmm = self._run_gaussian_mixture(X_data, max_k)
        results["gmm"] = best_gmm
        if best_gmm:
            print(
                f"   ✓ Best Components: {best_gmm['n_components']}, Silhouette: {best_gmm['silhouette']:.3f}"
            )

        # 3. HIERARCHICAL CLUSTERING WITH LINKAGE COMPARISON
        print("3. Hierarchical Clustering (Multiple Linkages)")
        best_hierarchical = self._run_hierarchical_clustering(X_data, max_k)
        results["hierarchical"] = best_hierarchical
        if best_hierarchical:
            print(
                f"   ✓ Best: {best_hierarchical['n_clusters']} clusters, {best_hierarchical['linkage']} linkage"
            )
            print(f"     Silhouette: {best_hierarchical['silhouette']:.3f}")

        # 4. OPTIMIZED DBSCAN WITH PARAMETER GRID SEARCH
        print("4. Optimized DBSCAN with Grid Search")
        best_dbscan = self._run_optimized_dbscan(X_data)
        results["optimized_dbscan"] = best_dbscan
        if best_dbscan:
            print(
                f"   ✓ Best: eps={best_dbscan['eps']:.3f}, min_samples={best_dbscan['min_samples']}"
            )
            print(
                f"     Clusters: {best_dbscan['n_clusters']}, Noise: {best_dbscan['n_noise']}"
            )
            print(f"     Silhouette: {best_dbscan['silhouette']:.3f}")
        else:
            print("   ⚠️ No suitable DBSCAN parameters found")

        print(f"\n🏆 Advanced Clustering Complete!")
        return results

    def _run_enhanced_kmeans(self, X_data, max_k):
        """Run enhanced K-means with multiple initializations"""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        import time

        best_kmeans = None
        best_kmeans_score = -1

        for k in range(2, max_k + 1):
            start = time.time()
            # Multiple random initializations for robustness
            kmeans = KMeans(
                n_clusters=k, init="k-means++", n_init=20, random_state=42, max_iter=300
            )
            labels = kmeans.fit_predict(X_data)

            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(X_data, labels)
                calinski = calinski_harabasz_score(X_data, labels)

                if silhouette > best_kmeans_score:
                    best_kmeans_score = silhouette
                    best_kmeans = {
                        "k": k,
                        "labels": labels,
                        "model": kmeans,
                        "silhouette": silhouette,
                        "calinski": calinski,
                        "time": time.time() - start,
                    }

        return best_kmeans

    def _run_gaussian_mixture(self, X_data, max_k):
        """Run Gaussian Mixture Models"""
        from sklearn.mixture import GaussianMixture
        from sklearn.metrics import silhouette_score
        import time

        best_gmm = None
        best_gmm_score = -1

        for n_components in range(2, max_k + 1):
            start = time.time()
            try:
                gmm = GaussianMixture(
                    n_components=n_components, random_state=42, max_iter=100
                )
                labels = gmm.fit_predict(X_data)

                if len(np.unique(labels)) > 1:
                    silhouette = silhouette_score(X_data, labels)
                    bic = gmm.bic(X_data)
                    aic = gmm.aic(X_data)

                    if silhouette > best_gmm_score:
                        best_gmm_score = silhouette
                        best_gmm = {
                            "n_components": n_components,
                            "labels": labels,
                            "model": gmm,
                            "silhouette": silhouette,
                            "bic": bic,
                            "aic": aic,
                            "time": time.time() - start,
                        }
            except Exception as e:
                continue

        return best_gmm

    def _run_hierarchical_clustering(self, X_data, max_k):
        """Run hierarchical clustering with multiple linkages"""
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        import time

        linkages = ["ward", "complete", "average"]
        best_hierarchical = None
        best_hier_score = -1

        for linkage in linkages:
            for n_clusters in range(
                2, min(max_k + 1, 8)
            ):  # Limit for computational efficiency
                start = time.time()
                try:
                    hierarchical = AgglomerativeClustering(
                        n_clusters=n_clusters, linkage=linkage
                    )
                    labels = hierarchical.fit_predict(X_data)

                    if len(np.unique(labels)) > 1:
                        silhouette = silhouette_score(X_data, labels)

                        if silhouette > best_hier_score:
                            best_hier_score = silhouette
                            best_hierarchical = {
                                "n_clusters": n_clusters,
                                "labels": labels,
                                "linkage": linkage,
                                "silhouette": silhouette,
                                "time": time.time() - start,
                            }
                except Exception as e:
                    continue

        return best_hierarchical

    def _run_optimized_dbscan(self, X_data):
        """Run DBSCAN with optimized parameters"""
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors
        from sklearn.metrics import silhouette_score
        import time

        # Calculate optimal eps using k-distance
        k = 4  # MinPts - 1
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(X_data)
        distances, indices = neighbors_fit.kneighbors(X_data)
        distances = np.sort(distances[:, k - 1], axis=0)

        # Find knee point (simplified approach)
        knee_idx = int(len(distances) * 0.95)  # 95th percentile as heuristic
        optimal_eps = distances[knee_idx]

        eps_range = np.linspace(optimal_eps * 0.5, optimal_eps * 2, 10)
        min_samples_range = [3, 4, 5, 6, 8, 10]

        best_dbscan = None
        best_dbscan_score = -1

        for eps in eps_range:
            for min_samples in min_samples_range:
                start = time.time()
                try:
                    dbscan = DBSCAN(
                        eps=eps, min_samples=min_samples, metric="euclidean"
                    )
                    labels = dbscan.fit_predict(X_data)

                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)

                    if (
                        n_clusters > 1 and n_noise < len(X_data) * 0.9
                    ):  # Avoid too much noise
                        silhouette = silhouette_score(X_data, labels)
                        if silhouette > best_dbscan_score:
                            best_dbscan_score = silhouette
                            best_dbscan = {
                                "eps": eps,
                                "min_samples": min_samples,
                                "labels": labels,
                                "n_clusters": n_clusters,
                                "n_noise": n_noise,
                                "silhouette": silhouette,
                                "time": time.time() - start,
                            }
                except Exception as e:
                    continue

        return best_dbscan


def evaluate_clustering_accuracy(y_true, clustering_results, method_names=None):
    """
    Evaluate clustering accuracy against ground truth using Hungarian algorithm.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth labels
    clustering_results : dict or list
        Dictionary with method names as keys and results as values, 
        or list of clustering labels arrays
    method_names : list, optional
        Names for the clustering methods (used when clustering_results is a list)
        
    Returns:
    --------
    dict : Accuracy results for each clustering method
    """
    from scipy.optimize import linear_sum_assignment
    
    print("🎯 Evaluating Clustering Accuracy Against Ground Truth")
    print("=" * 55)
    
    # Convert ground truth to numeric
    y_true_numeric = np.array([
        int(str(label)) if isinstance(label, (str, np.str_)) else label
        for label in y_true
    ])
    
    def calculate_clustering_accuracy(y_true, y_pred):
        """Calculate clustering accuracy using optimal label assignment"""
        true_labels, pred_labels = np.unique(y_true), np.unique(y_pred)
        n_true, n_pred = len(true_labels), len(pred_labels)
        
        # Create confusion matrix
        confusion_matrix = np.zeros((n_true, n_pred))
        for i, true_label in enumerate(true_labels):
            for j, pred_label in enumerate(pred_labels):
                confusion_matrix[i, j] = np.sum(
                    (y_true == true_label) & (y_pred == pred_label)
                )
        
        # Use Hungarian algorithm for optimal assignment
        row_indices, col_indices = linear_sum_assignment(-confusion_matrix)
        accuracy = confusion_matrix[row_indices, col_indices].sum() / len(y_true)
        
        return accuracy, confusion_matrix, row_indices, col_indices
    
    # Handle different input formats
    if isinstance(clustering_results, dict):
        methods_to_evaluate = clustering_results.items()
    elif isinstance(clustering_results, list):
        if method_names is None:
            method_names = [f"Method_{i+1}" for i in range(len(clustering_results))]
        methods_to_evaluate = zip(method_names, clustering_results)
    else:
        raise ValueError("clustering_results must be dict or list")
    
    print("📊 Clustering Method Accuracy Results:")
    print("-" * 50)
    
    accuracy_results = {}
    
    # Evaluate each method
    for method_name, method_result in methods_to_evaluate:
        try:
            # Extract labels from different result formats
            if isinstance(method_result, dict):
                if "labels" in method_result:
                    labels = method_result["labels"]
                elif "optimal_labels" in method_result:
                    labels = method_result["optimal_labels"]
                else:
                    # Try to find labels in nested structure
                    labels = None
                    for key, value in method_result.items():
                        if isinstance(value, dict) and "labels" in value:
                            labels = value["labels"]
                            break
                    if labels is None:
                        print(f"   ❌ {method_name}: Could not find labels in result")
                        continue
            else:
                # Assume it's already the labels array
                labels = method_result
            
            # Convert labels to numeric
            labels_numeric = np.array([
                int(str(label)) if isinstance(label, (str, np.str_)) else label
                for label in labels
            ])
            
            # Calculate accuracy
            accuracy, conf_matrix, row_ind, col_ind = calculate_clustering_accuracy(
                y_true_numeric, labels_numeric
            )
            
            # Store results
            accuracy_results[method_name] = {
                "accuracy": accuracy,
                "n_clusters": len(np.unique(labels_numeric)),
                "confusion_matrix": conf_matrix,
                "assignment": (row_ind, col_ind),
                "labels": labels_numeric
            }
            
            print(f"   • {method_name:<25}: {accuracy:.3f} ({accuracy*100:.1f}%) | Clusters: {len(np.unique(labels_numeric))}")
            
        except Exception as e:
            print(f"   ❌ {method_name}: Error - {str(e)}")
            continue
    
    # Display summary
    if accuracy_results:
        best_method = max(accuracy_results.items(), key=lambda x: x[1]["accuracy"])
        unique_true_classes = np.unique(y_true_numeric)
        
        print(f"\n🏆 Best Performing Method: {best_method[0]} ({best_method[1]['accuracy']*100:.1f}%)")
        print(f"📊 Ground Truth Classes: {len(unique_true_classes)} | Total Samples: {len(y_true_numeric)}")
    else:
        print("\n❌ No valid clustering results to evaluate")
    
    print(f"\n✅ Accuracy evaluation complete!")
    
    return accuracy_results