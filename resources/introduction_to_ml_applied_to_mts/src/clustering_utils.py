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
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate
import warnings
warnings.filterwarnings('ignore')


class ClusteringDataLoader:
    """Load and prepare 3W dataset for clustering analysis"""
    
    def __init__(self, target_length=500, max_files_per_class=20, enable_sampling=True):
        self.target_length = target_length
        self.max_files_per_class = max_files_per_class
        self.enable_sampling = enable_sampling
    
    def load_raw_data(self, persistence, raw_data_dir):
        """Load raw 3W dataset from pickle/parquet files"""
        print("Loading Raw 3W Dataset for Clustering Analysis")
        print("=" * 55)
        
        # Get fold directories
        fold_dirs = [
            d for d in os.listdir(raw_data_dir)
            if d.startswith("fold_") and os.path.isdir(os.path.join(raw_data_dir, d))
        ]
        fold_dirs.sort()
        
        if not fold_dirs:
            raise FileNotFoundError("No fold directories found")
        
        # Use the first fold for clustering analysis
        selected_fold = fold_dirs[0]
        fold_path = os.path.join(raw_data_dir, selected_fold)
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
                raw_dfs, raw_classes = persistence._load_dataframes(pickle_file, "pickle")
            elif os.path.exists(parquet_file):
                raw_dfs, raw_classes = persistence._load_from_parquet(parquet_file)
            else:
                continue
            
            if raw_dfs is None or len(raw_dfs) == 0:
                continue
            
            # Sample files if needed
            if self.enable_sampling and len(raw_dfs) > self.max_files_per_class * 10:
                n_samples = min(self.max_files_per_class * 10, len(raw_dfs))
                selected_indices = np.random.choice(len(raw_dfs), n_samples, replace=False)
                raw_dfs = [raw_dfs[i] for i in selected_indices]
                raw_classes = [raw_classes[i] for i in selected_indices]
            
            # Process each time series
            processed_data = self._process_time_series(raw_dfs, raw_classes, dataset_name)
            all_time_series.extend(processed_data['time_series'])
            all_class_labels.extend(processed_data['labels'])
            all_file_info.extend(processed_data['file_info'])
        
        # Convert to numpy arrays
        X_original = np.array(all_time_series)
        y_labels = np.array(all_class_labels)
        
        return {
            'X_original': X_original,
            'y_labels': y_labels,
            'file_info': all_file_info,
            'fold_used': selected_fold
        }
    
    def _process_time_series(self, raw_dfs, raw_classes, dataset_name):
        """Process individual time series data"""
        time_series = []
        labels = []
        file_info = []
        
        for i, (df, class_label) in enumerate(zip(raw_dfs, raw_classes)):
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
                            original_indices, series, kind="linear",
                            bounds_error=False, fill_value="extrapolate"
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
                file_info.append({
                    'dataset': dataset_name,
                    'index': i,
                    'original_length': original_length,
                    'sensors_used': valid_sensors,
                    'class': class_label
                })
                
            except Exception as e:
                continue
        
        return {
            'time_series': time_series,
            'labels': labels,
            'file_info': file_info
        }


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
            'X_original': X_clean,
            'X_scaled': X_scaled,
            'X_normalized': X_normalized,
            'X_pca_50': X_pca[:, :50],
            'X_pca_95': X_pca[:, :n_components_95],
            'X_pca_2d': X_pca[:, :2],
            'pca_model': self.pca,
            'scaler_standard': self.scaler_standard,
            'scaler_minmax': self.scaler_minmax,
            'variance_info': {
                'n_components_90': n_components_90,
                'n_components_95': n_components_95,
                'cumulative_variance': cumulative_variance
            }
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
    
    def find_optimal_k(self, X_scaled, X_pca_50):
        """Find optimal K using elbow method and silhouette analysis"""
        print("K-means: Finding Optimal K")
        print("=" * 30)
        
        results = {
            'scaled': {'wcss': [], 'silhouette': [], 'models': []},
            'pca_50': {'wcss': [], 'silhouette': [], 'models': []}
        }
        
        for k in self.k_range:
            print(f"Testing K={k}...", end=" ")
            
            # K-means on scaled data
            kmeans_scaled = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            labels_scaled = kmeans_scaled.fit_predict(X_scaled)
            results['scaled']['wcss'].append(kmeans_scaled.inertia_)
            results['scaled']['silhouette'].append(silhouette_score(X_scaled, labels_scaled))
            results['scaled']['models'].append(kmeans_scaled)
            
            # K-means on PCA data
            kmeans_pca = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            labels_pca = kmeans_pca.fit_predict(X_pca_50)
            results['pca_50']['wcss'].append(kmeans_pca.inertia_)
            results['pca_50']['silhouette'].append(silhouette_score(X_pca_50, labels_pca))
            results['pca_50']['models'].append(kmeans_pca)
            
            print("OK")
        
        # Find optimal K
        optimal_k_scaled = self.k_range[np.argmax(results['scaled']['silhouette'])]
        optimal_k_pca = self.k_range[np.argmax(results['pca_50']['silhouette'])]
        
        return results, optimal_k_scaled, optimal_k_pca
    
    def get_best_model(self, results, optimal_k_scaled, optimal_k_pca, X_scaled, X_pca_50):
        """Get the best performing K-means model"""
        max_silhouette_scaled = max(results['scaled']['silhouette'])
        max_silhouette_pca = max(results['pca_50']['silhouette'])
        
        if max_silhouette_scaled >= max_silhouette_pca:
            best_data = X_scaled
            best_k = optimal_k_scaled
            best_model = results['scaled']['models'][optimal_k_scaled - 2]
            data_type = "Scaled"
        else:
            best_data = X_pca_50
            best_k = optimal_k_pca
            best_model = results['pca_50']['models'][optimal_k_pca - 2]
            data_type = "PCA 50D"
        
        return {
            'best_model': best_model,
            'best_k': best_k,
            'best_data_type': data_type,
            'best_data': best_data,
            'optimal_labels': best_model.labels_
        }


class AdvancedClusteringAnalyzer:
    """Mean Shift and DBSCAN clustering analysis"""
    
    def run_mean_shift(self, X_pca_50, X_normalized):
        """Run Mean Shift clustering analysis"""
        print("Mean Shift Clustering")
        print("=" * 25)
        
        # Estimate bandwidth
        bandwidth_pca = estimate_bandwidth(X_pca_50, quantile=0.2, n_samples=500, random_state=42)
        bandwidth_norm = estimate_bandwidth(X_normalized, quantile=0.2, n_samples=500, random_state=42)
        
        # Mean Shift on PCA data
        ms_pca = MeanShift(bandwidth=bandwidth_pca, bin_seeding=True, max_iter=300)
        ms_labels_pca = ms_pca.fit_predict(X_pca_50)
        n_clusters_ms_pca = len(set(ms_labels_pca)) - (1 if -1 in ms_labels_pca else 0)
        
        # Mean Shift on normalized data (subset for speed)
        subset_size = min(1000, len(X_normalized))
        subset_indices = np.random.choice(len(X_normalized), subset_size, replace=False)
        X_norm_subset = X_normalized[subset_indices]
        
        ms_norm = MeanShift(bandwidth=bandwidth_norm * 10, bin_seeding=True, max_iter=200)
        ms_labels_norm = ms_norm.fit_predict(X_norm_subset)
        n_clusters_ms_norm = len(set(ms_labels_norm)) - (1 if -1 in ms_labels_norm else 0)
        
        return {
            'pca': {
                'labels': ms_labels_pca,
                'n_clusters': n_clusters_ms_pca,
                'model': ms_pca
            },
            'normalized': {
                'labels': ms_labels_norm,
                'n_clusters': n_clusters_ms_norm,
                'model': ms_norm
            }
        }
    
    def run_dbscan(self, X_pca_50):
        """Run DBSCAN clustering with parameter optimization"""
        print("DBSCAN Clustering")
        print("=" * 20)
        
        # Calculate k-distance for eps estimation
        k_distances = self._calculate_k_distance(X_pca_50, k=4)
        eps_estimate = np.percentile(k_distances, 90)
        
        # Test different parameters
        eps_values = [eps_estimate * 0.5, eps_estimate, eps_estimate * 1.5, eps_estimate * 2.0]
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
                            best_params = {'eps': eps, 'min_samples': min_samples}
                            best_results = {
                                'model': dbscan,
                                'labels': labels,
                                'n_clusters': n_clusters,
                                'n_noise': n_noise,
                                'silhouette': silhouette,
                                'params': best_params
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
        
        return {
            'silhouette_score': silhouette,
            'adjusted_rand_index': ari
        }
    
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
            labels=list(range(max(max(unique_true_classes), max(unique_clusters)) + 1))
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
                unique_classes_in_cluster, counts = np.unique(cluster_classes, return_counts=True)
                dominant_class = unique_classes_in_cluster[np.argmax(counts)]
                dominant_percentage = (max(counts) / len(cluster_classes)) * 100
                
                # Analyze feature characteristics
                cluster_data = X_original[mask]
                cluster_mean = np.mean(cluster_data, axis=0)
                cluster_std = np.std(cluster_data, axis=0)
                
                # Oil well operation interpretation
                operation_type, description = self._get_operation_interpretation(dominant_class)
                
                interpretations[cluster_id] = {
                    'dominant_class': dominant_class,
                    'dominant_percentage': dominant_percentage,
                    'size': len(cluster_classes),
                    'class_distribution': dict(zip(unique_classes_in_cluster, counts)),
                    'operation_type': operation_type,
                    'description': description,
                    'feature_stats': {
                        'mean': cluster_mean,
                        'std': cluster_std
                    }
                }
        
        return interpretations, cluster_class_matrix, unique_true_classes, unique_clusters
    
    def _get_operation_interpretation(self, dominant_class):
        """Get oil well operation interpretation for a class"""
        if dominant_class == 0:
            return "Normal Operation", "Stable oil well operation with standard sensor readings"
        elif dominant_class in [1, 2, 3]:
            return "Flow-related Issues", "Problems with flow rates, possibly low production or blockages"
        elif dominant_class in [4, 5, 6]:
            return "Pressure-related Issues", "Pressure anomalies, potentially equipment malfunction"
        elif dominant_class in [7, 8, 9]:
            return "Severe Faults", "Critical operational issues requiring immediate attention"
        else:
            return "Mixed Operation", "Multiple operational states represented"