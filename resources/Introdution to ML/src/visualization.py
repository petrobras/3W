"""
Visualization Module for 3W Dataset

This module provides comprehensive visualization utilities for analyzing
sensor data and scaling methods in the 3W dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple


class DataVisualizer:
    """
    A class for creating visualizations of the 3W dataset.
    
    Provides methods for plotting raw data, scaled data, distributions,
    and comparative analyses.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the DataVisualizer.
        
        Args:
            style (str): Matplotlib style to use
            figsize (Tuple[int, int]): Default figure size
        """
        # Set matplotlib style
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use('default')
        
        self.default_figsize = figsize
        self.color_palette = {
            'pressure': '#FF6B6B',
            'temperature': '#4ECDC4',
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01'
        }
    
    def plot_dataset_overview(self, classes: List[str], dataset_stats: Dict[str, Any]) -> None:
        """
        Plot overview of dataset structure and class distribution.
        
        Args:
            classes (List[str]): List of class labels
            dataset_stats (Dict): Dataset statistics from DataLoader
        """
        print("🔍 Dataset Analysis")
        print("=" * 50)
        
        # Basic statistics
        print(f"Total number of dataframes: {len(classes)}")
        print(f"Class distribution: {Counter(classes)}")
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('3W Dataset Overview', fontsize=16, fontweight='bold')
        
        # Plot 1: Class distribution
        class_counts = Counter(classes)
        class_names = list(class_counts.keys())
        counts = list(class_counts.values())
        
        axes[0].bar(class_names, counts, color=self.color_palette['primary'], alpha=0.7)
        axes[0].set_title('Distribution of Files by Class')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Number of Files')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Real vs Simulated distribution
        if 'real_files' in dataset_stats and 'simulated_files' in dataset_stats:
            real_count = dataset_stats['real_files']
            simulated_count = dataset_stats['simulated_files']
            
            axes[1].pie([real_count, simulated_count], 
                       labels=['Real Data', 'Simulated Data'], 
                       colors=[self.color_palette['accent'], self.color_palette['secondary']],
                       autopct='%1.1f%%')
            axes[1].set_title('Real vs Simulated Data Distribution')
        
        plt.tight_layout()
        plt.show()
        
        # Print memory usage if available
        if 'total_samples' in dataset_stats:
            print(f"\\n💾 Total samples: {dataset_stats['total_samples']:,}")
    
    def plot_raw_data_analysis(self, data: pd.DataFrame, pressure_col: str = 'P-TPT', 
                              temp_col: str = 'T-TPT', title_suffix: str = "") -> None:
        """
        Plot comprehensive analysis of raw sensor data.
        
        Args:
            data (pd.DataFrame): Raw sensor data
            pressure_col (str): Name of pressure column
            temp_col (str): Name of temperature column
            title_suffix (str): Additional text for plot title
        """
        print(f"\\n🔸 Plotting Raw Data Analysis{title_suffix}...")
        
        # Check if required columns exist
        if pressure_col not in data.columns or temp_col not in data.columns:
            print(f"❌ Required columns {pressure_col} and/or {temp_col} not found in data")
            return
        
        # Clean data
        clean_data = data[[pressure_col, temp_col]].dropna()
        
        if len(clean_data) == 0:
            print("❌ No valid data found after removing NaN values")
            return
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Raw Data Analysis - Pressure and Temperature{title_suffix}', 
                    fontsize=16, fontweight='bold')
        
        # Time series plots
        axes[0, 0].plot(clean_data[pressure_col], color=self.color_palette['pressure'], 
                       alpha=0.7, linewidth=0.5)
        axes[0, 0].set_title(f'Raw Pressure ({pressure_col}) Time Series')
        axes[0, 0].set_xlabel('Sample Index')
        axes[0, 0].set_ylabel('Pressure Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(clean_data[temp_col], color=self.color_palette['temperature'], 
                       alpha=0.7, linewidth=0.5)
        axes[0, 1].set_title(f'Raw Temperature ({temp_col}) Time Series')
        axes[0, 1].set_xlabel('Sample Index')
        axes[0, 1].set_ylabel('Temperature Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Distribution plots
        axes[1, 0].hist(clean_data[pressure_col], bins=50, color=self.color_palette['pressure'], 
                       alpha=0.7, density=True)
        axes[1, 0].set_title(f'Raw Pressure Distribution')
        axes[1, 0].set_xlabel('Pressure Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].hist(clean_data[temp_col], bins=50, color=self.color_palette['temperature'], 
                       alpha=0.7, density=True)
        axes[1, 1].set_title(f'Raw Temperature Distribution')
        axes[1, 1].set_xlabel('Temperature Value')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"\\n📈 Data Statistics:")
        print(f"Shape: {clean_data.shape}")
        print(clean_data.describe())
    
    def plot_scaled_data_analysis(self, scaled_data: pd.DataFrame, scaler_name: str,
                                 pressure_col: str = None, temp_col: str = None) -> None:
        """
        Plot analysis of scaled data for a specific scaling method.
        
        Args:
            scaled_data (pd.DataFrame): Scaled sensor data
            scaler_name (str): Name of the scaling method
            pressure_col (str): Name of scaled pressure column
            temp_col (str): Name of scaled temperature column
        """
        print(f"\\n🔸 Plotting {scaler_name} Analysis...")
        
        # Auto-detect columns if not provided
        if pressure_col is None or temp_col is None:
            numeric_cols = [col for col in scaled_data.columns if col != 'class']
            if len(numeric_cols) >= 2:
                pressure_col = pressure_col or numeric_cols[0]
                temp_col = temp_col or numeric_cols[1]
            else:
                print(f"❌ Insufficient numeric columns for plotting")
                return
        
        # Check if columns exist
        if pressure_col not in scaled_data.columns or temp_col not in scaled_data.columns:
            print(f"❌ Required columns {pressure_col} and/or {temp_col} not found")
            return
        
        # Clean data
        clean_data = scaled_data[[pressure_col, temp_col]].dropna()
        
        if len(clean_data) == 0:
            print("❌ No valid data found after removing NaN values")
            return
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{scaler_name} - Scaled Data Analysis', fontsize=16, fontweight='bold')
        
        # Time series plots
        axes[0, 0].plot(clean_data[pressure_col], color=self.color_palette['pressure'], 
                       alpha=0.7, linewidth=0.5)
        axes[0, 0].set_title(f'Scaled Pressure ({scaler_name})')
        axes[0, 0].set_xlabel('Sample Index')
        axes[0, 0].set_ylabel('Scaled Pressure Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(clean_data[temp_col], color=self.color_palette['temperature'], 
                       alpha=0.7, linewidth=0.5)
        axes[0, 1].set_title(f'Scaled Temperature ({scaler_name})')
        axes[0, 1].set_xlabel('Sample Index')
        axes[0, 1].set_ylabel('Scaled Temperature Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Distribution plots
        axes[1, 0].hist(clean_data[pressure_col], bins=50, color=self.color_palette['pressure'], 
                       alpha=0.7, density=True)
        axes[1, 0].set_title(f'Scaled Pressure Distribution ({scaler_name})')
        axes[1, 0].set_xlabel('Scaled Pressure Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].hist(clean_data[temp_col], bins=50, color=self.color_palette['temperature'], 
                       alpha=0.7, density=True)
        axes[1, 1].set_title(f'Scaled Temperature Distribution ({scaler_name})')
        axes[1, 1].set_xlabel('Scaled Temperature Value')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"Statistics for {scaler_name}:")
        print(f"Pressure - Mean: {clean_data[pressure_col].mean():.4f}, Std: {clean_data[pressure_col].std():.4f}")
        print(f"Temperature - Mean: {clean_data[temp_col].mean():.4f}, Std: {clean_data[temp_col].std():.4f}")
    
    def plot_scaling_comparison(self, scaling_results: Dict[str, pd.DataFrame], 
                               pressure_col: str = 'P-TPT', temp_col: str = 'T-TPT') -> None:
        """
        Create comparative plots for different scaling methods.
        
        Args:
            scaling_results (Dict): Dictionary with method names and scaled dataframes
            pressure_col (str): Name of pressure column
            temp_col (str): Name of temperature column
        """
        print(f"\\n🔸 Creating scaling comparison plots...")
        
        # Create comparison boxplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Scaling Methods Comparison - Boxplots', fontsize=16, fontweight='bold')
        
        # Prepare data for boxplots
        method_names = list(scaling_results.keys())
        
        # Pressure comparison
        pressure_data = []
        temp_data = []
        valid_methods = []
        
        for method, data in scaling_results.items():
            # Handle different column naming conventions
            p_col = pressure_col if pressure_col in data.columns else f"{pressure_col}_scaled"
            t_col = temp_col if temp_col in data.columns else f"{temp_col}_scaled"
            
            if p_col in data.columns and t_col in data.columns:
                pressure_data.append(data[p_col].dropna().values)
                temp_data.append(data[t_col].dropna().values)
                valid_methods.append(method)
        
        if not valid_methods:
            print("❌ No valid data found for comparison plots")
            return
        
        # Pressure boxplot
        bp1 = axes[0].boxplot(pressure_data, labels=valid_methods, patch_artist=True)
        for patch in bp1['boxes']:
            patch.set_facecolor(self.color_palette['pressure'])
            patch.set_alpha(0.7)
        axes[0].set_title('Pressure - All Scaling Methods')
        axes[0].set_ylabel('Scaled Values')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Temperature boxplot
        bp2 = axes[1].boxplot(temp_data, labels=valid_methods, patch_artist=True)
        for patch in bp2['boxes']:
            patch.set_facecolor(self.color_palette['temperature'])
            patch.set_alpha(0.7)
        axes[1].set_title('Temperature - All Scaling Methods')
        axes[1].set_ylabel('Scaled Values')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, data: pd.DataFrame, title: str = "Feature Correlation Matrix") -> None:
        """
        Plot correlation matrix heatmap for numeric features.
        
        Args:
            data (pd.DataFrame): Input data
            title (str): Plot title
        """
        # Get numeric columns only
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            print("❌ No numeric columns found for correlation analysis")
            return
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_time_series_comparison(self, data_dict: Dict[str, pd.DataFrame], 
                                   column: str, max_samples: int = 1000) -> None:
        """
        Plot time series comparison of the same column across different scaling methods.
        
        Args:
            data_dict (Dict): Dictionary with method names and dataframes
            column (str): Column to plot
            max_samples (int): Maximum number of samples to plot (for performance)
        """
        plt.figure(figsize=(15, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(data_dict)))
        
        for i, (method, data) in enumerate(data_dict.items()):
            # Handle different column naming conventions
            col_name = column if column in data.columns else f"{column}_scaled"
            
            if col_name in data.columns:
                # Subsample if data is too large
                plot_data = data[col_name].dropna()
                if len(plot_data) > max_samples:
                    indices = np.linspace(0, len(plot_data)-1, max_samples, dtype=int)
                    plot_data = plot_data.iloc[indices]
                
                plt.plot(plot_data, label=method, color=colors[i], alpha=0.7, linewidth=1)
        
        plt.title(f'Time Series Comparison - {column}', fontsize=14, fontweight='bold')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def print_scaling_summary(self) -> None:
        """Print a summary of different scaling methods and their use cases."""
        print(f"\\n📋 Scaling Methods Summary:")
        print("=" * 60)
        print("🔹 StandardScaler: Mean=0, Std=1 (assumes normal distribution)")
        print("🔹 MinMaxScaler: Scales to [0,1] range (preserves relationships)")
        print("🔹 RobustScaler: Uses median and IQR (robust to outliers)")
        print("🔹 Normalizer: Unit vector scaling (preserves direction)")
        print("\\n💡 When to use each:")
        print("- StandardScaler: Normal data, algorithms that assume normality (SVM, Neural Networks)")
        print("- MinMaxScaler: Need bounded values, preserve zero, distance-based algorithms")
        print("- RobustScaler: Data with outliers, non-normal distributions")
        print("- Normalizer: When direction matters more than magnitude (text analysis, clustering)")
    
    def create_summary_report(self, original_data: pd.DataFrame, scaled_results: Dict[str, pd.DataFrame], 
                             dataset_info: Dict[str, Any]) -> None:
        """
        Create a comprehensive summary report with visualizations.
        
        Args:
            original_data (pd.DataFrame): Original data
            scaled_results (Dict): Results from different scaling methods
            dataset_info (Dict): Dataset information
        """
        print("\\n📊 Creating Comprehensive Summary Report")
        print("=" * 60)
        
        # 1. Dataset overview
        print("\\n1. Dataset Overview:")
        for key, value in dataset_info.items():
            print(f"   {key}: {value}")
        
        # 2. Original data statistics
        print("\\n2. Original Data Statistics:")
        numeric_cols = original_data.select_dtypes(include=[np.number]).columns
        print(original_data[numeric_cols].describe())
        
        # 3. Scaling method comparison
        print("\\n3. Scaling Methods Applied:")
        for method in scaled_results.keys():
            print(f"   ✅ {method}")
        
        # 4. Create visualizations
        self.print_scaling_summary()
    
    def plot_windowing_overview(self, window_metadata: List[Dict], title: str = "Time Windowing Overview") -> None:
        """
        Plot overview of time windowing results.
        
        Args:
            window_metadata (List[Dict]): Metadata from windowing process
            title (str): Plot title
        """
        if not window_metadata:
            print("⚠️ No windowing metadata to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Extract data
        classes = [meta['class'] for meta in window_metadata]
        window_sizes = [meta['window_size'] for meta in window_metadata]
        original_lengths = [meta['original_length'] for meta in window_metadata]
        sample_ids = [meta['original_sample_id'] for meta in window_metadata]
        
        # 1. Class distribution
        class_counts = Counter(classes)
        axes[0, 0].bar(class_counts.keys(), class_counts.values(), 
                      color=[plt.cm.Set3(i) for i in range(len(class_counts))])
        axes[0, 0].set_title('Windows per Class', fontweight='bold')
        axes[0, 0].set_xlabel('Class')
        axes[0, 0].set_ylabel('Number of Windows')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Window size distribution
        axes[0, 1].hist(window_sizes, bins=20, color=self.color_palette['primary'], alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Window Size Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Window Size')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Original length distribution
        axes[1, 0].hist(original_lengths, bins=20, color=self.color_palette['secondary'], alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Original Sample Length Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Original Length')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Windows per original sample
        sample_window_counts = Counter(sample_ids)
        axes[1, 1].hist(sample_window_counts.values(), bins=20, color=self.color_palette['accent'], alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Windows per Original Sample', fontweight='bold')
        axes[1, 1].set_xlabel('Number of Windows')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_window_examples(self, original_df: pd.DataFrame, windowed_dfs: List[pd.DataFrame], 
                           window_metadata: List[Dict], num_examples: int = 3,
                           pressure_col: str = 'P-TPT', temp_col: str = 'T-TPT') -> None:
        """
        Plot examples of windowed data compared to original.
        
        Args:
            original_df (pd.DataFrame): Original dataframe
            windowed_dfs (List[pd.DataFrame]): List of windowed dataframes from the original
            window_metadata (List[Dict]): Corresponding metadata
            num_examples (int): Number of window examples to show
            pressure_col (str): Pressure column name
            temp_col (str): Temperature column name
        """
        # Find windows that come from the same original sample
        original_sample_id = window_metadata[0]['original_sample_id'] if window_metadata else 0
        same_sample_windows = [
            (i, df, meta) for i, (df, meta) in enumerate(zip(windowed_dfs, window_metadata))
            if meta['original_sample_id'] == original_sample_id
        ]
        
        num_examples = min(num_examples, len(same_sample_windows))
        
        if num_examples == 0:
            print("⚠️ No windows found for plotting")
            return
        
        # Auto-detect column names (handle scaled versions)
        pressure_cols = [col for col in original_df.columns if pressure_col.replace('_scaled', '') in col]
        temp_cols = [col for col in original_df.columns if temp_col.replace('_scaled', '') in col]
        
        if not pressure_cols:
            pressure_cols = [col for col in original_df.columns if 'P-TPT' in col]
        if not temp_cols:
            temp_cols = [col for col in original_df.columns if 'T-TPT' in col]
            
        actual_pressure_col = pressure_cols[0] if pressure_cols else None
        actual_temp_col = temp_cols[0] if temp_cols else None
        
        if not actual_pressure_col and not actual_temp_col:
            print("⚠️ No pressure or temperature columns found for plotting")
            print(f"Available columns: {list(original_df.columns)}")
            return
        
        # Determine subplot configuration
        has_both = actual_pressure_col is not None and actual_temp_col is not None
        n_cols = 2 if has_both else 1
        
        fig, axes = plt.subplots(num_examples, n_cols, figsize=(15, 4*num_examples))
        if num_examples == 1:
            axes = axes.reshape(1, -1) if has_both else [axes]
        elif not has_both:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'Window Examples from Original Sample {original_sample_id}', fontsize=16, fontweight='bold')
        
        for i in range(num_examples):
            window_idx, windowed_df, metadata = same_sample_windows[i]
            start_idx = metadata['start_idx']
            end_idx = metadata['end_idx']
            
            col_idx = 0
            
            # Plot pressure
            if actual_pressure_col:
                ax = axes[i, col_idx] if has_both else axes[i]
                ax.plot(original_df.index, original_df[actual_pressure_col], 
                       color='lightgray', alpha=0.5, label='Full original')
                ax.plot(range(start_idx, end_idx), original_df[actual_pressure_col].iloc[start_idx:end_idx],
                       color=self.color_palette['pressure'], linewidth=2, label=f'Window {i+1}')
                ax.axvspan(start_idx, end_idx-1, alpha=0.2, color=self.color_palette['pressure'])
                ax.set_title(f'Pressure Window {i+1} (indices {start_idx}-{end_idx})', fontweight='bold')
                ax.set_ylabel('Pressure')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                if i == num_examples - 1:
                    ax.set_xlabel('Time Index')
                
                col_idx += 1
            
            # Plot temperature
            if actual_temp_col:
                ax = axes[i, col_idx] if has_both else axes[i]
                ax.plot(original_df.index, original_df[actual_temp_col], 
                       color='lightgray', alpha=0.5, label='Full original')
                ax.plot(range(start_idx, end_idx), original_df[actual_temp_col].iloc[start_idx:end_idx],
                       color=self.color_palette['temperature'], linewidth=2, label=f'Window {i+1}')
                ax.axvspan(start_idx, end_idx-1, alpha=0.2, color=self.color_palette['temperature'])
                ax.set_title(f'Temperature Window {i+1} (indices {start_idx}-{end_idx})', fontweight='bold')
                ax.set_ylabel('Temperature')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                if i == num_examples - 1:
                    ax.set_xlabel('Time Index')
        
        plt.tight_layout()
        plt.show()
    
    def plot_windowing_comparison(self, train_stats: Dict, test_stats: Dict) -> None:
        """
        Compare windowing statistics between train and test sets.
        
        Args:
            train_stats (Dict): Training set windowing statistics
            test_stats (Dict): Test set windowing statistics
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Train vs Test Windowing Comparison', fontsize=16, fontweight='bold')
        
        # 1. Total windows comparison
        sets = ['Train', 'Test']
        total_windows = [train_stats['total_windows'], test_stats['total_windows']]
        axes[0, 0].bar(sets, total_windows, color=[self.color_palette['primary'], self.color_palette['secondary']])
        axes[0, 0].set_title('Total Windows', fontweight='bold')
        axes[0, 0].set_ylabel('Number of Windows')
        for i, v in enumerate(total_windows):
            axes[0, 0].text(i, v + max(total_windows)*0.01, str(v), ha='center', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Class distribution comparison
        train_classes = list(train_stats['class_distribution'].keys())
        test_classes = list(test_stats['class_distribution'].keys())
        all_classes = sorted(set(train_classes + test_classes))
        
        train_counts = [train_stats['class_distribution'].get(cls, 0) for cls in all_classes]
        test_counts = [test_stats['class_distribution'].get(cls, 0) for cls in all_classes]
        
        x = np.arange(len(all_classes))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, train_counts, width, label='Train', color=self.color_palette['primary'], alpha=0.8)
        axes[0, 1].bar(x + width/2, test_counts, width, label='Test', color=self.color_palette['secondary'], alpha=0.8)
        axes[0, 1].set_title('Class Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Number of Windows')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(all_classes)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Window size distribution
        train_win_stats = train_stats['window_size_stats']
        test_win_stats = test_stats['window_size_stats']
        
        categories = ['Mean', 'Min', 'Max', 'Std']
        train_values = [train_win_stats['mean'], train_win_stats['min'], train_win_stats['max'], train_win_stats['std']]
        test_values = [test_win_stats['mean'], test_win_stats['min'], test_win_stats['max'], test_win_stats['std']]
        
        x = np.arange(len(categories))
        axes[1, 0].bar(x - width/2, train_values, width, label='Train', color=self.color_palette['primary'], alpha=0.8)
        axes[1, 0].bar(x + width/2, test_values, width, label='Test', color=self.color_palette['secondary'], alpha=0.8)
        axes[1, 0].set_title('Window Size Statistics', fontweight='bold')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(categories)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Windows per sample statistics
        train_sample_stats = train_stats['windows_per_sample_stats']
        test_sample_stats = test_stats['windows_per_sample_stats']
        
        sample_values_train = [train_sample_stats['mean'], train_sample_stats['min'], train_sample_stats['max']]
        sample_values_test = [test_sample_stats['mean'], test_sample_stats['min'], test_sample_stats['max']]
        categories = ['Mean', 'Min', 'Max']
        
        x = np.arange(len(categories))
        axes[1, 1].bar(x - width/2, sample_values_train, width, label='Train', color=self.color_palette['primary'], alpha=0.8)
        axes[1, 1].bar(x + width/2, sample_values_test, width, label='Test', color=self.color_palette['secondary'], alpha=0.8)
        axes[1, 1].set_title('Windows per Sample Statistics', fontweight='bold')
        axes[1, 1].set_ylabel('Number of Windows')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(categories)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_windowing_summary(self, windowing_results: Dict) -> None:
        """
        Print a comprehensive summary of windowing results.
        
        Args:
            windowing_results (Dict): Results from windowing process
        """
        print("\\n🪟 Time Windowing Summary")
        print("=" * 60)
        
        params = windowing_results['windowing_parameters']
        train_stats = windowing_results['train_statistics']
        test_stats = windowing_results['test_statistics']
        
        print(f"\\n⚙️  Windowing Parameters:")
        print(f"   Window size: {params['window_size']}")
        print(f"   Stride: {params['stride']}")
        print(f"   Minimum window size: {params['min_window_size']}")
        
        print(f"\\n📊 Results Overview:")
        print(f"   Training windows: {train_stats['total_windows']}")
        print(f"   Test windows: {test_stats['total_windows']}")
        print(f"   Total windows: {train_stats['total_windows'] + test_stats['total_windows']}")
        
        print(f"\\n🎯 Class Distribution (Training):")
        for class_id, count in sorted(train_stats['class_distribution'].items()):
            percentage = (count / train_stats['total_windows']) * 100
            print(f"   Class {class_id}: {count} windows ({percentage:.1f}%)")
        
        print(f"\\n🎯 Class Distribution (Test):")
        for class_id, count in sorted(test_stats['class_distribution'].items()):
            percentage = (count / test_stats['total_windows']) * 100
            print(f"   Class {class_id}: {count} windows ({percentage:.1f}%)")
        
        print(f"\\n📏 Window Statistics:")
        print(f"   Training - Avg size: {train_stats['window_size_stats']['mean']:.1f}")
        print(f"   Test - Avg size: {test_stats['window_size_stats']['mean']:.1f}")
        print(f"   Training - Windows per sample: {train_stats['windows_per_sample_stats']['mean']:.1f}")
        print(f"   Test - Windows per sample: {test_stats['windows_per_sample_stats']['mean']:.1f}")
        
        print(f"\\n💡 Next Steps:")
        print("   1. Use train_windowed_dfs for model training")
        print("   2. Each window is now an independent sample")
        print("   3. Consider sequence-based models (LSTM, CNN)")
        print("   4. Apply feature engineering to windows if needed")
