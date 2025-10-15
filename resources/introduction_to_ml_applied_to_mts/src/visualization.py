"""
Visualization Module for 3W Dataset

This module provides comprehensive visualization utilities for analyzing
sensor data and scaling methods in the 3W dataset.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple


class DataVisualizer:
    """
    A class for creating visualizations of the 3W dataset.

    Provides methods for plotting complete data, scaled data, distributions,
    and comparative analyses.
    """

    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8)):
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
            plt.style.use("default")

        self.default_figsize = figsize
        self.color_palette = {
            "pressure": "#FF6B6B",
            "temperature": "#4ECDC4",
            "primary": "#2E86AB",
            "secondary": "#A23B72",
            "accent": "#F18F01",
        }

    def plot_dataset_overview(
        self, classes: List[str], dataset_stats: Dict[str, Any]
    ) -> None:
        """
        Plot overview of dataset structure and class distribution.

        Args:
            classes (List[str]): List of class labels
            dataset_stats (Dict): Dataset statistics from DataLoader
        """
        print("🔍 Dataset Analysis")
        print("=" * 50)

        # Basic statistics
        print(f"Total number of dataframes loaded: {len(classes)}")
        print(f"Class distribution: {Counter(classes)}")

        # Check if we have memory optimization stats
        memory_optimized = dataset_stats.get("memory_optimization", False)
        
        if memory_optimized:
            # Enhanced visualization for memory-optimized loading
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle("3W Dataset Overview (Memory Optimized)", fontsize=16, fontweight="bold")

            # Plot 1: Loaded files by class
            class_counts_loaded = Counter(classes)
            class_names = sorted(class_counts_loaded.keys())
            loaded_counts = [class_counts_loaded[cls] for cls in class_names]
            
            # Get available counts
            available_counts = [dataset_stats['classes_count'][cls]['total_available'] for cls in class_names]

            x_pos = range(len(class_names))
            width = 0.35

            axes[0, 0].bar([x - width/2 for x in x_pos], available_counts, width, 
                          label='Available', color=self.color_palette["secondary"], alpha=0.7)
            axes[0, 0].bar([x + width/2 for x in x_pos], loaded_counts, width, 
                          label='Loaded', color=self.color_palette["primary"], alpha=0.8)
            
            axes[0, 0].set_title("Files per Class: Available vs Loaded")
            axes[0, 0].set_xlabel("Class")
            axes[0, 0].set_ylabel("Number of Files")
            axes[0, 0].set_xticks(x_pos)
            axes[0, 0].set_xticklabels(class_names)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Add value labels on bars
            for i, (avail, loaded) in enumerate(zip(available_counts, loaded_counts)):
                axes[0, 0].text(i - width/2, avail + 1, str(avail), ha='center', va='bottom', fontsize=9)
                axes[0, 0].text(i + width/2, loaded + 1, str(loaded), ha='center', va='bottom', fontsize=9)

            # Plot 2: Real vs Simulated distribution (loaded)
            real_loaded = dataset_stats["real_files_loaded"]
            simulated_loaded = dataset_stats["simulated_files_loaded"]

            axes[0, 1].pie(
                [real_loaded, simulated_loaded],
                labels=[f"Real Data\n({real_loaded})", f"Simulated Data\n({simulated_loaded})"],
                colors=[self.color_palette["accent"], self.color_palette["secondary"]],
                autopct="%1.1f%%",
                startangle=90
            )
            axes[0, 1].set_title("Real vs Simulated Data (Loaded)")

            # Plot 3: Memory optimization summary
            total_available = dataset_stats["total_files_available"] 
            total_loaded = dataset_stats["total_files_loaded"]
            max_per_class = dataset_stats["max_files_per_class"]
            
            categories = ['Total Available', 'Total Loaded', f'Max Limit\n({max_per_class} × {len(class_names)})']
            values = [total_available, total_loaded, max_per_class * len(class_names)]
            colors = [self.color_palette["secondary"], self.color_palette["primary"], self.color_palette["accent"]]
            
            bars = axes[1, 0].bar(categories, values, color=colors, alpha=0.7)
            axes[1, 0].set_title("Memory Optimization Summary")
            axes[1, 0].set_ylabel("Number of Files")
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                               str(value), ha='center', va='bottom', fontweight='bold')

            # Plot 4: Real vs Simulated comparison (available vs loaded)
            categories = ['Real Files', 'Simulated Files']
            available = [dataset_stats["real_files_available"], dataset_stats["simulated_files_available"]]
            loaded = [real_loaded, simulated_loaded]

            x_pos = range(len(categories))
            width = 0.35

            axes[1, 1].bar([x - width/2 for x in x_pos], available, width, 
                          label='Available', color=self.color_palette["secondary"], alpha=0.7)
            axes[1, 1].bar([x + width/2 for x in x_pos], loaded, width, 
                          label='Loaded', color=self.color_palette["primary"], alpha=0.8)
            
            axes[1, 1].set_title("Real vs Simulated: Available vs Loaded")
            axes[1, 1].set_ylabel("Number of Files")
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(categories)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            # Add value labels
            for i, (avail, load) in enumerate(zip(available, loaded)):
                axes[1, 1].text(i - width/2, avail + max(available)*0.01, str(avail), ha='center', va='bottom', fontsize=10)
                axes[1, 1].text(i + width/2, load + max(available)*0.01, str(load), ha='center', va='bottom', fontsize=10)

        else:
            # Original visualization for backward compatibility
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle("3W Dataset Overview", fontsize=16, fontweight="bold")

            # Plot 1: Class distribution
            class_counts = Counter(classes)
            class_names = list(class_counts.keys())
            counts = list(class_counts.values())

            axes[0].bar(class_names, counts, color=self.color_palette["primary"], alpha=0.7)
            axes[0].set_title("Distribution of Files by Class")
            axes[0].set_xlabel("Class")
            axes[0].set_ylabel("Number of Files")
            axes[0].tick_params(axis="x", rotation=45)
            axes[0].grid(True, alpha=0.3)

            # Plot 2: Real vs Simulated distribution
            if "real_files" in dataset_stats and "simulated_files" in dataset_stats:
                real_count = dataset_stats["real_files"]
                simulated_count = dataset_stats["simulated_files"]

                axes[1].pie(
                    [real_count, simulated_count],
                    labels=["Real Data", "Simulated Data"],
                    colors=[self.color_palette["accent"], self.color_palette["secondary"]],
                    autopct="%1.1f%%",
                )
                axes[1].set_title("Real vs Simulated Data Distribution")

        plt.tight_layout()
        plt.show()

        # Print memory usage summary
        if memory_optimized:
            print(f"\n💾 Memory Optimization Results:")
            print(f"   • Files available: {dataset_stats['total_files_available']:,}")
            print(f"   • Files loaded: {dataset_stats['total_files_loaded']:,}")
            print(f"   • Memory reduction: {((dataset_stats['total_files_available'] - dataset_stats['total_files_loaded']) / dataset_stats['total_files_available'] * 100):.1f}%")
            print(f"   • Real data priority: {dataset_stats['real_files_loaded']}/{dataset_stats['real_files_available']} real files loaded")
        elif "total_samples" in dataset_stats:
            print(f"\n💾 Total samples: {dataset_stats['total_samples']:,}")

    def plot_raw_data_analysis(
        self,
        data: pd.DataFrame,
        pressure_col: str = "P-TPT",
        temp_col: str = "T-TPT",
        title_suffix: str = "",
    ) -> None:
        """
        Plot comprehensive analysis of raw sensor data.

        Args:
            data (pd.DataFrame): Complete sensor data
            pressure_col (str): Name of pressure column
            temp_col (str): Name of temperature column
            title_suffix (str): Additional text for plot title
        """

        # Check if required columns exist
        if pressure_col not in data.columns or temp_col not in data.columns:
            print(
                f"❌ Required columns {pressure_col} and/or {temp_col} not found in data"
            )
            return

        # Clean data
        clean_data = data[[pressure_col, temp_col]].dropna()

        if len(clean_data) == 0:
            print("❌ No valid data found after removing NaN values")
            return

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f"Raw Data Analysis - Pressure and Temperature{title_suffix}",
            fontsize=16,
            fontweight="bold",
        )

        # Time series plots
        axes[0, 0].plot(
            clean_data[pressure_col],
            color=self.color_palette["pressure"],
            alpha=0.7,
            linewidth=0.5,
        )
        axes[0, 0].set_title(f"Raw Pressure ({pressure_col}) Time Series")
        axes[0, 0].set_xlabel("Sample Index")
        axes[0, 0].set_ylabel("Pressure Value")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(
            clean_data[temp_col],
            color=self.color_palette["temperature"],
            alpha=0.7,
            linewidth=0.5,
        )
        axes[0, 1].set_title(f"Raw Temperature ({temp_col}) Time Series")
        axes[0, 1].set_xlabel("Sample Index")
        axes[0, 1].set_ylabel("Temperature Value")
        axes[0, 1].grid(True, alpha=0.3)

        # Distribution plots
        axes[1, 0].hist(
            clean_data[pressure_col],
            bins=50,
            color=self.color_palette["pressure"],
            alpha=0.7,
            density=True,
        )
        axes[1, 0].set_title(f"Raw Pressure Distribution")
        axes[1, 0].set_xlabel("Pressure Value")
        axes[1, 0].set_ylabel("Density")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].hist(
            clean_data[temp_col],
            bins=50,
            color=self.color_palette["temperature"],
            alpha=0.7,
            density=True,
        )
        axes[1, 1].set_title(f"Raw Temperature Distribution")
        axes[1, 1].set_xlabel("Temperature Value")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print statistics
        print(f"\\n📈 Data Statistics:")
        print(f"Shape: {clean_data.shape}")
        print(clean_data.describe())

    def plot_scaled_data_analysis(
        self,
        scaled_data: pd.DataFrame,
        scaler_name: str,
        pressure_col: str = None,
        temp_col: str = None,
    ) -> None:
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
            numeric_cols = [col for col in scaled_data.columns if col != "class"]
            if len(numeric_cols) >= 2:
                pressure_col = pressure_col or numeric_cols[0]
                temp_col = temp_col or numeric_cols[1]
            else:
                print(f"❌ Insufficient numeric columns for plotting")
                return

        # Check if columns exist
        if (
            pressure_col not in scaled_data.columns
            or temp_col not in scaled_data.columns
        ):
            print(f"❌ Required columns {pressure_col} and/or {temp_col} not found")
            return

        # Clean data
        clean_data = scaled_data[[pressure_col, temp_col]].dropna()

        if len(clean_data) == 0:
            print("❌ No valid data found after removing NaN values")
            return

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f"{scaler_name} - Scaled Data Analysis", fontsize=16, fontweight="bold"
        )

        # Time series plots
        axes[0, 0].plot(
            clean_data[pressure_col],
            color=self.color_palette["pressure"],
            alpha=0.7,
            linewidth=0.5,
        )
        axes[0, 0].set_title(f"Scaled Pressure ({scaler_name})")
        axes[0, 0].set_xlabel("Sample Index")
        axes[0, 0].set_ylabel("Scaled Pressure Value")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(
            clean_data[temp_col],
            color=self.color_palette["temperature"],
            alpha=0.7,
            linewidth=0.5,
        )
        axes[0, 1].set_title(f"Scaled Temperature ({scaler_name})")
        axes[0, 1].set_xlabel("Sample Index")
        axes[0, 1].set_ylabel("Scaled Temperature Value")
        axes[0, 1].grid(True, alpha=0.3)

        # Distribution plots
        axes[1, 0].hist(
            clean_data[pressure_col],
            bins=50,
            color=self.color_palette["pressure"],
            alpha=0.7,
            density=True,
        )
        axes[1, 0].set_title(f"Scaled Pressure Distribution ({scaler_name})")
        axes[1, 0].set_xlabel("Scaled Pressure Value")
        axes[1, 0].set_ylabel("Density")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].hist(
            clean_data[temp_col],
            bins=50,
            color=self.color_palette["temperature"],
            alpha=0.7,
            density=True,
        )
        axes[1, 1].set_title(f"Scaled Temperature Distribution ({scaler_name})")
        axes[1, 1].set_xlabel("Scaled Temperature Value")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print statistics
        print(f"Statistics for {scaler_name}:")
        print(
            f"Pressure - Mean: {clean_data[pressure_col].mean():.4f}, Std: {clean_data[pressure_col].std():.4f}"
        )
        print(
            f"Temperature - Mean: {clean_data[temp_col].mean():.4f}, Std: {clean_data[temp_col].std():.4f}"
        )

    def plot_scaling_comparison(
        self,
        scaling_results: Dict[str, pd.DataFrame],
        pressure_col: str = "P-TPT",
        temp_col: str = "T-TPT",
    ) -> None:
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
        fig.suptitle(
            "Scaling Methods Comparison - Boxplots", fontsize=16, fontweight="bold"
        )

        # Prepare data for boxplots
        method_names = list(scaling_results.keys())

        # Pressure comparison
        pressure_data = []
        temp_data = []
        valid_methods = []

        for method, data in scaling_results.items():
            # Handle different column naming conventions
            p_col = (
                pressure_col
                if pressure_col in data.columns
                else f"{pressure_col}_scaled"
            )
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
        for patch in bp1["boxes"]:
            patch.set_facecolor(self.color_palette["pressure"])
            patch.set_alpha(0.7)
        axes[0].set_title("Pressure - All Scaling Methods")
        axes[0].set_ylabel("Scaled Values")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].grid(True, alpha=0.3)

        # Temperature boxplot
        bp2 = axes[1].boxplot(temp_data, labels=valid_methods, patch_artist=True)
        for patch in bp2["boxes"]:
            patch.set_facecolor(self.color_palette["temperature"])
            patch.set_alpha(0.7)
        axes[1].set_title("Temperature - All Scaling Methods")
        axes[1].set_ylabel("Scaled Values")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(
        self, data: pd.DataFrame, title: str = "Feature Correlation Matrix"
    ) -> None:
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
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
        )
        plt.title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def plot_time_series_comparison(
        self, data_dict: Dict[str, pd.DataFrame], column: str, max_samples: int = 1000
    ) -> None:
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
                    indices = np.linspace(0, len(plot_data) - 1, max_samples, dtype=int)
                    plot_data = plot_data.iloc[indices]

                plt.plot(
                    plot_data, label=method, color=colors[i], alpha=0.7, linewidth=1
                )

        plt.title(f"Time Series Comparison - {column}", fontsize=14, fontweight="bold")
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
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
        print(
            "- StandardScaler: Normal data, algorithms that assume normality (SVM, Neural Networks)"
        )
        print(
            "- MinMaxScaler: Need bounded values, preserve zero, distance-based algorithms"
        )
        print("- RobustScaler: Data with outliers, non-normal distributions")
        print(
            "- Normalizer: When direction matters more than magnitude (text analysis, clustering)"
        )

    def create_summary_report(
        self,
        original_data: pd.DataFrame,
        scaled_results: Dict[str, pd.DataFrame],
        dataset_info: Dict[str, Any],
    ) -> None:
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

    def plot_windowing_overview(
        self, window_metadata: List[Dict], title: str = "Time Windowing Overview"
    ) -> None:
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
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Extract data
        classes = [meta["class"] for meta in window_metadata]
        window_sizes = [meta["window_size"] for meta in window_metadata]
        original_lengths = [meta["original_length"] for meta in window_metadata]
        sample_ids = [meta["original_sample_id"] for meta in window_metadata]

        # 1. Class distribution
        class_counts = Counter(classes)
        axes[0, 0].bar(
            class_counts.keys(),
            class_counts.values(),
            color=[plt.cm.Set3(i) for i in range(len(class_counts))],
        )
        axes[0, 0].set_title("Windows per Class", fontweight="bold")
        axes[0, 0].set_xlabel("Class")
        axes[0, 0].set_ylabel("Number of Windows")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Window size distribution
        axes[0, 1].hist(
            window_sizes,
            bins=20,
            color=self.color_palette["primary"],
            alpha=0.7,
            edgecolor="black",
        )
        axes[0, 1].set_title("Window Size Distribution", fontweight="bold")
        axes[0, 1].set_xlabel("Window Size")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Original length distribution
        axes[1, 0].hist(
            original_lengths,
            bins=20,
            color=self.color_palette["secondary"],
            alpha=0.7,
            edgecolor="black",
        )
        axes[1, 0].set_title("Original Sample Length Distribution", fontweight="bold")
        axes[1, 0].set_xlabel("Original Length")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Windows per original sample
        sample_window_counts = Counter(sample_ids)
        axes[1, 1].hist(
            sample_window_counts.values(),
            bins=20,
            color=self.color_palette["accent"],
            alpha=0.7,
            edgecolor="black",
        )
        axes[1, 1].set_title("Windows per Original Sample", fontweight="bold")
        axes[1, 1].set_xlabel("Number of Windows")
        axes[1, 1].set_ylabel("Number of Samples")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_window_examples(
        self,
        original_df: pd.DataFrame,
        windowed_dfs: List[pd.DataFrame],
        window_metadata: List[Dict],
        num_examples: int = 3,
        pressure_col: str = "P-TPT",
        temp_col: str = "T-TPT",
    ) -> None:
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
        original_sample_id = (
            window_metadata[0]["original_sample_id"] if window_metadata else 0
        )
        same_sample_windows = [
            (i, df, meta)
            for i, (df, meta) in enumerate(zip(windowed_dfs, window_metadata))
            if meta["original_sample_id"] == original_sample_id
        ]

        num_examples = min(num_examples, len(same_sample_windows))

        if num_examples == 0:
            print("⚠️ No windows found for plotting")
            return

        # Auto-detect column names (handle scaled versions)
        pressure_cols = [
            col
            for col in original_df.columns
            if pressure_col.replace("_scaled", "") in col
        ]
        temp_cols = [
            col for col in original_df.columns if temp_col.replace("_scaled", "") in col
        ]

        if not pressure_cols:
            pressure_cols = [col for col in original_df.columns if "P-TPT" in col]
        if not temp_cols:
            temp_cols = [col for col in original_df.columns if "T-TPT" in col]

        actual_pressure_col = pressure_cols[0] if pressure_cols else None
        actual_temp_col = temp_cols[0] if temp_cols else None

        if not actual_pressure_col and not actual_temp_col:
            print("⚠️ No pressure or temperature columns found for plotting")
            print(f"Available columns: {list(original_df.columns)}")
            return

        # Determine subplot configuration
        has_both = actual_pressure_col is not None and actual_temp_col is not None
        n_cols = 2 if has_both else 1

        fig, axes = plt.subplots(num_examples, n_cols, figsize=(15, 4 * num_examples))
        if num_examples == 1:
            axes = axes.reshape(1, -1) if has_both else [axes]
        elif not has_both:
            axes = axes.reshape(-1, 1)

        fig.suptitle(
            f"Window Examples from Original Sample {original_sample_id}",
            fontsize=16,
            fontweight="bold",
        )

        for i in range(num_examples):
            window_idx, windowed_df, metadata = same_sample_windows[i]
            start_idx = metadata["start_idx"]
            end_idx = metadata["end_idx"]

            col_idx = 0

            # Plot pressure
            if actual_pressure_col:
                ax = axes[i, col_idx] if has_both else axes[i]
                ax.plot(
                    original_df.index,
                    original_df[actual_pressure_col],
                    color="lightgray",
                    alpha=0.5,
                    label="Full original",
                )
                ax.plot(
                    range(start_idx, end_idx),
                    original_df[actual_pressure_col].iloc[start_idx:end_idx],
                    color=self.color_palette["pressure"],
                    linewidth=2,
                    label=f"Window {i+1}",
                )
                ax.axvspan(
                    start_idx,
                    end_idx - 1,
                    alpha=0.2,
                    color=self.color_palette["pressure"],
                )
                ax.set_title(
                    f"Pressure Window {i+1} (indices {start_idx}-{end_idx})",
                    fontweight="bold",
                )
                ax.set_ylabel("Pressure")
                ax.legend()
                ax.grid(True, alpha=0.3)

                if i == num_examples - 1:
                    ax.set_xlabel("Time Index")

                col_idx += 1

            # Plot temperature
            if actual_temp_col:
                ax = axes[i, col_idx] if has_both else axes[i]
                ax.plot(
                    original_df.index,
                    original_df[actual_temp_col],
                    color="lightgray",
                    alpha=0.5,
                    label="Full original",
                )
                ax.plot(
                    range(start_idx, end_idx),
                    original_df[actual_temp_col].iloc[start_idx:end_idx],
                    color=self.color_palette["temperature"],
                    linewidth=2,
                    label=f"Window {i+1}",
                )
                ax.axvspan(
                    start_idx,
                    end_idx - 1,
                    alpha=0.2,
                    color=self.color_palette["temperature"],
                )
                ax.set_title(
                    f"Temperature Window {i+1} (indices {start_idx}-{end_idx})",
                    fontweight="bold",
                )
                ax.set_ylabel("Temperature")
                ax.legend()
                ax.grid(True, alpha=0.3)

                if i == num_examples - 1:
                    ax.set_xlabel("Time Index")

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
        fig.suptitle(
            "Train vs Test Windowing Comparison", fontsize=16, fontweight="bold"
        )

        # 1. Total windows comparison
        sets = ["Train", "Test"]
        total_windows = [train_stats["total_windows"], test_stats["total_windows"]]
        axes[0, 0].bar(
            sets,
            total_windows,
            color=[self.color_palette["primary"], self.color_palette["secondary"]],
        )
        axes[0, 0].set_title("Total Windows", fontweight="bold")
        axes[0, 0].set_ylabel("Number of Windows")
        for i, v in enumerate(total_windows):
            axes[0, 0].text(
                i, v + max(total_windows) * 0.01, str(v), ha="center", fontweight="bold"
            )
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Class distribution comparison
        train_classes = list(train_stats["class_distribution"].keys())
        test_classes = list(test_stats["class_distribution"].keys())
        all_classes = sorted(set(train_classes + test_classes))

        train_counts = [
            train_stats["class_distribution"].get(cls, 0) for cls in all_classes
        ]
        test_counts = [
            test_stats["class_distribution"].get(cls, 0) for cls in all_classes
        ]

        x = np.arange(len(all_classes))
        width = 0.35

        axes[0, 1].bar(
            x - width / 2,
            train_counts,
            width,
            label="Train",
            color=self.color_palette["primary"],
            alpha=0.8,
        )
        axes[0, 1].bar(
            x + width / 2,
            test_counts,
            width,
            label="Test",
            color=self.color_palette["secondary"],
            alpha=0.8,
        )
        axes[0, 1].set_title("Class Distribution", fontweight="bold")
        axes[0, 1].set_xlabel("Class")
        axes[0, 1].set_ylabel("Number of Windows")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(all_classes)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Window size distribution
        train_win_stats = train_stats["window_size_stats"]
        test_win_stats = test_stats["window_size_stats"]

        categories = ["Mean", "Min", "Max", "Std"]
        train_values = [
            train_win_stats["mean"],
            train_win_stats["min"],
            train_win_stats["max"],
            train_win_stats["std"],
        ]
        test_values = [
            test_win_stats["mean"],
            test_win_stats["min"],
            test_win_stats["max"],
            test_win_stats["std"],
        ]

        x = np.arange(len(categories))
        axes[1, 0].bar(
            x - width / 2,
            train_values,
            width,
            label="Train",
            color=self.color_palette["primary"],
            alpha=0.8,
        )
        axes[1, 0].bar(
            x + width / 2,
            test_values,
            width,
            label="Test",
            color=self.color_palette["secondary"],
            alpha=0.8,
        )
        axes[1, 0].set_title("Window Size Statistics", fontweight="bold")
        axes[1, 0].set_ylabel("Value")
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(categories)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Windows per sample statistics
        train_sample_stats = train_stats["windows_per_sample_stats"]
        test_sample_stats = test_stats["windows_per_sample_stats"]

        sample_values_train = [
            train_sample_stats["mean"],
            train_sample_stats["min"],
            train_sample_stats["max"],
        ]
        sample_values_test = [
            test_sample_stats["mean"],
            test_sample_stats["min"],
            test_sample_stats["max"],
        ]
        categories = ["Mean", "Min", "Max"]

        x = np.arange(len(categories))
        axes[1, 1].bar(
            x - width / 2,
            sample_values_train,
            width,
            label="Train",
            color=self.color_palette["primary"],
            alpha=0.8,
        )
        axes[1, 1].bar(
            x + width / 2,
            sample_values_test,
            width,
            label="Test",
            color=self.color_palette["secondary"],
            alpha=0.8,
        )
        axes[1, 1].set_title("Windows per Sample Statistics", fontweight="bold")
        axes[1, 1].set_ylabel("Number of Windows")
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

        params = windowing_results["windowing_parameters"]
        train_stats = windowing_results["train_statistics"]
        test_stats = windowing_results["test_statistics"]

        print(f"\\n⚙️  Windowing Parameters:")
        print(f"   Window size: {params['window_size']}")
        print(f"   Stride: {params['stride']}")
        print(f"   Minimum window size: {params['min_window_size']}")

        print(f"\\n📊 Results Overview:")
        print(f"   Training windows: {train_stats['total_windows']}")
        print(f"   Test windows: {test_stats['total_windows']}")
        print(
            f"   Total windows: {train_stats['total_windows'] + test_stats['total_windows']}"
        )

        print(f"\\n🎯 Class Distribution (Training):")
        for class_id, count in sorted(train_stats["class_distribution"].items()):
            percentage = (count / train_stats["total_windows"]) * 100
            print(f"   Class {class_id}: {count} windows ({percentage:.1f}%)")

        print(f"\\n🎯 Class Distribution (Test):")
        for class_id, count in sorted(test_stats["class_distribution"].items()):
            percentage = (count / test_stats["total_windows"]) * 100
            print(f"   Class {class_id}: {count} windows ({percentage:.1f}%)")

        print(f"\\n📏 Window Statistics:")
        print(f"   Training - Avg size: {train_stats['window_size_stats']['mean']:.1f}")
        print(f"   Test - Avg size: {test_stats['window_size_stats']['mean']:.1f}")
        print(
            f"   Training - Windows per sample: {train_stats['windows_per_sample_stats']['mean']:.1f}"
        )
        print(
            f"   Test - Windows per sample: {test_stats['windows_per_sample_stats']['mean']:.1f}"
        )

        print(f"\\n💡 Next Steps:")
        print("   1. Use train_windowed_dfs for model training")
        print("   2. Each window is now an independent sample")
        print("   3. Consider sequence-based models (LSTM, CNN)")
        print("   4. Apply feature engineering to windows if needed")


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

        # Get target classes and max samples
        if (
            hasattr(config, "CLASSIFICATION_CONFIG")
            and "selected_classes" in config.CLASSIFICATION_CONFIG
        ):
            target_classes = config.CLASSIFICATION_CONFIG["selected_classes"]
        else:
            target_classes = list(range(1, 10))

        max_samples_per_class = getattr(config, "VISUALIZATION_MAX_SAMPLES", 50)

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
            print(f"⚠️ Shape mismatch, fixing...")
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
        import time

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
                f"   • t-SNE #{i+1}: perplexity={config['perplexity']}, lr={config['learning_rate']}..."
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
            print("❌ UMAP not available. Install with: pip install umap-learn")
            return None

        import time

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
                f"   • UMAP #{i+1}: neighbors={config['n_neighbors']}, dist={config['min_dist']}..."
            )

            try:
                start_time = time.time()

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
                        f"⚡ {result['time']:.1f}s",
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

            print(f"⚡ Configuration Performance:")
            for i, result in enumerate(results):
                if result["embedding"] is not None:
                    status = (
                        " ⚡ Fastest"
                        if result["time"] == fastest_config["time"]
                        else ""
                    )
                    print(f"   • Config {i+1}: {result['time']:.1f}s{status}")
                else:
                    print(f"   • Config {i+1}: Failed")
        else:
            print("   ❌ All configurations failed")

    def visualize_window_samples(self, test_dfs, mapped_classes, samples_per_class=5):
        """
        Visualize window sensor data for samples of each class.

        Args:
            test_dfs: List of dataframes containing windowed time series data
            mapped_classes: List of class labels for each window
            samples_per_class: Number of samples to show per class
        """
        print("📊 Creating window Sensor Data Visualizations")
        print("=" * 60)

        # Get unique classes
        unique_classes = sorted(set(mapped_classes))
        print(
            f"📈 Visualizing {samples_per_class} samples for each of {len(unique_classes)} classes"
        )

        # Define color palette for sensors
        sensor_colors = {
            "P-PDG": "#FF6B6B",  # Red
            "P-TPT": "#4ECDC4",  # Teal
            "T-TPT": "#45B7D1",  # Blue
            "P-MON-CKP": "#96CEB4",  # Green
            "T-JUS-CKP": "#FFEAA7",  # Yellow
            "P-JUS-CKGL": "#DDA0DD",  # Plum
            "QGL": "#FFB347",  # Orange
        }

        # Create figures for each class
        for class_label in unique_classes:
            print(f"\n🎯 Class {class_label}:")

            # Get indices for this class
            class_indices = [
                i for i, cls in enumerate(mapped_classes) if cls == class_label
            ]

            if len(class_indices) == 0:
                print(f"   ⚠️ No samples found for class {class_label}")
                continue

            # Sample random windows for this class
            n_samples = min(samples_per_class, len(class_indices))
            sampled_indices = np.random.choice(
                class_indices, size=n_samples, replace=False
            )

            print(
                f"   📊 Showing {n_samples} samples (out of {len(class_indices)} available)"
            )

            # Create subplot figure for this class
            fig, axes = plt.subplots(n_samples, 1, figsize=(15, 3 * n_samples))
            if n_samples == 1:
                axes = [axes]

            for idx, window_idx in enumerate(sampled_indices):
                window_df = test_dfs[window_idx]
                ax = axes[idx]

                # Get sensor columns (exclude class column)
                sensor_columns = [col for col in window_df.columns if col != "class"]

                # Plot each sensor with different colors
                for sensor in sensor_columns:
                    color = sensor_colors.get(
                        sensor,
                        np.random.choice(
                            [
                                "#FF6B6B",
                                "#4ECDC4",
                                "#45B7D1",
                                "#96CEB4",
                                "#FFEAA7",
                                "#DDA0DD",
                                "#FFB347",
                            ]
                        ),
                    )
                    ax.plot(
                        window_df.index,
                        window_df[sensor],
                        label=sensor,
                        color=color,
                        linewidth=1.5,
                        alpha=0.8,
                    )

                # Customize subplot
                ax.set_title(
                    f"Class {class_label} - Sample {idx+1}",
                    fontsize=12,
                    fontweight="bold",
                )
                ax.set_xlabel("Time Step")
                ax.set_ylabel("Sensor Value")
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                ax.grid(True, alpha=0.3)

                # Add sample info
                ax.text(
                    0.02,
                    0.98,
                    f"Window {window_idx}",
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

            plt.suptitle(
                f"Windowed Sensor Data - Class {class_label}",
                fontsize=16,
                fontweight="bold",
                y=0.98,
            )
            plt.tight_layout()
            plt.show()

            print(f"   ✅ Displayed {n_samples} samples for class {class_label}")


class AnomalyDetectionVisualizer:
    """
    Visualizer for anomaly detection results using autoencoders.
    """
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        self.colors = {
            'normal': '#1f77b4',  # blue
            'anomaly': '#d62728',  # red
            'threshold': '#2ca02c',  # green
            'threshold_alt': '#ff7f0e',  # orange
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
        avg_normal_acc = np.mean([r['normal_accuracy'] for r in valid_results])
        avg_anomaly_acc = np.mean([r['anomaly_accuracy'] for r in valid_results])
        avg_overall_acc = np.mean([r['overall_accuracy'] for r in valid_results])
        
        std_normal_acc = np.std([r['normal_accuracy'] for r in valid_results])
        std_anomaly_acc = np.std([r['anomaly_accuracy'] for r in valid_results])
        std_overall_acc = np.std([r['overall_accuracy'] for r in valid_results])
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Per-fold performance
        fold_names = list(fold_results.keys())
        normal_accs = [fold_results[f]['normal_accuracy'] if fold_results[f] else 0 for f in fold_names]
        anomaly_accs = [fold_results[f]['anomaly_accuracy'] if fold_results[f] else 0 for f in fold_names]
        
        x = np.arange(len(fold_names))
        width = 0.35
        
        ax1.bar(x - width/2, normal_accs, width, label='Normal', alpha=0.7, color=self.colors['normal'])
        ax1.bar(x + width/2, anomaly_accs, width, label='Anomaly', alpha=0.7, color=self.colors['anomaly'])
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Per-Fold Detection Accuracy')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f.replace('fold_', 'F') for f in fold_names])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Overall statistics
        categories = ['Normal', 'Anomaly', 'Overall']
        means = [avg_normal_acc, avg_anomaly_acc, avg_overall_acc]
        stds = [std_normal_acc, std_anomaly_acc, std_overall_acc]
        
        ax2.bar(categories, means, yerr=stds, capsize=5, alpha=0.7, 
                color=[self.colors['normal'], self.colors['anomaly'], 'green'])
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Average Performance ± Std')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_reconstruction_errors(self, normal_errors, anomaly_errors, anomaly_classes=None, 
                                 threshold_percentile=95, threshold_std_multiplier=2):
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
        threshold_std = np.mean(normal_errors) + threshold_std_multiplier * np.std(normal_errors)
        
        # Create figure
        if anomaly_classes is not None and len(set(anomaly_classes)) > 1:
            fig, axes = plt.subplots(2, 2, figsize=self.figsize)
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes = [axes[0], axes[1]]
        
        # Plot 1: Overall distribution
        ax = axes[0]
        ax.hist(normal_errors, bins=50, alpha=0.7, label='Normal', color=self.colors['normal'], density=True)
        ax.hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly', color=self.colors['anomaly'], density=True)
        ax.axvline(threshold_pct, color=self.colors['threshold'], linestyle='--', 
                  label=f'{threshold_percentile}th percentile')
        ax.axvline(threshold_std, color=self.colors['threshold_alt'], linestyle='--', 
                  label=f'μ + {threshold_std_multiplier}σ')
        ax.set_xlabel('Reconstruction Error')
        ax.set_ylabel('Density')
        ax.set_title('Reconstruction Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Box plot comparison
        ax = axes[1]
        data_to_plot = [normal_errors, anomaly_errors]
        labels = ['Normal', 'Anomaly']
        colors = [self.colors['normal'], self.colors['anomaly']]
        
        box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.axhline(threshold_pct, color=self.colors['threshold'], linestyle='--', 
                  label=f'{threshold_percentile}th percentile')
        ax.axhline(threshold_std, color=self.colors['threshold_alt'], linestyle='--', 
                  label=f'μ + {threshold_std_multiplier}σ')
        ax.set_ylabel('Reconstruction Error')
        ax.set_title('Error Distribution by Data Type')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Per-class distributions (if applicable)
        if anomaly_classes is not None and len(set(anomaly_classes)) > 1 and len(axes) > 2:
            ax = axes[2]
            unique_classes = sorted(set(anomaly_classes))
            
            # Normal data
            ax.hist(normal_errors, bins=30, alpha=0.5, label='Normal', 
                   color=self.colors['normal'], density=True)
            
            # Each anomaly class
            class_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_classes)))
            for i, cls in enumerate(unique_classes):
                if cls != 0:  # Skip normal class
                    cls_indices = [j for j, c in enumerate(anomaly_classes) if c == cls]
                    cls_errors = anomaly_errors[cls_indices]
                    ax.hist(cls_errors, bins=30, alpha=0.7, label=f'Class {cls}', 
                           color=class_colors[i], density=True)
            
            ax.axvline(threshold_pct, color=self.colors['threshold'], linestyle='--', 
                      label=f'{threshold_percentile}th percentile')
            ax.axvline(threshold_std, color=self.colors['threshold_alt'], linestyle='--', 
                      label=f'μ + {threshold_std_multiplier}σ')
            ax.set_xlabel('Reconstruction Error')
            ax.set_ylabel('Density')
            ax.set_title('Per-Class Error Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Class performance summary
            if len(axes) > 3:
                ax = axes[3]
                class_stats = []
                class_names = []
                
                for cls in unique_classes:
                    if cls != 0:
                        cls_indices = [j for j, c in enumerate(anomaly_classes) if c == cls]
                        cls_errors = anomaly_errors[cls_indices]
                        detection_rate_pct = np.mean(cls_errors > threshold_pct)
                        detection_rate_std = np.mean(cls_errors > threshold_std)
                        class_stats.append([detection_rate_pct, detection_rate_std])
                        class_names.append(f'Class {cls}')
                
                class_stats = np.array(class_stats)
                x = np.arange(len(class_names))
                width = 0.35
                
                ax.bar(x - width/2, class_stats[:, 0], width, 
                      label=f'{threshold_percentile}th percentile', alpha=0.7, 
                      color=self.colors['threshold'])
                ax.bar(x + width/2, class_stats[:, 1], width, 
                      label=f'μ + {threshold_std_multiplier}σ', alpha=0.7, 
                      color=self.colors['threshold_alt'])
                
                ax.set_xlabel('Anomaly Class')
                ax.set_ylabel('Detection Rate')
                ax.set_title('Detection Rate by Class and Threshold')
                ax.set_xticks(x)
                ax.set_xticklabels(class_names)
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print threshold information
        print(f"Threshold Analysis:")
        print(f"  • {threshold_percentile}th percentile: {threshold_pct:.6f}")
        print(f"  • μ + {threshold_std_multiplier}σ: {threshold_std:.6f}")
        
    def plot_latent_space_tsne(self, autoencoder, normal_data, anomaly_data, anomaly_classes=None, 
                              max_samples=1000, perplexity=30, random_state=42):
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
            normal_indices = np.random.choice(len(normal_data), max_samples // 2, replace=False)
            normal_sample = normal_data[normal_indices]
        else:
            normal_sample = normal_data
            
        if len(anomaly_data) > max_samples // 2:
            anomaly_indices = np.random.choice(len(anomaly_data), max_samples // 2, replace=False)
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
        labels = ['Normal'] * len(normal_latent) + ['Anomaly'] * len(anomaly_latent)
        
        # Compute t-SNE
        print(f"Computing t-SNE embedding for {len(all_latent)} samples...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        latent_2d = tsne.fit_transform(all_latent)
        
        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Normal vs Anomaly
        ax = axes[0]
        normal_mask = np.array(labels) == 'Normal'
        anomaly_mask = np.array(labels) == 'Anomaly'
        
        ax.scatter(latent_2d[normal_mask, 0], latent_2d[normal_mask, 1], 
                  c=self.colors['normal'], alpha=0.6, label='Normal', s=20)
        ax.scatter(latent_2d[anomaly_mask, 0], latent_2d[anomaly_mask, 1], 
                  c=self.colors['anomaly'], alpha=0.6, label='Anomaly', s=20)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title('Latent Space: Normal vs Anomaly')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: By anomaly class
        ax = axes[1]
        ax.scatter(latent_2d[normal_mask, 0], latent_2d[normal_mask, 1], 
                  c=self.colors['normal'], alpha=0.6, label='Normal', s=20)
        
        if anomaly_classes_sample is not None:
            unique_classes = sorted(set(anomaly_classes_sample))
            class_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_classes)))
            
            for i, cls in enumerate(unique_classes):
                if cls != 0:  # Skip normal class
                    cls_mask = np.array(anomaly_classes_sample) == cls
                    anomaly_start_idx = len(normal_latent)
                    global_mask = np.zeros(len(latent_2d), dtype=bool)
                    global_mask[anomaly_start_idx:][cls_mask] = True
                    
                    ax.scatter(latent_2d[global_mask, 0], latent_2d[global_mask, 1], 
                              c=[class_colors[i]], alpha=0.7, label=f'Class {cls}', s=20)
        else:
            ax.scatter(latent_2d[anomaly_mask, 0], latent_2d[anomaly_mask, 1], 
                      c=self.colors['anomaly'], alpha=0.6, label='Anomaly', s=20)
        
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title('Latent Space: By Anomaly Class')
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
        y_true = np.concatenate([np.zeros(len(normal_errors)), np.ones(len(anomaly_errors))])
        y_scores = np.concatenate([normal_errors, anomaly_errors])
        
        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Compute Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        # Create comprehensive ROC analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: ROC Curve
        ax = axes[0, 0]
        ax.plot(fpr, tpr, color=self.colors['anomaly'], lw=2, 
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.8, 
                label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (1 - Specificity)')
        ax.set_ylabel('True Positive Rate (Sensitivity)')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        # Add optimal threshold point
        # Find threshold that maximizes TPR - FPR (Youden's index)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        optimal_tpr = tpr[optimal_idx]
        
        ax.plot(optimal_fpr, optimal_tpr, 'ro', markersize=8, 
                label=f'Optimal Point (threshold={optimal_threshold:.4f})')
        ax.legend(loc="lower right")
        
        # Plot 2: Precision-Recall Curve
        ax = axes[0, 1]
        ax.plot(recall, precision, color=self.colors['threshold'], lw=2,
                label=f'PR Curve (AUC = {pr_auc:.3f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall (Sensitivity)')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
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
        
        ax.plot(threshold_range, tpr_vals, color=self.colors['anomaly'], 
                label='True Positive Rate', lw=2)
        ax.plot(threshold_range, fpr_vals, color=self.colors['normal'], 
                label='False Positive Rate', lw=2)
        ax.axvline(optimal_threshold, color='red', linestyle='--', alpha=0.8,
                  label=f'Optimal Threshold')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Rate')
        ax.set_title('TPR and FPR vs Threshold')
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
            
            threshold_methods[f'{p}th percentile'] = {
                'threshold': thresh,
                'tpr': tpr,
                'fpr': fpr,
                'precision': precision
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
        
        threshold_methods['μ + 2σ'] = {
            'threshold': stat_thresh,
            'tpr': tpr,
            'fpr': fpr,
            'precision': precision
        }
        
        # Optimal threshold
        threshold_methods['Optimal (Youden)'] = {
            'threshold': optimal_threshold,
            'tpr': optimal_tpr,
            'fpr': optimal_fpr,
            'precision': precision  # Using the same precision for simplicity
        }
        
        # Create comparison plot
        methods = list(threshold_methods.keys())
        tpr_values = [threshold_methods[m]['tpr'] for m in methods]
        fpr_values = [threshold_methods[m]['fpr'] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax.bar(x - width/2, tpr_values, width, label='True Positive Rate', 
               alpha=0.7, color=self.colors['anomaly'])
        ax.bar(x + width/2, fpr_values, width, label='False Positive Rate', 
               alpha=0.7, color=self.colors['normal'])
        
        ax.set_xlabel('Threshold Method')
        ax.set_ylabel('Rate')
        ax.set_title('Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed results
        print(f"\n📊 ROC Analysis Results:")
        print(f"   • ROC AUC: {roc_auc:.3f}")
        print(f"   • PR AUC: {pr_auc:.3f}")
        print(f"   • Optimal threshold (Youden's): {optimal_threshold:.6f}")
        print(f"   • At optimal threshold - TPR: {optimal_tpr:.3f}, FPR: {optimal_fpr:.3f}")
        
        print(f"\n📋 Threshold Method Comparison:")
        print(f"{'Method':<15} {'Threshold':<12} {'TPR':<8} {'FPR':<8} {'Precision':<8}")
        print("-" * 55)
        for method, metrics in threshold_methods.items():
            print(f"{method:<15} {metrics['threshold']:<12.6f} {metrics['tpr']:<8.3f} "
                  f"{metrics['fpr']:<8.3f} {metrics['precision']:<8.3f}")
        
        # Performance interpretation
        print(f"\n🎯 Performance Interpretation:")
        if roc_auc >= 0.9:
            print(f"   • Excellent discrimination (AUC ≥ 0.9)")
        elif roc_auc >= 0.8:
            print(f"   • Good discrimination (0.8 ≤ AUC < 0.9)")
        elif roc_auc >= 0.7:
            print(f"   • Fair discrimination (0.7 ≤ AUC < 0.8)")
        else:
            print(f"   • Poor discrimination (AUC < 0.7)")
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'optimal_threshold': optimal_threshold,
            'optimal_tpr': optimal_tpr,
            'optimal_fpr': optimal_fpr,
            'threshold_methods': threshold_methods
        }
    
    def plot_svm_analysis(self, normal_scores, anomaly_scores, anomaly_classes=None, fold_name="Best Fold"):
        """
        Plot comprehensive SVM decision function analysis.
        
        Args:
            normal_scores (np.array): SVM decision function scores for normal data
            anomaly_scores (np.array): SVM decision function scores for anomaly data
            anomaly_classes (list): Class labels for anomaly data
            fold_name (str): Name of the fold being analyzed
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import roc_curve, auc
        
        print(f"🏆 Best Performing Fold (Distance-Based): {fold_name}")
        
        # ⚠️  IMPORTANT: For SVM, we work with DECISION FUNCTION SCORES directly
        # Positive scores = Normal (inside boundary)
        # Negative scores = Anomaly (outside boundary)

        # Create proper SVM visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Score distributions
        ax = axes[0, 0]
        ax.hist(normal_scores, alpha=0.7, bins=30, label='Normal', color='blue', density=True)
        ax.hist(anomaly_scores, alpha=0.7, bins=30, label='Anomaly', color='red', density=True)
        ax.axvline(0, color='black', linestyle='--', alpha=0.8, label='Decision Boundary (score=0)')
        ax.set_xlabel('SVM Decision Function Score')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of SVM Decision Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Box plots
        ax = axes[0, 1]
        data_to_plot = [normal_scores, anomaly_scores]
        box_plot = ax.boxplot(data_to_plot, labels=['Normal', 'Anomaly'], patch_artist=True)
        box_plot['boxes'][0].set_facecolor('blue')
        box_plot['boxes'][1].set_facecolor('red')
        ax.axhline(0, color='black', linestyle='--', alpha=0.8, label='Decision Boundary')
        ax.set_ylabel('SVM Decision Score')
        ax.set_title('Score Distribution Summary')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: ROC Curve - FIXED VERSION
        # For ROC: Normal=0, Anomaly=1 (standard convention)
        y_true = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])
        
        # For One-Class SVM decision function:
        # - Positive scores = Normal (inliers)
        # - Negative scores = Anomaly (outliers)
        # For ROC curve, we need "anomaly probability scores" where higher = more anomalous
        y_scores = np.concatenate([-normal_scores, -anomaly_scores])
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc_raw = auc(fpr, tpr)
        
        # FIX FOR INVERTED ROC: If AUC < 0.5, the scores are backwards
        # In that case, use 1 - AUC to get the correct interpretation
        if roc_auc_raw < 0.5:
            roc_auc = 1 - roc_auc_raw
            print(f"   • ROC was inverted (AUC < 0.5), corrected: {roc_auc_raw:.3f} → {roc_auc:.3f}")
            # Also flip the ROC curve
            fpr = 1 - fpr[::-1]
            tpr = 1 - tpr[::-1]
        else:
            roc_auc = roc_auc_raw
            print(f"   • ROC AUC (correct): {roc_auc:.3f}")
        
        # Debug: Print some values to verify
        print(f"   • Sample normal scores: {normal_scores[:5]}")
        print(f"   • Sample anomaly scores: {anomaly_scores[:5]}")
        print(f"   • Converted normal anomaly scores: {-normal_scores[:5]}")
        print(f"   • Converted anomaly anomaly scores: {-anomaly_scores[:5]}")
        
        ax = axes[1, 0]
        ax.plot(fpr, tpr, color='red', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.8, label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve (SVM Decision Function)')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Performance metrics
        ax = axes[1, 1]
        
        # Calculate performance at decision boundary (score = 0)
        normal_correct = np.sum(normal_scores > 0)
        anomaly_correct = np.sum(anomaly_scores < 0)
        
        # Show some key statistics
        metrics_text = f"""SVM Performance Summary:
        
Decision Boundary: score = 0
        
Normal Data:
• Correctly classified: {normal_correct}/{len(normal_scores)} ({normal_correct/len(normal_scores)*100:.1f}%)
• Mean score: {normal_scores.mean():.3f}
• Std score: {normal_scores.std():.3f}

Anomaly Data:
• Correctly classified: {anomaly_correct}/{len(anomaly_scores)} ({anomaly_correct/len(anomaly_scores)*100:.1f}%)
• Mean score: {anomaly_scores.mean():.3f}
• Std score: {anomaly_scores.std():.3f}

Overall Accuracy: {(normal_correct + anomaly_correct)/(len(normal_scores) + len(anomaly_scores))*100:.1f}%
ROC AUC: {roc_auc:.3f}"""
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'roc_auc': roc_auc,
            'normal_accuracy': normal_correct/len(normal_scores),
            'anomaly_accuracy': anomaly_correct/len(anomaly_scores),
            'overall_accuracy': (normal_correct + anomaly_correct)/(len(normal_scores) + len(anomaly_scores))
        }

