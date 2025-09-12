"""
Data Loading Module for 3W Dataset

This module provides functionality to load and organize the 3W dataset
containing sensor data for fault detection in oil wells.
"""

import os
from glob import glob
import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Dict, Any
from collections import Counter


class DataLoader:
    """
    A class for loading and organizing the 3W dataset.
    
    The 3W dataset contains sensor data from oil wells organized in folders
    representing different fault classes (0-9).
    """
    
    def __init__(self, dataset_path: str = '../../dataset/'):
        """
        Initialize the DataLoader.
        
        Args:
            dataset_path (str): Path to the 3W dataset directory
        """
        self.dataset_path = dataset_path
        self.class_folders = []
        self.stats = {}
        
    def explore_structure(self) -> Dict[str, Any]:
        """
        Explore the dataset structure and return information about files.
        
        Returns:
            Dict containing information about the dataset structure
        """
        print("📁 Dataset Structure:")
        print("=" * 50)
        
        structure_info = {
            'classes': [],
            'total_files': 0,
            'real_files': 0,
            'simulated_files': 0
        }
        
        for folder_name in sorted(os.listdir(self.dataset_path)):
            if folder_name not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                continue
                
            self.class_folders.append(folder_name)
            structure_info['classes'].append(folder_name)
            folder_path = os.path.join(self.dataset_path, folder_name)
            
            if os.path.isdir(folder_path):
                files = os.listdir(folder_path)
                parquet_files = [f for f in files if f.endswith('.parquet')]
                structure_info['total_files'] += len(parquet_files)
                
                print(f"Class {folder_name}: {len(parquet_files)} parquet files")
                
                # Show file types (real vs simulated)
                real_files = [f for f in parquet_files if 'WELL' in f]
                simulated_files = [f for f in parquet_files if 'WELL' not in f]
                
                structure_info['real_files'] += len(real_files)
                structure_info['simulated_files'] += len(simulated_files)
                
                print(f"  └── Real data: {len(real_files)} files")
                print(f"  └── Simulated data: {len(simulated_files)} files")
                print()
        
        print(f"Total classes found: {len(self.class_folders)}")
        print(f"Classes: {self.class_folders}")
        
        return structure_info
    
    def load_dataset(self, class_folders: List[str] = None) -> Tuple[List[pd.DataFrame], List[str], Dict[str, Any]]:
        """
        Load the 3W dataset from parquet files.
        
        Args:
            class_folders (List[str], optional): List of class folders to load. 
                                               If None, loads all available classes.
        
        Returns:
            Tuple containing:
                - List of dataframes with loaded data
                - List of corresponding class labels
                - Dictionary with dataset statistics
        """
        if class_folders is None:
            class_folders = self.class_folders
        
        # Initialize storage containers - separate real and simulated
        dfs_real = []
        dfs_simulated = []
        classes_real = []
        classes_simulated = []
        filenames_real = []
        filenames_simulated = []
        
        # Statistics tracking
        stats = {
            'total_files': 0,
            'real_files': 0,
            'simulated_files': 0,
            'classes_count': {str(i): {'real': 0, 'simulated': 0} for i in range(10)},
            'empty_files': 0,
            'total_samples': 0,
            'file_tracking': {'real': [], 'simulated': []}
        }
        
        print("🔄 Loading 3W Dataset...")
        print("=" * 50)
        
        # Process each class folder
        for class_folder in tqdm(sorted(class_folders), desc="Processing classes"):
            folder_path = os.path.join(self.dataset_path, class_folder)
            parquet_files = glob(os.path.join(folder_path, "*.parquet"))
            
            # Process each parquet file in the class folder
            for file_path in tqdm(parquet_files, desc=f"Class {class_folder}", leave=False):
                stats['total_files'] += 1
                
                try:
                    df = self._load_and_clean_file(file_path, stats)
                    
                    if df is None or len(df) == 0:
                        continue
                    
                    # Categorize as real or simulated data based on filename patterns
                    filename = os.path.basename(file_path)
                    is_real_data = self._is_real_data(filename)
                    
                    stats['total_samples'] += len(df)
                    
                    if is_real_data:
                        dfs_real.append(df)
                        classes_real.append(class_folder)
                        filenames_real.append(filename)
                        stats['real_files'] += 1
                        stats['classes_count'][class_folder]['real'] += 1
                        stats['file_tracking']['real'].append(filename)
                    else:
                        dfs_simulated.append(df)
                        classes_simulated.append(class_folder)
                        filenames_simulated.append(filename)
                        stats['simulated_files'] += 1
                        stats['classes_count'][class_folder]['simulated'] += 1
                        stats['file_tracking']['simulated'].append(filename)
                
                except Exception as e:
                    print(f"❌ Error loading {file_path}: {str(e)}")
                    stats['empty_files'] += 1
                    continue
        
        # Combine real and simulated data
        dfs_3w = dfs_real + dfs_simulated
        classes_3w = classes_real + classes_simulated
        
        # Add metadata about data sources
        stats['real_data'] = {
            'dfs': dfs_real,
            'classes': classes_real, 
            'filenames': filenames_real
        }
        stats['simulated_data'] = {
            'dfs': dfs_simulated,
            'classes': classes_simulated,
            'filenames': filenames_simulated
        }
        
        # Print loading summary
        self._print_loading_summary(stats, len(dfs_3w))
        
        self.stats = stats
        return dfs_3w, classes_3w, stats
    
    def _is_real_data(self, filename: str) -> bool:
        """
        Determine if a file contains real or simulated data based on filename patterns.
        
        Args:
            filename (str): Name of the file
            
        Returns:
            bool: True if real data, False if simulated
        """
        # Based on 3W dataset conventions:
        # Real data: Contains 'WELL' in filename
        # Simulated data: Starts with 'SIMULATED' or 'OLGA' 
        # Hand-drawn data: Starts with 'DRAWN' or 'DESENHADA'
        
        filename_upper = filename.upper()
        
        # Check for simulated patterns first
        if (filename_upper.startswith('SIMULATED') or 
            filename_upper.startswith('OLGA') or
            filename_upper.startswith('DRAWN') or 
            filename_upper.startswith('DESENHADA')):
            return False
            
        # Check for real data patterns
        if 'WELL' in filename_upper:
            return True
            
        # Default assumption for other patterns (could be refined)
        # If it doesn't match simulated patterns and contains well-like patterns, consider real
        return True
    
    def get_separated_data(self) -> Dict[str, Any]:
        """
        Get real and simulated data separately.
        
        Returns:
            Dictionary containing separated real and simulated data
        """
        if not hasattr(self, 'stats') or not self.stats:
            raise ValueError("No data loaded yet. Call load_dataset() first.")
            
        return {
            'real': self.stats['real_data'],
            'simulated': self.stats['simulated_data']
        }
    
    def _load_and_clean_file(self, file_path: str, stats: Dict[str, Any]) -> pd.DataFrame:
        """
        Load and clean a single parquet file.
        
        Args:
            file_path (str): Path to the parquet file
            stats (Dict): Statistics dictionary to update
            
        Returns:
            pd.DataFrame: Cleaned dataframe or None if file is invalid
        """
        # Load data
        df = pd.read_parquet(file_path)
        
        # Debug information for first few files
        if stats['total_files'] < 3:
            print(f"Debug - File: {os.path.basename(file_path)}")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Has 'class' column: {'class' in df.columns}")
        
        # Skip files without any data
        if len(df) == 0:
            stats['empty_files'] += 1
            return None
        
        # Check if we have class column
        if 'class' not in df.columns:
            stats['empty_files'] += 1
            if stats['total_files'] < 3:
                print(f"  Skipping - no 'class' column")
            return None
        
        # Remove rows with missing class labels
        df = df.dropna(subset=['class']).reset_index(drop=True)
        
        if len(df) == 0:
            stats['empty_files'] += 1
            if stats['total_files'] < 3:
                print(f"  Skipping - all class values are NaN")
            return None
        
        # Handle missing values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        
        # Remove rows with all NaN values in numeric columns
        df = df.dropna(subset=numeric_cols, how='all').reset_index(drop=True)
        
        if len(df) == 0:
            stats['empty_files'] += 1
            if stats['total_files'] < 3:
                print(f"  Skipping - no valid data after cleaning")
            return None
        
        if stats['total_files'] < 3:
            print(f"  ✅ Successfully loaded: {len(df)} samples")
        
        return df
    
    def _print_loading_summary(self, stats: Dict[str, Any], total_loaded: int) -> None:
        """Print loading summary statistics."""
        print("\n📊 Loading Summary:")
        print("=" * 50)
        print(f"Total files processed: {stats['total_files']}")
        print(f"Successfully loaded: {total_loaded} files")
        print(f"Real data files: {stats['real_files']}")
        print(f"Simulated data files: {stats['simulated_files']}")
        print(f"Empty/invalid files: {stats['empty_files']}")
        print(f"Total samples: {stats['total_samples']:,}")
    
    def filter_target_features(self, dfs: List[pd.DataFrame], classes: List[str], 
                              target_features: List[str]) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        Filter dataframes to contain only target features.
        
        Args:
            dfs (List[pd.DataFrame]): List of dataframes to filter
            classes (List[str]): List of corresponding class labels
            target_features (List[str]): List of features to keep
            
        Returns:
            Tuple containing filtered dataframes and their classes
        """
        print("🔍 Filtering Data to Key Sensor Variables")
        print("=" * 60)
        print(f"Target features: {target_features}")
        
        filtered_dfs = []
        filtered_classes = []
        
        for i, (df, class_label) in enumerate(zip(dfs, classes)):
            # Check which target features are available
            available_features = [col for col in target_features if col in df.columns]
            
            if len(available_features) >= 2:  # Need at least 2 features
                filtered_df = df[available_features].copy()
                
                # Only keep if we have some data
                if len(filtered_df.dropna()) > 0:
                    filtered_dfs.append(filtered_df)
                    filtered_classes.append(class_label)
                    
                    # Show info for first few dataframes
                    if len(filtered_dfs) <= 3:
                        print(f"\n📄 DataFrame {len(filtered_dfs)} (Class {class_label}):")
                        print(f"  Available features: {available_features}")
                        print(f"  Shape after filtering: {filtered_df.shape}")
                        print(f"  Missing values per column:")
                        for col in filtered_df.columns:
                            missing_count = filtered_df[col].isnull().sum()
                            missing_pct = (missing_count / len(filtered_df)) * 100
                            print(f"    {col}: {missing_count} ({missing_pct:.1f}%)")
        
        print(f"\n📊 Filtering Results:")
        print("=" * 40)
        print(f"Datasets after filtering: {len(filtered_dfs)}")
        print(f"Total samples: {sum(len(df) for df in filtered_dfs):,}")
        
        # Check feature availability
        if len(filtered_dfs) > 0:
            feature_availability = {}
            for feature in target_features:
                count = sum(1 for df in filtered_dfs if feature in df.columns)
                feature_availability[feature] = count
            
            print(f"\n🎯 Feature Availability Across Datasets:")
            print("-" * 40)
            for feature, count in feature_availability.items():
                percentage = (count / len(filtered_dfs)) * 100
                print(f"{feature}: {count}/{len(filtered_dfs)} datasets ({percentage:.1f}%)")
        
        return filtered_dfs, filtered_classes
    
    def get_dataset_info(self, dfs: List[pd.DataFrame], classes: List[str]) -> Dict[str, Any]:
        """
        Get comprehensive information about the loaded dataset.
        
        Args:
            dfs (List[pd.DataFrame]): List of dataframes
            classes (List[str]): List of corresponding class labels
            
        Returns:
            Dict containing dataset information
        """
        if not dfs:
            return {}
        
        info = {
            'total_dataframes': len(dfs),
            'class_distribution': dict(Counter(classes)),
            'sample_dataframe_shape': dfs[0].shape,
            'sample_columns': list(dfs[0].columns),
            'total_samples': sum(len(df) for df in dfs),
            'memory_usage_mb': sum(df.memory_usage(deep=True).sum() for df in dfs) / 1024**2
        }
        
        return info
