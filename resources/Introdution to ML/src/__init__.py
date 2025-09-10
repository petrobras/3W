"""
3W Dataset Processing and Analysis Toolkit

This package provides utilities for loading, preprocessing, and analyzing 
the 3W dataset for oil well fault detection.
"""

__version__ = "1.0.0"
__author__ = "3W Workshop Team"
__email__ = "your-email@example.com"

from .data_loader import DataLoader
from .preprocessing import DataPreprocessor
from .visualization import DataVisualizer
from .cross_validation import CrossValidator
from .data_augmentation import DataAugmentor
from .data_persistence import DataPersistence
from .training_utils import training_notebook_setup, quick_load_cv_data, get_fold_data
from . import config

__all__ = [
    'DataLoader', 'DataPreprocessor', 'DataVisualizer', 'CrossValidator', 
    'DataAugmentor', 'DataPersistence', 'training_notebook_setup', 
    'quick_load_cv_data', 'get_fold_data', 'config'
]
