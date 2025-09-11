"""
3W Dataset Loading Utilities

This package provides comprehensive data loading utilities for the 3W oil well dataset,
including windowed time series data loading, cross-validation support, and data validation.
"""

from .three_w_data_loader import ThreeWDataLoader, create_data_loader, load_complete_dataset

__version__ = "1.0.0"
__author__ = "3W Team"

__all__ = [
    'ThreeWDataLoader',
    'create_data_loader', 
    'load_complete_dataset'
]
