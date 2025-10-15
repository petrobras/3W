"""
This package provides utilities for loading, preprocessing, and analyzing the 3W dataset for oil well fault detection.
"""

__version__ = "1.0.0"
__maintainer__ = ["Lucas Lopes <lucas.lopes@lccv.ufal.br>"]

from .data_loader import DataLoader
from .preprocessing import DataPreprocessor
from .visualization import DataVisualizer
from .cross_validation import CrossValidator
from .data_augmentation import DataAugmentor
from .data_persistence import DataPersistence
from .autoencoder_models import StableLSTMAutoencoder
from .unsupervised_preprocessing import UnsupervisedDataPreprocessor
from .anomaly_detection import AnomalyDetector
from . import config

__all__ = [
    "DataLoader",
    "DataPreprocessor",
    "DataVisualizer",
    "CrossValidator",
    "DataAugmentor",
    "DataPersistence",
    "StableLSTMAutoencoder",
    "UnsupervisedDataPreprocessor",
    "AnomalyDetector",
    "config",
]
