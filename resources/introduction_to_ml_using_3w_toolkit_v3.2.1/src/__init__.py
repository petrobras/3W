"""
This package provides utilities for loading, preprocessing, and analyzing the 3W dataset for oil well fault detection.
"""

__version__ = "1.0.0"
__maintainer__ = ["Lucas Lopes <lucas.lopes@lccv.ufal.br>"]

import logging

from .data_augmentation import DataAugmentor
from .data_persistence import DataPersistence
from . import config

# Importing ThreeWToolkit configures the root logger at INFO level, which floods
# the notebook outputs with library messages.
logging.getLogger().setLevel(logging.WARNING)

__all__ = [
    "DataAugmentor",
    "DataPersistence",
    "config",
]
