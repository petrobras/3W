"""
Configuration settings for 3W Dataset Processing Toolkit
"""

# Dataset settings
DATASET_PATH = "../../dataset/"
TARGET_FEATURES = ["P-PDG", "P-TPT", "T-TPT", "class"]
RAW_DATA_DIR = "./processed_data/cv_splits/raw/"
CLASS_COLUMN = "class"

# Processing settings
DEFAULT_SCALING_METHOD = "minmax"
RANDOM_SEED = 42
MIN_SAMPLES_THRESHOLD = 100

# Data sampling settings
ENABLE_DATA_SAMPLING = True  # Enable sampling to reduce data size
SAMPLING_RATE = 5  # Sample every 5th row (1 line each 5)
SAMPLING_METHOD = "uniform"  # Options: 'uniform', 'random'

# Cross-validation settings
N_FOLDS = 3
CV_RANDOM_STATE = 42
CV_VERBOSE = True
FALLBACK_REAL_PROPORTION = 0.7  # For real/simulated data separation fallback

# Time windowing settings
WINDOW_SIZE = 300
WINDOW_STRIDE = WINDOW_SIZE // 2  # Overlapping windows (150), use WINDOW_SIZE for non-overlapping
MIN_WINDOW_SIZE = 300  # Only keep full-size windows

# Data analysis settings
SAMPLE_ANALYSIS_MIN_SAMPLES = 100
REQUIRED_COLUMNS = TARGET_FEATURES

# Sensor column names for plotting
PRESSURE_COLUMN = "P-TPT"
TEMPERATURE_COLUMN = "T-TPT"


# File output settings
OUTPUT_DIR = "processed_data"
SAVE_FORMAT = "pickle"  # Options: 'pickle' (fastest loading), 'parquet' (cross-platform), 'csv' (universal)

# Data persistence settings
PROCESSED_DATA_DIR = "processed_data"
CV_SPLITS_DIR = "cv_splits"
METADATA_FILE = "processing_metadata.json"

# Save configuration
SAVE_CV_FOLDS = True
SAVE_WINDOWED_DATA = True
SAVE_RAW_SPLITS = True
SAVE_METADATA = True

# File naming patterns
FOLD_PATTERN = "fold_{fold_num}"
TRAIN_PATTERN = "train_data"
TEST_PATTERN = "test_data"
WINDOWED_PATTERN = "windowed_data"
METADATA_PATTERN = "metadata"

# Supervised Classification settings
CLASSIFICATION_CONFIG = {
    # Class selection options - Choose which classes to include in analysis
    "selected_classes": [1, 2, 3, 4, 5, 8],  # Specific fault types of interest
    # Alternative class selection examples:
    # 'selected_classes': None,              # Default: all fault types (exclude class 0)
    # 'selected_classes': [1, 2, 3, 4, 5],  # Focus on first 5 fault types
    # 'selected_classes': [7, 8, 9],        # Focus on last 3 fault types
    # 'selected_classes': [1, 3, 5, 7, 9],  # Focus on odd-numbered fault types
    # Test data balancing settings
    "balance_test": False,  # Balance test data for robust evaluation
    "min_test_samples_per_class": 300,  # Ensure minimum samples per class in test
    # Training settings
    "balance_classes": True,  # Use data augmentation for class balancing (training)
    "balance_strategy": "combined",  # Combined over/undersampling strategy
    "max_samples_per_class": 1000,  # Limit for computational efficiency (training)
    "verbose": True,  # Show detailed progress
}

# Pre-configured class selection sets for common use cases
CLASSIFICATION_PRESETS = {
    "all_faults": None,  # All fault types (exclude only class 0)
    "specific_faults": [2, 3, 8],  # User-defined specific faults
    "early_faults": [1, 2, 3, 4, 5],  # First 5 fault types
    "late_faults": [7, 8, 9],  # Last 3 fault types
    "odd_faults": [1, 3, 5, 7, 9],  # Odd-numbered fault types
    "even_faults": [2, 4, 6, 8],  # Even-numbered fault types
    "critical_faults": [3, 6, 8, 9],  # Example: Critical operational faults
    "minor_faults": [1, 2, 4, 5, 7],  # Example: Minor operational faults
    "binary_test": [3, 8],  # Simple binary classification test
}

# Visualization settings
DEFAULT_FIGSIZE = (12, 8)

# Display settings
SEPARATOR_LENGTH = 50
HEADER_SEPARATOR_LENGTH = 70
PROGRESS_SEPARATOR_LENGTH = 60
