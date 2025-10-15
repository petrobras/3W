"""
Configuration settings for 3W Dataset Processing Toolkit

Parameter Ranges and Validation:
- SAMPLING_RATE: 1-10 (higher = more aggressive sampling)
- WINDOW_SIZE: 100-1000 (depends on data frequency and analysis needs)
- N_FOLDS: 2-10 (more folds = better CV but higher computation)
- RANDOM_SEED: Any integer (for reproducibility)
- MAX_FILES_PER_CLASS: 50-500 (memory optimization, real data prioritized)
"""

# Dataset settings
TARGET_FEATURES = ["P-PDG", "P-TPT", "T-TPT", "class"]
CLASS_COLUMN = "class"
MAX_FILES_PER_CLASS = 50  # Range: 50-500, Maximum files to load per class (memory optimization)
# 50 files will use about 4 to 5 GB of RAM
# Processing settings
DEFAULT_SCALING_METHOD = (
    "minmax"  # Options: 'standard', 'minmax', 'robust', 'normalizer'
)
RANDOM_SEED = 42  # Range: any integer

# Data sampling settings
ENABLE_DATA_SAMPLING = True  # Enable sampling to reduce data size
SAMPLING_RATE = 5  # Range: 1-10, Sample every nth row (1 line each n)
SAMPLING_METHOD = "uniform"  # Options: 'uniform', 'random'

# Cross-validation settings
N_FOLDS = 3  # Range: 2-10
CV_RANDOM_STATE = 42
CV_VERBOSE = False  # Reduced verbosity
FALLBACK_REAL_PROPORTION = (
    0.7  # Range: 0.5-0.9, For real/simulated data separation fallback
)

# Time windowing settings
WINDOW_SIZE = 300  # Range: 100-1000
WINDOW_STRIDE = (
    WINDOW_SIZE // 2
)  # Range: 1 to WINDOW_SIZE, Overlapping windows (150), use WINDOW_SIZE for non-overlapping
MIN_WINDOW_SIZE = 300  # Range: 50 to WINDOW_SIZE, Only keep full-size windows

# Data analysis settings
SAMPLE_ANALYSIS_MIN_SAMPLES = 100
REQUIRED_COLUMNS = TARGET_FEATURES

# Sensor column names for plotting
PRESSURE_COLUMN = "P-TPT"
TEMPERATURE_COLUMN = "T-TPT"


# File output settings
SAVE_FORMAT = "pickle"  # Options: 'pickle' (fastest loading), 'parquet' (cross-platform), 'csv' (universal)

# Data persistence settings
PROCESSED_DATA_DIR = "processed_data"

# Supervised Classification settings
CLASSIFICATION_CONFIG = {
    # Class selection options - Choose which classes to include in analysis
    "selected_classes": [3, 4, 8],  # Specific fault types of interest
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


# Display settings
SEPARATOR_LENGTH = 50
HEADER_SEPARATOR_LENGTH = 60  # Reduced for more concise display

def validate_config():
    """Validate configuration parameters and provide warnings for invalid values."""
    warnings = []

    # Validate sampling rate
    if not (1 <= SAMPLING_RATE <= 10):
        warnings.append(f"SAMPLING_RATE ({SAMPLING_RATE}) should be between 1-10")

    # Validate window size
    if not (100 <= WINDOW_SIZE <= 1000):
        warnings.append(f"WINDOW_SIZE ({WINDOW_SIZE}) should be between 100-1000")

    # Validate cross-validation folds
    if not (2 <= N_FOLDS <= 10):
        warnings.append(f"N_FOLDS ({N_FOLDS}) should be between 2-10")

    # Validate scaling method
    valid_scaling = ["standard", "minmax", "robust", "normalizer"]
    if DEFAULT_SCALING_METHOD not in valid_scaling:
        warnings.append(
            f"DEFAULT_SCALING_METHOD ({DEFAULT_SCALING_METHOD}) should be one of {valid_scaling}"
        )

    return warnings

# Visualization settings
VISUALIZATION_MAX_SAMPLES = (
    100  # Max samples per class for visualization (Range: 10-100)
)
VISUALIZATION_RANDOM_SEED = 42  # For reproducible sampling
VISUALIZATION_FIGURE_SIZE = (16, 12)  # Default figure size for plots
VISUALIZATION_DPI = 100  # Plot resolution
VISUALIZATION_STYLE = "default"  # Matplotlib style

# Dimensionality reduction settings
TSNE_CONFIGS = [
    {"perplexity": 30, "learning_rate": 200, "title": "Standard t-SNE"},
    {"perplexity": 10, "learning_rate": 100, "title": "Low Perplexity (Local Focus)"},
    {"perplexity": 50, "learning_rate": 300, "title": "High Perplexity (Global)"},
    {"perplexity": 30, "learning_rate": 500, "title": "Fast Learning Rate"},
]

UMAP_CONFIGS = [
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
]
