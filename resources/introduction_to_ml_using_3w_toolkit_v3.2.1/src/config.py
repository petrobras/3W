"""
Configuration settings for 3W Dataset Processing Toolkit

Parameter Ranges and Validation:
- SAMPLING_RATE: 1-10 (higher = more aggressive sampling)
- WINDOW_SIZE: 100-1000 (depends on data frequency and analysis needs)
- WINDOW_OVERLAP: 0.0 to <1.0 (fraction of a window shared with the next one)
- N_FOLDS: 2-10 (more folds = better CV but higher computation)
- RANDOM_SEED: Any integer (for reproducibility)
- MAX_FILES_PER_CLASS: 50-500 (memory optimization, real data prioritized)
"""

# Version of the 3W Toolkit with which the results of this resource were measured
EXPECTED_TOOLKIT_VERSION = "3.2.1"

# Dataset settings
TARGET_FEATURES = ["P-PDG", "P-TPT", "T-TPT", "class"]
CLASS_COLUMN = "class"
# Event provenances loaded after the real ones, up to MAX_FILES_PER_CLASS.
# Options: "simulated", "drawn". Use ["simulated"] to exclude hand-drawn events
# (see the companion book, appendix "Reproduction conditions").
NON_REAL_EVENT_TYPES = ["simulated", "drawn"]
MAX_FILES_PER_CLASS = (
    50  # Range: 50-500, Maximum files to load per class (memory optimization)
)
# 50 files per class peak at about 2.3 GB of RAM (notebook 1, CPU, default settings)
# Processing settings
# 'normalizer' is the toolkit `Normalize` step, fitted on the training events of
# each fold only. The other options are scikit-learn scalers, fitted per event.
DEFAULT_SCALING_METHOD = (
    "normalizer"  # Options: 'normalizer', 'standard', 'minmax', 'robust'
)
NORMALIZE_NORM = "l2"  # Dispersion used by `Normalize`: 'l1', 'l2' (z-score) or 'max'
RANDOM_SEED = 42  # Range: any integer

# Data sampling settings
ENABLE_DATA_SAMPLING = True  # Enable sampling to reduce data size
SAMPLING_RATE = 5  # Range: 1-10, Sample every nth row (1 line each n)
SAMPLING_METHOD = "uniform"  # Options: 'uniform', 'random'

# Cross-validation settings
N_FOLDS = 3  # Range: 2-10
CV_RANDOM_STATE = 42

# Time windowing settings (the parameters of the toolkit `Windowing` step)
WINDOW_SIZE = 300  # Range: 100-1000, samples per window
WINDOW_OVERLAP = 0.5  # Range: 0.0 to <1.0, 0.5 = windows 150 samples apart

# File output settings
SAVE_FORMAT = "pickle"

# Data persistence settings
PROCESSED_DATA_DIR = "processed_data"

# Unsupervised settings
# Data Loading Configuration for Per-Fold Evaluation
MAX_NORMAL_SAMPLES = 500  # Limit normal samples per fold for faster processing
MAX_ANOMALY_SAMPLES = 200  # Limit anomaly samples per fold
SELECTED_ANOMALY_CLASSES = [3, 4, 8]  # Target anomaly classes


# Visualization settings
VISUALIZATION_MAX_SAMPLES = (
    100  # Max samples per class for visualization (Range: 10-100)
)
VISUALIZATION_RANDOM_SEED = 42  # For reproducible sampling

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
