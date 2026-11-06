"""
Helpers for the supervised classification notebook (notebook 4).

The models, the metrics and the confusion matrix come from the 3W Toolkit and
are called in the notebook. What stays here has no toolkit equivalent: reading
the windows saved by notebook 1 with their fold, and turning them into the
flattened, class-balanced arrays the classifiers are trained on.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .data_augmentation import quick_balance_classes
from .data_persistence import DataPersistence

# Transient classes (101-109) are the transition into the fault class 1-9
TRANSIENT_TO_BASE = {101: 1, 102: 2, 105: 5, 106: 6, 107: 7, 108: 8, 109: 9}


def load_3w_data(config, verbose=True):
    """
    Load the windows saved by notebook 1, with the fold of each window.

    Returns:
        tuple: (train_dfs, train_classes, train_fold_info,
                test_dfs, test_classes, test_fold_info)
    """
    if verbose:
        print("Loading 3W Dataset for Classification")
        print("=" * 40)

    try:
        persistence = DataPersistence(base_dir=config.PROCESSED_DATA_DIR, verbose=False)
        windowed_dir = os.path.join(persistence.cv_splits_dir, "windowed")
        if not os.path.exists(windowed_dir):
            raise FileNotFoundError("Run Data Treatment notebook first!")

        fold_dirs = sorted(
            d
            for d in os.listdir(windowed_dir)
            if d.startswith("fold_") and os.path.isdir(os.path.join(windowed_dir, d))
        )
        if verbose:
            print(f"Found {len(fold_dirs)} folds")

        train_dfs, train_classes, train_fold_info = [], [], []
        test_dfs, test_classes, test_fold_info = [], [], []
        for fold_name in fold_dirs:
            for split, dfs, classes, info in (
                ("train", train_dfs, train_classes, train_fold_info),
                ("test", test_dfs, test_classes, test_fold_info),
            ):
                path = os.path.join(
                    windowed_dir, fold_name, f"{split}_windowed.{config.SAVE_FORMAT}"
                )
                if os.path.exists(path):
                    fold_dfs, fold_classes = persistence._load_dataframes(
                        path, config.SAVE_FORMAT
                    )
                    dfs.extend(fold_dfs)
                    classes.extend(fold_classes)
                    info.extend([fold_name] * len(fold_dfs))

        if verbose:
            print("✅ Data loaded successfully!")
            print(f"   Training windows: {len(train_dfs)}")
            print(f"   Test windows: {len(test_dfs)}")
            if train_dfs:
                print(f"   Window shape: {train_dfs[0].shape}")
        return (
            train_dfs,
            train_classes,
            train_fold_info,
            test_dfs,
            test_classes,
            test_fold_info,
        )

    except Exception as e:
        if verbose:
            print(f"❌ Error: {e}")
        return [], [], [], [], [], []


def _window_classes(dfs: List[pd.DataFrame], fallback: List[str]) -> list:
    """Class of each window: the last value of its label column, transient classes mapped."""
    classes = [
        df["class"].iloc[-1] if "class" in df.columns else fallback[i]
        for i, df in enumerate(dfs)
    ]
    return [TRANSIENT_TO_BASE.get(c, c) for c in classes]


def _limit_per_class(dfs, classes, max_per_class):
    """Random subset with at most `max_per_class` windows of each class."""
    chosen = []
    for target in np.unique(classes):
        positions = [i for i, c in enumerate(classes) if c == target]
        n = min(max_per_class, len(positions))
        chosen.extend(np.random.choice(positions, size=n, replace=False))
    return [dfs[i] for i in chosen], [classes[i] for i in chosen]


def _flatten(dfs: List[pd.DataFrame]) -> np.ndarray:
    """One feature vector per window: all the sensor values, without the label column."""
    return np.array(
        [df[[c for c in df.columns if c != "class"]].values.flatten() for df in dfs]
    )


def prepare_window_data(
    train_dfs: List[pd.DataFrame],
    train_classes: List[str],
    test_dfs: List[pd.DataFrame],
    test_classes: List[str],
    selected_classes: Optional[list] = None,
    balance_strategy: str = "combined",
    max_samples_per_class: int = 1000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """
    Flatten the windows into arrays for the classifiers.

    The training set is balanced and capped per class; the test set is only
    capped (to a third of that), keeping its real class distribution.

    Returns:
        (X_train, y_train, X_test, y_test, label_encoder) with encoded labels
    """
    np.random.seed(random_state)
    y_train = _window_classes(train_dfs, train_classes)
    y_test = _window_classes(test_dfs, test_classes)

    if selected_classes is not None:
        keep_train = [i for i, c in enumerate(y_train) if c in selected_classes]
        keep_test = [i for i, c in enumerate(y_test) if c in selected_classes]
        train_dfs = [train_dfs[i] for i in keep_train]
        y_train = [y_train[i] for i in keep_train]
        test_dfs = [test_dfs[i] for i in keep_test]
        y_test = [y_test[i] for i in keep_test]

    train_dfs, y_train = quick_balance_classes(
        train_dfs, y_train, strategy=balance_strategy
    )
    train_dfs, y_train = _limit_per_class(train_dfs, y_train, max_samples_per_class)
    test_dfs, y_test = _limit_per_class(test_dfs, y_test, max_samples_per_class // 3)

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(np.array(y_train))
    y_test_encoded = encoder.transform(np.array(y_test))
    return (
        _flatten(train_dfs),
        y_train_encoded,
        _flatten(test_dfs),
        y_test_encoded,
        encoder,
    )
