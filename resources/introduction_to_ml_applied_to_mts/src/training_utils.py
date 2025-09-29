"""
Quick Data Loader for Training Notebooks

This utility provides simple functions to quickly load processed data
from the Data Treatment notebook for use in training/modeling notebooks.
"""

import os
import json
from typing import List, Dict, Any, Tuple, Optional
from .data_persistence import DataPersistence
from . import config


def quick_load_cv_data(
    base_dir: str = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Quickly load cross-validation data and metadata.

    Args:
        base_dir (str, optional): Directory containing processed data.
                                If None, uses config.PROCESSED_DATA_DIR

    Returns:
        tuple: (cv_folds, processing_metadata)
    """
    if base_dir is None:
        base_dir = config.PROCESSED_DATA_DIR

    # Load CV folds
    persistence = DataPersistence(base_dir=base_dir, verbose=True)
    cv_folds = persistence.load_cv_folds()

    # Load processing metadata
    metadata_file = os.path.join(base_dir, "metadata", "processing_summary.json")
    processing_metadata = {}
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            processing_metadata = json.load(f)

    print(f"✅ Loaded {len(cv_folds)} CV folds from: {base_dir}")

    return cv_folds, processing_metadata


def get_fold_data(
    cv_folds: List[Dict[str, Any]], fold_number: int
) -> Tuple[List, List, List, List]:
    """
    Get specific fold data for training.

    Args:
        cv_folds (List[Dict]): Loaded CV folds
        fold_number (int): Fold number (1-indexed)

    Returns:
        tuple: (train_dfs, train_classes, test_dfs, test_classes)
    """
    if fold_number < 1 or fold_number > len(cv_folds):
        raise ValueError(f"Fold number must be between 1 and {len(cv_folds)}")

    fold = cv_folds[fold_number - 1]
    return (
        fold["train_dfs"],
        fold["train_classes"],
        fold["test_dfs"],
        fold["test_classes"],
    )


def print_data_summary(
    cv_folds: List[Dict[str, Any]], processing_metadata: Dict[str, Any] = None
):
    """
    Print summary of loaded data.

    Args:
        cv_folds (List[Dict]): Loaded CV folds
        processing_metadata (Dict, optional): Processing metadata
    """
    print("📊 Data Summary:")
    print("-" * 40)

    if processing_metadata:
        print(
            f"   • Processing config: {processing_metadata.get('processing_config', {}).get('scaling_method', 'unknown')} scaling"
        )
        print(
            f"   • Window size: {processing_metadata.get('processing_config', {}).get('window_size', 'unknown')}"
        )
        print(
            f"   • Features: {processing_metadata.get('processing_config', {}).get('target_features', [])}"
        )

    print(f"   • Cross-validation folds: {len(cv_folds)}")

    if cv_folds:
        # Calculate totals
        total_train_samples = sum(len(fold["train_dfs"]) for fold in cv_folds)
        total_test_samples = sum(len(fold["test_dfs"]) for fold in cv_folds)
        avg_train = total_train_samples // len(cv_folds)
        avg_test = total_test_samples // len(cv_folds)

        print(f"   • Avg training samples per fold: {avg_train}")
        print(f"   • Avg test samples per fold: {avg_test}")

        # Show fold details
        print(f"\n📁 Fold Details:")
        for i, fold in enumerate(cv_folds):
            fold_num = i + 1
            train_count = len(fold["train_dfs"])
            test_count = len(fold["test_dfs"])
            print(f"   • Fold {fold_num}: {train_count} train, {test_count} test")


def training_notebook_setup(base_dir: str = None) -> Dict[str, Any]:
    """
    Complete setup for training notebook.

    Args:
        base_dir (str, optional): Directory containing processed data

    Returns:
        Dict: Dictionary containing all loaded data and utilities
    """
    print("🚀 Training Notebook Setup")
    print("=" * 50)

    # Load data
    cv_folds, processing_metadata = quick_load_cv_data(base_dir)

    # Import utilities
    from .data_augmentation import DataAugmentor
    from .cross_validation import CrossValidator

    # Initialize utilities
    augmentor = DataAugmentor(
        random_state=processing_metadata.get("processing_config", {}).get(
            "random_seed", 42
        ),
        verbose=True,
    )

    # Print summary
    print_data_summary(cv_folds, processing_metadata)

    print(f"\n🎯 Ready for Training!")
    print("   • Use setup_data['cv_folds'] for cross-validation")
    print("   • Use setup_data['augmentor'] for data augmentation")
    print("   • Use get_fold_data(cv_folds, fold_number) for specific folds")

    return {
        "cv_folds": cv_folds,
        "processing_metadata": processing_metadata,
        "augmentor": augmentor,
        "config": processing_metadata.get("processing_config", {}),
    }


# Example usage functions
def example_training_loop(cv_folds: List[Dict[str, Any]]):
    """
    Show example of how to iterate through CV folds for training.

    Args:
        cv_folds (List[Dict]): Cross-validation folds
    """
    print("📋 Example Training Loop:")
    print("-" * 30)

    for fold_num in range(1, len(cv_folds) + 1):
        train_dfs, train_classes, test_dfs, test_classes = get_fold_data(
            cv_folds, fold_num
        )

        print(f"\n📁 Fold {fold_num}:")
        print(f"   • Training: {len(train_dfs)} samples")
        print(f"   • Testing: {len(test_dfs)} samples")

        # This is where your training code would go:
        # model = YourModel()
        # model.fit(train_dfs, train_classes)
        # predictions = model.predict(test_dfs)
        # score = evaluate_model(predictions, test_classes)
        # print(f"   • Fold {fold_num} score: {score}")

    print("\n✅ Complete all folds and average results for final performance")
