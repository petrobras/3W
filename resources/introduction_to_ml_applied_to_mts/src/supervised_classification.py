"""
Supervised Classification Module for 3W Dataset

This module provides comprehensive supervised classification algorithms and utilities
for the 3W oil well fault detection dataset including:
- Decision Trees and Random Forest
- Support Vector Machines (Linear and RBF)
- Neural Networks (Simple, Deep, Regularized)
- Model comparison and evaluation
- Feature engineering for time series data
- Class balancing using data augmentation
- Utility functions for student-friendly analysis

The module integrates with the existing data augmentation utilities for handling
class imbalance and provides a unified interface for training and evaluating
multiple classification algorithms.
"""

import numpy as np
import pandas as pd
import time
from typing import List, Tuple, Dict, Optional, Any
from collections import Counter

# Scikit-learn imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Local imports
from .data_augmentation import quick_balance_classes


# ============================================================
# UTILITY FUNCTIONS FOR STUDENT NOTEBOOKS
# ============================================================


def load_3w_data(config, verbose=True):
    """
    Load 3W dataset for classification analysis.

    Args:
        config: Configuration module
        verbose: Whether to print loading information

    Returns:
        tuple: (train_dfs, train_classes, train_fold_info, test_dfs, test_classes, test_fold_info)
    """
    from .data_persistence import DataPersistence
    import os

    if verbose:
        print("Loading 3W Dataset for Classification")
        print("=" * 40)

    try:
        # Initialize data persistence
        persistence = DataPersistence(base_dir=config.PROCESSED_DATA_DIR, verbose=False)
        windowed_dir = os.path.join(persistence.cv_splits_dir, "windowed")

        if not os.path.exists(windowed_dir):
            raise FileNotFoundError("Run Data Treatment notebook first!")

        # Find fold directories
        fold_dirs = [
            d
            for d in os.listdir(windowed_dir)
            if d.startswith("fold_") and os.path.isdir(os.path.join(windowed_dir, d))
        ]
        fold_dirs.sort()

        if verbose:
            print(f"Found {len(fold_dirs)} folds")

        # Load all data
        all_train_windows, all_train_classes, all_train_fold_info = [], [], []
        all_test_windows, all_test_classes, all_test_fold_info = [], [], []

        for fold_name in fold_dirs:
            fold_path = os.path.join(windowed_dir, fold_name)

            # Load training data
            train_file = os.path.join(fold_path, f"train_windowed.{config.SAVE_FORMAT}")
            if os.path.exists(train_file):
                fold_train_dfs, fold_train_classes = persistence._load_dataframes(
                    train_file, config.SAVE_FORMAT
                )
                all_train_windows.extend(fold_train_dfs)
                all_train_classes.extend(fold_train_classes)
                all_train_fold_info.extend([fold_name] * len(fold_train_dfs))

            # Load test data
            test_file = os.path.join(fold_path, f"test_windowed.{config.SAVE_FORMAT}")
            if os.path.exists(test_file):
                fold_test_dfs, fold_test_classes = persistence._load_dataframes(
                    test_file, config.SAVE_FORMAT
                )
                all_test_windows.extend(fold_test_dfs)
                all_test_classes.extend(fold_test_classes)
                all_test_fold_info.extend([fold_name] * len(fold_test_dfs))

        if verbose:
            print(f"✅ Data loaded successfully!")
            print(f"   Training windows: {len(all_train_windows)}")
            print(f"   Test windows: {len(all_test_windows)}")
            if all_train_windows:
                print(f"   Window shape: {all_train_windows[0].shape}")

        return (
            all_train_windows,
            all_train_classes,
            all_train_fold_info,
            all_test_windows,
            all_test_classes,
            all_test_fold_info,
        )

    except Exception as e:
        if verbose:
            print(f"❌ Error: {e}")
        return [], [], [], [], [], []


def validate_configuration(selected_classes, test_classes, verbose=True):
    """
    Validate classification configuration.

    Args:
        selected_classes: List of classes to analyze
        test_classes: Available test classes
        verbose: Whether to print validation info

    Returns:
        bool: True if configuration is valid
    """
    if verbose:
        print(f"Selected classes: {selected_classes}")

    # Check if selected classes exist
    test_classes_array = np.array(test_classes)
    unique_test_classes = np.unique(test_classes_array)
    available_selected = [cls for cls in selected_classes if cls in unique_test_classes]

    if len(available_selected) == len(selected_classes):
        if verbose:
            print(f"✅ All selected classes found in data")
        return True
    else:
        return False


def analyze_results_by_category(results, verbose=True):
    """
    Analyze results by algorithm category.

    Args:
        results: List of result dictionaries
        verbose: Whether to print analysis

    Returns:
        dict: Analysis by category with best algorithm and accuracy for each category
    """
    # Categorize models
    tree_models = [
        r for r in results if "Tree" in r["model_name"] or "Forest" in r["model_name"]
    ]
    svm_models = [r for r in results if "SVM" in r["model_name"]]
    nn_models = [r for r in results if "Neural Network" in r["model_name"]]

    # Create analysis with best performers
    analysis = {}

    if tree_models:
        best_tree = max(tree_models, key=lambda x: x["test_accuracy"])
        analysis["Tree-Based"] = {
            "best_algorithm": best_tree["model_name"],
            "best_accuracy": best_tree["test_accuracy"],
            "count": len(tree_models),
        }

    if svm_models:
        best_svm = max(svm_models, key=lambda x: x["test_accuracy"])
        analysis["Support Vector Machines"] = {
            "best_algorithm": best_svm["model_name"],
            "best_accuracy": best_svm["test_accuracy"],
            "count": len(svm_models),
        }

    if nn_models:
        best_nn = max(nn_models, key=lambda x: x["test_accuracy"])
        analysis["Neural Networks"] = {
            "best_algorithm": best_nn["model_name"],
            "best_accuracy": best_nn["test_accuracy"],
            "count": len(nn_models),
        }

    if verbose:
        print(f"🔧 Algorithm Analysis:")
        print(f"   • Tree-Based: {len(tree_models)} models")
        print(f"   • Support Vector Machines: {len(svm_models)} models")
        print(f"   • Neural Networks: {len(nn_models)} models")

        # Best performers by category
        if tree_models:
            best_tree = max(tree_models, key=lambda x: x["test_accuracy"])
            print(
                f"\n🌳 Best Tree: {best_tree['model_name']} ({best_tree['test_accuracy']:.3f})"
            )

        if svm_models:
            best_svm = max(svm_models, key=lambda x: x["test_accuracy"])
            print(
                f"⚡ Best SVM: {best_svm['model_name']} ({best_svm['test_accuracy']:.3f})"
            )

        if nn_models:
            best_nn = max(nn_models, key=lambda x: x["test_accuracy"])
            print(
                f"🧠 Best NN: {best_nn['model_name']} ({best_nn['test_accuracy']:.3f})"
            )

    return analysis


class SupervisedClassifier:
    """
    Comprehensive supervised classification toolkit for 3W dataset.
    """

    def __init__(self, random_state: int = 42, verbose: bool = True):
        """
        Initialize SupervisedClassifier.

        Args:
            random_state (int): Random state for reproducibility
            verbose (bool): Whether to print detailed information
        """
        self.random_state = random_state
        self.verbose = verbose
        self.results = []
        self.label_encoder = LabelEncoder()
        self.class_names = None

        # Set random seeds
        np.random.seed(random_state)

    def prepare_data(
        self,
        train_dfs: List[pd.DataFrame],
        train_classes: List[str],
        test_dfs: List[pd.DataFrame],
        test_classes: List[str],
        balance_classes: bool = True,
        balance_strategy: str = "combined",
        max_samples_per_class: int = 300,
        selected_classes: List = None,
        internal_verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for supervised classification.

        Args:
            train_dfs (List[pd.DataFrame]): Training window dataframes
            train_classes (List[str]): Training class labels
            test_dfs (List[pd.DataFrame]): Test window dataframes
            test_classes (List[str]): Test class labels
            balance_classes (bool): Whether to balance classes using augmentation
            balance_strategy (str): Strategy for class balancing ('combined', 'oversample', 'undersample')
            max_samples_per_class (int): Maximum samples per class after balancing
            selected_classes (List): List of classes to include (None = all classes)

        Returns:
            tuple: (X_train, y_train, X_test, y_test) - prepared arrays
        """
        if internal_verbose:
            print("Preparing Data for Supervised Classification")
            print("=" * 55)

        # Extract class labels from windows
        if internal_verbose:
            print("Processing class labels...", end=" ")

        def extract_window_classes(window_dfs, fallback_classes):
            """Extract class labels from windows or use fallback classes"""
            window_classes = []
            for i, window_df in enumerate(window_dfs):
                if "class" in window_df.columns:
                    # Get the last value of the class column
                    window_class = window_df["class"].iloc[-1]
                    window_classes.append(window_class)
                else:
                    window_classes.append(fallback_classes[i])
            return window_classes

        train_window_classes = extract_window_classes(train_dfs, train_classes)
        test_window_classes = extract_window_classes(test_dfs, test_classes)

        # Map transient classes to their base classes
        if self.verbose:
            print("Mapping transient classes to base classes...", end=" ")

        transient_mapping = {101: 1, 102: 2, 105: 5, 106: 6, 107: 7, 108: 8, 109: 9}

        def map_transient_classes(classes):
            """Map transient classes to their base classes"""
            mapped = []
            for cls in classes:
                if cls in transient_mapping:
                    mapped.append(transient_mapping[cls])
                else:
                    mapped.append(cls)
            return mapped

        train_mapped_classes = map_transient_classes(train_window_classes)
        test_mapped_classes = map_transient_classes(test_window_classes)

        # Apply class filtering after extraction and mapping
        if selected_classes is not None:
            if self.verbose:
                print(f"Filtering to selected classes {selected_classes}...", end=" ")

            # Filter training data
            filtered_train_dfs = []
            filtered_train_classes = []
            for i, cls in enumerate(train_mapped_classes):
                if cls in selected_classes:
                    filtered_train_dfs.append(train_dfs[i])
                    filtered_train_classes.append(cls)

            # Filter test data
            filtered_test_dfs = []
            filtered_test_classes = []
            for i, cls in enumerate(test_mapped_classes):
                if cls in selected_classes:
                    filtered_test_dfs.append(test_dfs[i])
                    filtered_test_classes.append(cls)

            # Update the data
            train_dfs = filtered_train_dfs
            train_mapped_classes = filtered_train_classes
            test_dfs = filtered_test_dfs
            test_mapped_classes = filtered_test_classes

            if self.verbose:
                print(f"✅ Train: {len(train_dfs)}, Test: {len(test_dfs)}")

        if internal_verbose:
            print(
                f"Training class distribution: {dict(zip(*np.unique(train_mapped_classes, return_counts=True)))}"
            )
            print(
                f"Test class distribution: {dict(zip(*np.unique(test_mapped_classes, return_counts=True)))}"
            )

        # Balance classes using data augmentation
        if balance_classes:
            if self.verbose:
                print(
                    f"Balancing classes using '{balance_strategy}' strategy...",
                    end=" ",
                )

            train_balanced_dfs, train_balanced_classes = quick_balance_classes(
                train_dfs,
                train_mapped_classes,
                strategy=balance_strategy,
                internal_verbose=internal_verbose,
            )

        else:
            train_balanced_dfs = train_dfs
            train_balanced_classes = train_mapped_classes

        # Limit samples per class if specified
        if max_samples_per_class is not None:
            if self.verbose:
                print(
                    f"Limiting to max {max_samples_per_class} samples per class...",
                    end=" ",
                )

            train_balanced_dfs, train_balanced_classes = self._limit_samples_per_class(
                train_balanced_dfs, train_balanced_classes, max_samples_per_class
            )
            test_limited_dfs, test_limited_classes = self._limit_samples_per_class(
                test_dfs, test_mapped_classes, max_samples_per_class // 3
            )

        else:
            test_limited_dfs = test_dfs
            test_limited_classes = test_mapped_classes

        if self.verbose:
            print(
                f"Final training distribution: {dict(zip(*np.unique(train_balanced_classes, return_counts=True)))}"
            )
            print(
                f"Final test distribution: {dict(zip(*np.unique(test_limited_classes, return_counts=True)))}"
            )

        # Flatten windows to feature vectors (data is already normalized)
        if self.verbose:
            print("Flattening windows to feature vectors...", end=" ")

        X_train = self._flatten_windows(train_balanced_dfs)
        X_test = self._flatten_windows(test_limited_dfs)
        y_train = np.array(train_balanced_classes)
        y_test = np.array(test_limited_classes)

        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        self.class_names = self.label_encoder.classes_

        return X_train, y_train_encoded, X_test, y_test_encoded

    def _limit_samples_per_class(
        self, dfs: List[pd.DataFrame], classes: List[str], max_per_class: int
    ) -> Tuple[List[pd.DataFrame], List[str]]:
        """Limit the number of samples per class."""
        selected_indices = []
        selected_classes = []

        unique_classes = np.unique(classes)
        for target_class in unique_classes:
            class_indices = [i for i, cls in enumerate(classes) if cls == target_class]

            if len(class_indices) > 0:
                n_samples = min(max_per_class, len(class_indices))
                sampled_indices = np.random.choice(
                    class_indices, size=n_samples, replace=False
                )
                selected_indices.extend(sampled_indices)
                selected_classes.extend([target_class] * len(sampled_indices))

        selected_dfs = [dfs[i] for i in selected_indices]
        return selected_dfs, selected_classes

    def _flatten_windows(self, window_dfs: List[pd.DataFrame]) -> np.ndarray:
        """Convert windowed time series to flattened feature vectors."""
        flattened_windows = []

        for window_df in window_dfs:
            # Exclude the 'class' column, keep all sensor data
            feature_columns = [col for col in window_df.columns if col != "class"]

            # Flatten the feature data
            flattened = window_df[feature_columns].values.flatten()
            flattened_windows.append(flattened)

        return np.array(flattened_windows)

    def train_decision_trees(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Train Decision Tree and Random Forest models.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data

        Returns:
            dict: Results dictionary with model performance
        """
        if self.verbose:
            print("Training Decision Trees and Random Forest")
            print("=" * 50)

        results = {}

        # 1. DECISION TREE
        if self.verbose:
            print("Training Decision Tree...")

        start_time = time.time()
        dt_classifier = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=self.random_state,
        )

        dt_classifier.fit(X_train, y_train)
        dt_train_time = time.time() - start_time

        # Make predictions
        dt_train_pred = dt_classifier.predict(X_train)
        dt_test_pred = dt_classifier.predict(X_test)

        # Calculate accuracies
        dt_train_acc = accuracy_score(y_train, dt_train_pred)
        dt_test_acc = accuracy_score(y_test, dt_test_pred)

        results["decision_tree"] = {
            "model": dt_classifier,
            "model_name": "Decision Tree",
            "train_accuracy": dt_train_acc,
            "test_accuracy": dt_test_acc,
            "training_time": dt_train_time,
            "predictions": dt_test_pred,
        }

        if self.verbose:
            print(f"Decision Tree trained in {dt_train_time:.3f}s")
            print(f"   • Training Accuracy: {dt_train_acc:.3f}")
            print(f"   • Test Accuracy: {dt_test_acc:.3f}")

        # 2. RANDOM FOREST
        if self.verbose:
            print("\nTraining Random Forest...")

        start_time = time.time()
        rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1,
        )

        rf_classifier.fit(X_train, y_train)
        rf_train_time = time.time() - start_time

        # Make predictions
        rf_train_pred = rf_classifier.predict(X_train)
        rf_test_pred = rf_classifier.predict(X_test)

        # Calculate accuracies
        rf_train_acc = accuracy_score(y_train, rf_train_pred)
        rf_test_acc = accuracy_score(y_test, rf_test_pred)

        results["random_forest"] = {
            "model": rf_classifier,
            "model_name": "Random Forest",
            "train_accuracy": rf_train_acc,
            "test_accuracy": rf_test_acc,
            "training_time": rf_train_time,
            "predictions": rf_test_pred,
            "feature_importance": rf_classifier.feature_importances_,
        }

        if self.verbose:
            print(f"Random Forest trained in {rf_train_time:.3f}s")
            print(f"   • Training Accuracy: {rf_train_acc:.3f}")
            print(f"   • Test Accuracy: {rf_test_acc:.3f}")

            # Show feature importance
            print(f"\nTop 10 Most Important Features:")
            feature_importance = rf_classifier.feature_importances_
            top_features_idx = np.argsort(feature_importance)[-10:][::-1]

            for i, idx in enumerate(top_features_idx, 1):
                print(f"   {i:2d}. Feature {idx:4d}: {feature_importance[idx]:.4f}")

        return results

    def train_svm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        max_samples: int = 1000,
    ) -> Dict[str, Any]:
        """
        Train Support Vector Machine models.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            max_samples: Maximum samples for SVM (computational efficiency)

        Returns:
            dict: Results dictionary with model performance
        """
        if self.verbose:
            print("⚡ Training Support Vector Machines")
            print("=" * 40)

        # Use subset for SVM efficiency
        n_train_svm = min(max_samples, X_train.shape[0])
        n_test_svm = min(max_samples // 2, X_test.shape[0])

        train_indices = np.random.choice(X_train.shape[0], n_train_svm, replace=False)
        test_indices = np.random.choice(X_test.shape[0], n_test_svm, replace=False)

        X_train_svm = X_train[train_indices]
        y_train_svm = y_train[train_indices]
        X_test_svm = X_test[test_indices]
        y_test_svm = y_test[test_indices]

        if self.verbose:
            print(
                f"Using subset: {X_train_svm.shape[0]} train, {X_test_svm.shape[0]} test"
            )

        results = {}

        # 1. LINEAR SVM
        if self.verbose:
            print(f"\nTraining Linear SVM...")

        start_time = time.time()
        linear_svm = SVC(kernel="linear", C=1.0, random_state=self.random_state)
        linear_svm.fit(X_train_svm, y_train_svm)
        linear_train_time = time.time() - start_time

        # Predictions
        linear_train_pred = linear_svm.predict(X_train_svm)
        linear_test_pred = linear_svm.predict(X_test_svm)

        linear_train_acc = accuracy_score(y_train_svm, linear_train_pred)
        linear_test_acc = accuracy_score(y_test_svm, linear_test_pred)

        results["linear_svm"] = {
            "model": linear_svm,
            "model_name": "Linear SVM",
            "train_accuracy": linear_train_acc,
            "test_accuracy": linear_test_acc,
            "training_time": linear_train_time,
            "predictions": linear_test_pred,
            "test_indices": test_indices,
        }

        if self.verbose:
            print(f"Linear SVM trained in {linear_train_time:.3f}s")
            print(f"   • Training Accuracy: {linear_train_acc:.3f}")
            print(f"   • Test Accuracy: {linear_test_acc:.3f}")

        # 2. RBF SVM
        if self.verbose:
            print(f"\nTraining RBF SVM...")

        start_time = time.time()
        rbf_svm = SVC(
            kernel="rbf", C=1.0, gamma="scale", random_state=self.random_state
        )
        rbf_svm.fit(X_train_svm, y_train_svm)
        rbf_train_time = time.time() - start_time

        # Predictions
        rbf_train_pred = rbf_svm.predict(X_train_svm)
        rbf_test_pred = rbf_svm.predict(X_test_svm)

        rbf_train_acc = accuracy_score(y_train_svm, rbf_train_pred)
        rbf_test_acc = accuracy_score(y_test_svm, rbf_test_pred)

        results["rbf_svm"] = {
            "model": rbf_svm,
            "model_name": "RBF SVM",
            "train_accuracy": rbf_train_acc,
            "test_accuracy": rbf_test_acc,
            "training_time": rbf_train_time,
            "predictions": rbf_test_pred,
            "test_indices": test_indices,
        }

        if self.verbose:
            print(f"RBF SVM trained in {rbf_train_time:.3f}s")
            print(f"   • Training Accuracy: {rbf_train_acc:.3f}")
            print(f"   • Test Accuracy: {rbf_test_acc:.3f}")

        return results

    def train_neural_networks(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Train Neural Network models.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data

        Returns:
            dict: Results dictionary with model performance
        """
        if self.verbose:
            print("Training Neural Networks")
            print("=" * 35)
            print(f"Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            print(f"Test: {X_test.shape[0]} samples")
            print(f"Classes: {len(np.unique(y_train))}")

        results = {}

        # 1. SIMPLE NEURAL NETWORK
        if self.verbose:
            print(f"\nTraining Simple Neural Network...")

        start_time = time.time()
        simple_nn = MLPClassifier(
            hidden_layer_sizes=(100,),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            max_iter=200,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )

        simple_nn.fit(X_train, y_train)
        simple_train_time = time.time() - start_time

        # Predictions
        simple_train_pred = simple_nn.predict(X_train)
        simple_test_pred = simple_nn.predict(X_test)

        simple_train_acc = accuracy_score(y_train, simple_train_pred)
        simple_test_acc = accuracy_score(y_test, simple_test_pred)

        results["simple_nn"] = {
            "model": simple_nn,
            "model_name": "Simple Neural Network",
            "train_accuracy": simple_train_acc,
            "test_accuracy": simple_test_acc,
            "training_time": simple_train_time,
            "predictions": simple_test_pred,
            "iterations": simple_nn.n_iter_,
        }

        if self.verbose:
            print(
                f"Simple NN trained in {simple_train_time:.3f}s ({simple_nn.n_iter_} iterations)"
            )
            print(f"   • Training Accuracy: {simple_train_acc:.3f}")
            print(f"   • Test Accuracy: {simple_test_acc:.3f}")

        # 2. DEEP NEURAL NETWORK
        if self.verbose:
            print(f"\nTraining Deep Neural Network...")

        start_time = time.time()
        deep_nn = MLPClassifier(
            hidden_layer_sizes=(200, 100, 50),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            max_iter=200,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )

        deep_nn.fit(X_train, y_train)
        deep_train_time = time.time() - start_time

        # Predictions
        deep_train_pred = deep_nn.predict(X_train)
        deep_test_pred = deep_nn.predict(X_test)

        deep_train_acc = accuracy_score(y_train, deep_train_pred)
        deep_test_acc = accuracy_score(y_test, deep_test_pred)

        results["deep_nn"] = {
            "model": deep_nn,
            "model_name": "Deep Neural Network",
            "train_accuracy": deep_train_acc,
            "test_accuracy": deep_test_acc,
            "training_time": deep_train_time,
            "predictions": deep_test_pred,
            "iterations": deep_nn.n_iter_,
        }

        if self.verbose:
            print(
                f"Deep NN trained in {deep_train_time:.3f}s ({deep_nn.n_iter_} iterations)"
            )
            print(f"   • Training Accuracy: {deep_train_acc:.3f}")
            print(f"   • Test Accuracy: {deep_test_acc:.3f}")

        # 3. REGULARIZED NEURAL NETWORK
        if self.verbose:
            print(f"\nTraining Regularized Neural Network...")

        start_time = time.time()
        regularized_nn = MLPClassifier(
            hidden_layer_sizes=(150, 100),
            activation="relu",
            solver="adam",
            alpha=0.001,  # Higher regularization
            learning_rate="adaptive",
            max_iter=300,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
        )

        regularized_nn.fit(X_train, y_train)
        reg_train_time = time.time() - start_time

        # Predictions
        reg_train_pred = regularized_nn.predict(X_train)
        reg_test_pred = regularized_nn.predict(X_test)

        reg_train_acc = accuracy_score(y_train, reg_train_pred)
        reg_test_acc = accuracy_score(y_test, reg_test_pred)

        results["regularized_nn"] = {
            "model": regularized_nn,
            "model_name": "Regularized Neural Network",
            "train_accuracy": reg_train_acc,
            "test_accuracy": reg_test_acc,
            "training_time": reg_train_time,
            "predictions": reg_test_pred,
            "iterations": regularized_nn.n_iter_,
        }

        if self.verbose:
            print(
                f"Regularized NN trained in {reg_train_time:.3f}s ({regularized_nn.n_iter_} iterations)"
            )
            print(f"   • Training Accuracy: {reg_train_acc:.3f}")
            print(f"   • Test Accuracy: {reg_test_acc:.3f}")

        return results

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Train all classification models and return results.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data

        Returns:
            list: List of all model results
        """
        if self.verbose:
            print("Training All Classification Models")
            print("=" * 50)

        all_results = []

        # Train Decision Trees
        tree_results = self.train_decision_trees(X_train, y_train, X_test, y_test)
        all_results.extend(list(tree_results.values()))

        # Train SVMs
        svm_results = self.train_svm(X_train, y_train, X_test, y_test)
        all_results.extend(list(svm_results.values()))

        # Train Neural Networks
        nn_results = self.train_neural_networks(X_train, y_train, X_test, y_test)
        all_results.extend(list(nn_results.values()))

        # Store results
        self.results = all_results

        return all_results

    def compare_models(self, results: List[Dict[str, Any]], y_test: np.ndarray) -> None:
        """
        Compare all trained models and display comprehensive analysis.

        Args:
            results: List of model results
            y_test: True test labels
        """
        if self.verbose:
            print("Final Model Comparison and Analysis")
            print("=" * 45)

        if not results:
            print(" No model results available.")
            return

        # Performance comparison table
        print("Complete Model Performance Comparison:")
        print("=" * 80)
        print(
            f"{'Model':<25} {'Train Acc':<12} {'Test Acc':<12} {'Train Time':<12} {'Overfitting':<12}"
        )
        print("-" * 80)

        for result in results:
            overfitting = result["train_accuracy"] - result["test_accuracy"]
            print(
                f"{result['model_name']:<25} {result['train_accuracy']:<12.3f} "
                f"{result['test_accuracy']:<12.3f} {result['training_time']:<12.3f} {overfitting:<12.3f}"
            )

        # Best model identification
        best_model = max(results, key=lambda x: x["test_accuracy"])
        fastest_model = min(results, key=lambda x: x["training_time"])
        least_overfitting = min(
            results, key=lambda x: abs(x["train_accuracy"] - x["test_accuracy"])
        )

        print(f"\nModel Rankings:")
        print(
            f"   Best Test Accuracy: {best_model['model_name']} ({best_model['test_accuracy']:.3f})"
        )
        print(
            f"   ⚡ Fastest Training: {fastest_model['model_name']} ({fastest_model['training_time']:.3f}s)"
        )
        print(
            f"   Least Overfitting: {least_overfitting['model_name']} "
            f"(gap: {abs(least_overfitting['train_accuracy'] - least_overfitting['test_accuracy']):.3f})"
        )

        # Best model identification completed
        pass


def enhanced_fold_analysis(
    train_dfs: List[pd.DataFrame],
    train_classes: List[str],
    test_dfs: List[pd.DataFrame],
    test_classes: List[str],
    train_fold_info: List[str] = None,
    test_fold_info: List[str] = None,
    balance_classes: bool = True,
    balance_strategy: str = "combined",
    max_samples_per_class: int = 300,
    balance_test: bool = False,
    min_test_samples_per_class: int = 300,
    selected_classes: List[str] = None,
    verbose: bool = False,
) -> SupervisedClassifier:
    """
    Enhanced supervised classification with proper cross-fold validation.

    This function performs true cross-fold validation by training models on specific folds
    and testing on other folds, providing per-fold accuracy analysis.

    Args:
        train_dfs, train_classes: Training data
        test_dfs, test_classes: Test data
        train_fold_info: List indicating which fold each training sample belongs to
        test_fold_info: List indicating which fold each test sample belongs to
        balance_classes: Whether to balance training classes
        balance_strategy: Strategy for balancing ('combined', 'oversample', 'undersample')
        max_samples_per_class: Max samples per class for training
        balance_test: Whether to balance test classes (usually False for fold analysis)
        min_test_samples_per_class: Minimum samples per class in test set
        selected_classes: List of classes to include in analysis (None = all classes except 0)
        verbose: Print details

    Returns:
        SupervisedClassifier: Trained classifier with enhanced fold-wise results
    """

    # Validate fold information
    if len(train_fold_info) != len(train_dfs):
        raise ValueError(
            f"train_fold_info length ({len(train_fold_info)}) != train_dfs length ({len(train_dfs)})"
        )

    if len(test_fold_info) != len(test_dfs):
        raise ValueError(
            f"test_fold_info length ({len(test_fold_info)}) != test_dfs length ({len(test_dfs)})"
        )

    # Get unique folds
    train_folds = sorted(set(train_fold_info))
    test_folds = sorted(set(test_fold_info))
    all_folds = sorted(set(train_folds + test_folds))

    # Filter data based on selected_classes parameter
    if selected_classes is None:
        filter_condition = lambda cls: cls != 0 and cls != "0"
    else:
        selected_classes_str = [str(c) for c in selected_classes]
        filter_condition = lambda cls: str(cls) in selected_classes_str

    # Filter training and test data based on class selection
    filtered_train_data = []
    filtered_test_data = []

    for i, (df, cls, fold) in enumerate(zip(train_dfs, train_classes, train_fold_info)):
        if filter_condition(cls):
            filtered_train_data.append((df, cls, fold, i))

    for i, (df, cls, fold) in enumerate(zip(test_dfs, test_classes, test_fold_info)):
        if filter_condition(cls):
            filtered_test_data.append((df, cls, fold, i))

    if len(filtered_train_data) == 0 or len(filtered_test_data) == 0:
        raise ValueError("No data remaining after class filtering")

    # Initialize classifier for consistent label encoding across folds
    classifier = SupervisedClassifier(verbose=verbose)

    # Prepare all data once to establish consistent label encoding
    all_train_classes = [item[1] for item in filtered_train_data]
    all_test_classes = [item[1] for item in filtered_test_data]

    # Create label encoder with all classes
    all_classes = list(set(all_train_classes + all_test_classes))
    classifier.label_encoder.fit(all_classes)
    classifier.class_names = classifier.label_encoder.classes_

    # Store results for each fold and model
    fold_results = {}
    model_names = [
        "Decision Tree",
        "Random Forest",
        "Linear SVM",
        "RBF SVM",
        "Simple Neural Network",
        "Deep Neural Network",
        "Regularized Neural Network",
    ]

    # Initialize fold results structure
    for fold in all_folds:
        fold_results[fold] = {}
        for model_name in model_names:
            fold_results[fold][model_name] = 0.0

    # Perform cross-fold validation
    for test_fold in all_folds:
        # Separate training and test data for this fold iteration
        fold_train_dfs = []
        fold_train_classes = []
        fold_test_dfs = []
        fold_test_classes = []

        # Collect training data (all folds except test_fold)
        for df, cls, fold, _ in filtered_train_data:
            if fold == test_fold:
                fold_train_dfs.append(df)
                fold_train_classes.append(cls)

        # Collect test data (only test_fold)
        for df, cls, fold, _ in filtered_test_data:
            if fold == test_fold:
                fold_test_dfs.append(df)
                fold_test_classes.append(cls)

        # Create a temporary classifier for this fold
        temp_classifier = SupervisedClassifier(verbose=False)
        temp_classifier.label_encoder = (
            classifier.label_encoder
        )  # Use consistent encoding
        temp_classifier.class_names = classifier.class_names

        # Prepare data for this fold
        try:
            X_train, y_train, X_test, y_test = temp_classifier.prepare_data(
                fold_train_dfs,
                fold_train_classes,
                fold_test_dfs,
                fold_test_classes,
                balance_classes=balance_classes,
                balance_strategy=balance_strategy,
                max_samples_per_class=max_samples_per_class,
                selected_classes=selected_classes,  # Pass selected_classes to prepare_data
                internal_verbose=False,
            )

            if verbose:
                print(
                    f"   Data prepared - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}"
                )

            # Train all models for this fold
            fold_model_results = temp_classifier.train_all_models(
                X_train, y_train, X_test, y_test
            )

            # Extract test accuracies for this fold
            for result in fold_model_results:
                model_name = result["model_name"]
                test_accuracy = result["test_accuracy"]
                fold_results[test_fold][model_name] = test_accuracy

                if verbose:
                    print(f"   {model_name}: {test_accuracy:.3f}")

        except Exception as e:
            if verbose:
                print(f"    Error processing fold {test_fold}: {str(e)}")
            continue

    # Prepare data for the last fold to return some results
    X_train, y_train, X_test, y_test = classifier.prepare_data(
        fold_train_dfs,
        fold_train_classes,
        fold_test_dfs,
        fold_test_classes,
        balance_classes=balance_classes,
        balance_strategy=balance_strategy,
        max_samples_per_class=max_samples_per_class,
        selected_classes=selected_classes,  # Pass selected_classes to prepare_data
        internal_verbose=True,
    )

    # Train all models for this fold
    last_fold_model_results = classifier.train_all_models(
        X_train, y_train, X_test, y_test
    )
    # Store fold results in classifier for analysis
    classifier.fold_results = fold_results
    classifier.fold_names = all_folds
    classifier.compare_models(last_fold_model_results, y_test)

    return classifier


# ============================================================
# 🧠 NEURAL NETWORK ARCHITECTURES VISUALIZATION
# ============================================================


def print_neural_network_architectures():
    """
    Display the neural network architectures used in the experiments.
    This function shows the structure, parameters, and characteristics of each NN model.
    """
    print("🧠 NEURAL NETWORK ARCHITECTURES USED IN EXPERIMENTS")
    print("=" * 65)

    print("\n📊 ARCHITECTURE OVERVIEW:")
    print("├─ Simple Neural Network: Single hidden layer (basic)")
    print("├─ Deep Neural Network: Three hidden layers (complex)")
    print("└─ Regularized Neural Network: Two layers + strong regularization")

    print("\n" + "=" * 65)
    print("🔹 1. SIMPLE NEURAL NETWORK")
    print("=" * 30)
    print("Architecture:")
    print("  Input Layer    → Hidden Layer → Output Layer")
    print("  [8 features]   →   [100]     →   [9 classes]")
    print()
    print("Parameters:")
    print("  • hidden_layer_sizes: (100,)")
    print("  • activation: 'relu'")
    print("  • solver: 'adam'")
    print("  • alpha (L2 penalty): 0.0001 (low regularization)")
    print("  • max_iter: 200")
    print("  • early_stopping: True")
    print()
    print("Characteristics:")
    print("  ✓ Fast training")
    print("  ✓ Good baseline performance")
    print("  ✓ Low computational cost")
    print("  ⚠ Limited complexity")

    print("\n" + "=" * 65)
    print("🔹 2. DEEP NEURAL NETWORK")
    print("=" * 25)
    print("Architecture:")
    print("  Input → Hidden 1 → Hidden 2 → Hidden 3 → Output")
    print("  [8]   →   [200]  →   [100]  →   [50]   →   [9]")
    print()
    print("Parameters:")
    print("  • hidden_layer_sizes: (200, 100, 50)")
    print("  • activation: 'relu'")
    print("  • solver: 'adam'")
    print("  • alpha (L2 penalty): 0.0001 (low regularization)")
    print("  • max_iter: 200")
    print("  • early_stopping: True")
    print()
    print("Characteristics:")
    print("  ✓ High learning capacity")
    print("  ✓ Can learn complex patterns")
    print("  ⚠ Prone to overfitting")
    print("  ⚠ Longer training time")

    print("\n" + "=" * 65)
    print("🔹 3. REGULARIZED NEURAL NETWORK")
    print("=" * 33)
    print("Architecture:")
    print("  Input Layer → Hidden 1 → Hidden 2 → Output Layer")
    print("  [8 features] →  [150]   →  [100]   →  [9 classes]")
    print()
    print("Parameters:")
    print("  • hidden_layer_sizes: (150, 100)")
    print("  • activation: 'relu'")
    print("  • solver: 'adam'")
    print("  • alpha (L2 penalty): 0.001 (HIGH regularization)")
    print("  • learning_rate: 'adaptive'")
    print("  • max_iter: 300")
    print("  • early_stopping: True")
    print("  • validation_fraction: 0.15 (larger validation set)")
    print()
    print("Characteristics:")
    print("  ✓ Best generalization")
    print("  ✓ Resistant to overfitting")
    print("  ✓ Adaptive learning rate")
    print("  ⚠ Slower convergence")

    print("\n" + "=" * 65)
    print("📈 ARCHITECTURE COMPARISON TABLE")
    print("=" * 35)

    # Create comparison table
    table_data = [
        ["Model", "Layers", "Neurons", "Regularization", "Complexity"],
        ["-" * 18, "-" * 8, "-" * 12, "-" * 14, "-" * 10],
        ["Simple NN", "1 hidden", "100", "Low (0.0001)", "Basic"],
        ["Deep NN", "3 hidden", "200+100+50", "Low (0.0001)", "High"],
        ["Regularized NN", "2 hidden", "150+100", "High (0.001)", "Medium"],
    ]

    for row in table_data:
        print(f"{row[0]:<18} {row[1]:<8} {row[2]:<12} {row[3]:<14} {row[4]:<10}")

    print("\n💡 ARCHITECTURE DESIGN PRINCIPLES:")
    print("   • Simple NN: Fast baseline with single hidden layer")
    print("   • Deep NN: Maximum expressiveness with depth")
    print("   • Regularized NN: Balanced complexity with overfitting control")
    print("   • All use ReLU activation for non-linearity")
    print("   • Adam optimizer for adaptive learning")
    print("   • Early stopping to prevent overfitting")

    print("\n🎯 EDUCATIONAL INSIGHTS:")
    print("   • More layers ≠ always better performance")
    print("   • Regularization is crucial for generalization")
    print("   • Architecture choice depends on data complexity")
    print("   • Validation performance guides model selection")


def tree_based_fold_analysis(
    train_dfs: List[pd.DataFrame],
    train_classes: List[str],
    test_dfs: List[pd.DataFrame],
    test_classes: List[str],
    train_fold_info: List[str],
    test_fold_info: List[str],
    selected_classes: List = None,
    balance_classes: bool = True,
    balance_strategy: str = "combined",
    max_samples_per_class: int = 1000,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Train Decision Tree and Random Forest models separately for each fold.
    
    Args:
        train_dfs: List of training dataframes
        train_classes: List of training class labels
        test_dfs: List of test dataframes
        test_classes: List of test class labels
        train_fold_info: List of fold identifiers for training data
        test_fold_info: List of fold identifiers for test data
        selected_classes: List of classes to include in analysis
        balance_classes: Whether to balance classes
        balance_strategy: Strategy for balancing ("combined", "oversample", "undersample")
        max_samples_per_class: Maximum samples per class after balancing
        verbose: Whether to print detailed information
    
    Returns:
        dict: Results containing fold_results, summary_df, and best_overall model
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import time
    import pandas as pd
    
    if verbose:
        print("🌳 TREE-BASED FOLD ANALYSIS")
        print("=" * 40)
    
    # Get unique folds
    unique_folds = sorted(set(test_fold_info))
    if verbose:
        print(f"📁 Processing {len(unique_folds)} folds: {unique_folds}")
    
    fold_tree_results = {}
    summary_data = []
    
    # Process each fold separately
    for fold_name in unique_folds:
        if verbose:
            print(f"\n📁 Processing Fold {fold_name}...")
        
        # Filter data for this fold
        train_fold_indices = [i for i, fold in enumerate(train_fold_info) if fold == fold_name]
        test_fold_indices = [i for i, fold in enumerate(test_fold_info) if fold == fold_name]
        
        fold_train_dfs = [train_dfs[i] for i in train_fold_indices]
        fold_train_classes = [train_classes[i] for i in train_fold_indices]
        fold_test_dfs = [test_dfs[i] for i in test_fold_indices]
        fold_test_classes = [test_classes[i] for i in test_fold_indices]
        
        # Initialize classifier for this fold
        fold_classifier = SupervisedClassifier(random_state=42, verbose=False)
        
        try:
            # Prepare data for this fold
            X_train_fold, y_train_fold, X_test_fold, y_test_fold = fold_classifier.prepare_data(
                train_dfs=fold_train_dfs,
                train_classes=fold_train_classes,
                test_dfs=fold_test_dfs,
                test_classes=fold_test_classes,
                balance_classes=balance_classes,
                balance_strategy=balance_strategy,
                max_samples_per_class=max_samples_per_class,
                selected_classes=selected_classes,
                internal_verbose=False
            )
            
            fold_results = []
            
            # Train Decision Tree
            start_time = time.time()
            dt_classifier = DecisionTreeClassifier(
                max_depth=10, min_samples_split=20, min_samples_leaf=10, random_state=42
            )
            dt_classifier.fit(X_train_fold, y_train_fold)
            dt_train_time = time.time() - start_time
            
            dt_train_acc = accuracy_score(y_train_fold, dt_classifier.predict(X_train_fold))
            dt_test_acc = accuracy_score(y_test_fold, dt_classifier.predict(X_test_fold))
            
            fold_results.append({
                'model_name': 'Decision Tree',
                'model': dt_classifier,
                'train_accuracy': dt_train_acc,
                'test_accuracy': dt_test_acc,
                'training_time': dt_train_time,
                'fold': fold_name
            })
            
            # Train Random Forest
            start_time = time.time()
            rf_classifier = RandomForestClassifier(
                n_estimators=100, max_depth=15, min_samples_split=10, 
                min_samples_leaf=5, random_state=42, n_jobs=-1
            )
            rf_classifier.fit(X_train_fold, y_train_fold)
            rf_train_time = time.time() - start_time
            
            rf_train_acc = accuracy_score(y_train_fold, rf_classifier.predict(X_train_fold))
            rf_test_acc = accuracy_score(y_test_fold, rf_classifier.predict(X_test_fold))
            
            fold_results.append({
                'model_name': 'Random Forest',
                'model': rf_classifier,
                'train_accuracy': rf_train_acc,
                'test_accuracy': rf_test_acc,
                'training_time': rf_train_time,
                'feature_importance': rf_classifier.feature_importances_,
                'fold': fold_name
            })
            
            # Store fold results
            fold_tree_results[fold_name] = fold_results
            
            # Add to summary data
            for result in fold_results:
                summary_data.append({
                    'Fold': fold_name,
                    'Model': result['model_name'],
                    'Train Acc': f"{result['train_accuracy']:.3f}",
                    'Test Acc': f"{result['test_accuracy']:.3f}",
                    'Overfitting': f"{result['train_accuracy'] - result['test_accuracy']:.3f}",
                    'Time (s)': f"{result['training_time']:.3f}"
                })
            
            if verbose:
                best_fold = max(fold_results, key=lambda x: x['test_accuracy'])
                print(f"   Best: {best_fold['model_name']} ({best_fold['test_accuracy']:.3f})")
                
        except Exception as e:
            if verbose:
                print(f"   ❌ Error: {e}")
            fold_tree_results[fold_name] = []
    
    # Create summary DataFrame
    df_summary = pd.DataFrame(summary_data) if summary_data else pd.DataFrame()
    
    # Find best overall model
    all_results = []
    for fold_results in fold_tree_results.values():
        all_results.extend(fold_results)
    
    best_overall = max(all_results, key=lambda x: x['test_accuracy']) if all_results else None
    
    # Print concise summary
    if verbose and not df_summary.empty:
        print(f"\n📋 FOLD COMPARISON SUMMARY:")
        
        # Decision Tree summary
        dt_results = [r for r in all_results if r['model_name'] == 'Decision Tree']
        if dt_results:
            dt_accs = [r['test_accuracy'] for r in dt_results]
            print(f"🌲 Decision Tree: {np.mean(dt_accs):.3f} avg (range: {min(dt_accs):.3f}-{max(dt_accs):.3f})")
        
        # Random Forest summary
        rf_results = [r for r in all_results if r['model_name'] == 'Random Forest']
        if rf_results:
            rf_accs = [r['test_accuracy'] for r in rf_results]
            print(f"🌳 Random Forest: {np.mean(rf_accs):.3f} avg (range: {min(rf_accs):.3f}-{max(rf_accs):.3f})")
        
        if best_overall:
            print(f"\n🏆 Best Overall: {best_overall['model_name']} from {best_overall['fold']} ({best_overall['test_accuracy']:.3f})")
    
    return {
        'fold_results': fold_tree_results,
        'summary_df': df_summary,
        'best_overall': best_overall,
        'all_results': all_results
    }


def svm_based_fold_analysis(
    train_dfs: List[pd.DataFrame],
    train_classes: List[str],
    test_dfs: List[pd.DataFrame],
    test_classes: List[str],
    train_fold_info: List[str],
    test_fold_info: List[str],
    selected_classes: List = None,
    balance_classes: bool = True,
    balance_strategy: str = "combined",
    max_samples_per_class: int = 1000,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Train Linear and RBF SVM models separately for each fold.
    
    Args:
        train_dfs: List of training dataframes
        train_classes: List of training class labels
        test_dfs: List of test dataframes
        test_classes: List of test class labels
        train_fold_info: List of fold identifiers for training data
        test_fold_info: List of fold identifiers for test data
        selected_classes: List of classes to include in analysis
        balance_classes: Whether to balance classes
        balance_strategy: Strategy for balancing ("combined", "oversample", "undersample")
        max_samples_per_class: Maximum samples per class after balancing
        verbose: Whether to print detailed information
    
    Returns:
        dict: Results containing fold_results, summary_df, and best_overall model
    """
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    import time
    import pandas as pd
    
    if verbose:
        print("⚡ SVM-BASED FOLD ANALYSIS")
        print("=" * 35)
    
    # Get unique folds
    unique_folds = sorted(set(test_fold_info))
    if verbose:
        print(f"📁 Processing {len(unique_folds)} folds: {unique_folds}")
    
    fold_svm_results = {}
    summary_data = []
    
    # Process each fold separately
    for fold_name in unique_folds:
        if verbose:
            print(f"\n📁 Processing Fold {fold_name}...")
        
        # Filter data for this fold
        train_fold_indices = [i for i, fold in enumerate(train_fold_info) if fold == fold_name]
        test_fold_indices = [i for i, fold in enumerate(test_fold_info) if fold == fold_name]
        
        fold_train_dfs = [train_dfs[i] for i in train_fold_indices]
        fold_train_classes = [train_classes[i] for i in train_fold_indices]
        fold_test_dfs = [test_dfs[i] for i in test_fold_indices]
        fold_test_classes = [test_classes[i] for i in test_fold_indices]
        
        # Initialize classifier for this fold
        fold_classifier = SupervisedClassifier(random_state=42, verbose=False)
        
        try:
            # Prepare data for this fold
            X_train_fold, y_train_fold, X_test_fold, y_test_fold = fold_classifier.prepare_data(
                train_dfs=fold_train_dfs,
                train_classes=fold_train_classes,
                test_dfs=fold_test_dfs,
                test_classes=fold_test_classes,
                balance_classes=balance_classes,
                balance_strategy=balance_strategy,
                max_samples_per_class=max_samples_per_class,
                selected_classes=selected_classes,
                internal_verbose=False
            )
            
            fold_results = []
            
            # Limit samples for SVM to avoid memory issues
            max_svm_samples = min(1000, X_train_fold.shape[0])
            if X_train_fold.shape[0] > max_svm_samples:
                # Random sampling for SVM
                sample_indices = np.random.choice(X_train_fold.shape[0], max_svm_samples, replace=False)
                X_train_svm = X_train_fold[sample_indices]
                y_train_svm = y_train_fold[sample_indices]
            else:
                X_train_svm = X_train_fold
                y_train_svm = y_train_fold
            
            # Train Linear SVM
            start_time = time.time()
            linear_svm = SVC(kernel='linear', C=1.0, random_state=42)
            linear_svm.fit(X_train_svm, y_train_svm)
            linear_train_time = time.time() - start_time
            
            linear_train_acc = accuracy_score(y_train_svm, linear_svm.predict(X_train_svm))
            linear_test_acc = accuracy_score(y_test_fold, linear_svm.predict(X_test_fold))
            
            fold_results.append({
                'model_name': 'Linear SVM',
                'model': linear_svm,
                'train_accuracy': linear_train_acc,
                'test_accuracy': linear_test_acc,
                'training_time': linear_train_time,
                'fold': fold_name,
                'train_samples': X_train_svm.shape[0]
            })
            
            # Train RBF SVM
            start_time = time.time()
            rbf_svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
            rbf_svm.fit(X_train_svm, y_train_svm)
            rbf_train_time = time.time() - start_time
            
            rbf_train_acc = accuracy_score(y_train_svm, rbf_svm.predict(X_train_svm))
            rbf_test_acc = accuracy_score(y_test_fold, rbf_svm.predict(X_test_fold))
            
            fold_results.append({
                'model_name': 'RBF SVM',
                'model': rbf_svm,
                'train_accuracy': rbf_train_acc,
                'test_accuracy': rbf_test_acc,
                'training_time': rbf_train_time,
                'fold': fold_name,
                'train_samples': X_train_svm.shape[0]
            })
            
            # Store fold results
            fold_svm_results[fold_name] = fold_results
            
            # Add to summary data
            for result in fold_results:
                summary_data.append({
                    'Fold': fold_name,
                    'Model': result['model_name'],
                    'Train Acc': f"{result['train_accuracy']:.3f}",
                    'Test Acc': f"{result['test_accuracy']:.3f}",
                    'Overfitting': f"{result['train_accuracy'] - result['test_accuracy']:.3f}",
                    'Time (s)': f"{result['training_time']:.3f}",
                    'Samples': result['train_samples']
                })
            
            if verbose:
                best_fold = max(fold_results, key=lambda x: x['test_accuracy'])
                print(f"   Best: {best_fold['model_name']} ({best_fold['test_accuracy']:.3f})")
                print(f"   Trained on {X_train_svm.shape[0]} samples")
                
        except Exception as e:
            if verbose:
                print(f"   ❌ Error: {e}")
            fold_svm_results[fold_name] = []
    
    # Create summary DataFrame
    df_summary = pd.DataFrame(summary_data) if summary_data else pd.DataFrame()
    
    # Find best overall model
    all_results = []
    for fold_results in fold_svm_results.values():
        all_results.extend(fold_results)
    
    best_overall = max(all_results, key=lambda x: x['test_accuracy']) if all_results else None
    
    # Print concise summary
    if verbose and not df_summary.empty:
        print(f"\n📋 SVM FOLD COMPARISON SUMMARY:")
        
        # Linear SVM summary
        linear_results = [r for r in all_results if r['model_name'] == 'Linear SVM']
        if linear_results:
            linear_accs = [r['test_accuracy'] for r in linear_results]
            linear_times = [r['training_time'] for r in linear_results]
            print(f"📐 Linear SVM: {np.mean(linear_accs):.3f} avg (range: {min(linear_accs):.3f}-{max(linear_accs):.3f}) | Avg time: {np.mean(linear_times):.3f}s")
        
        # RBF SVM summary
        rbf_results = [r for r in all_results if r['model_name'] == 'RBF SVM']
        if rbf_results:
            rbf_accs = [r['test_accuracy'] for r in rbf_results]
            rbf_times = [r['training_time'] for r in rbf_results]
            print(f"🔮 RBF SVM: {np.mean(rbf_accs):.3f} avg (range: {min(rbf_accs):.3f}-{max(rbf_accs):.3f}) | Avg time: {np.mean(rbf_times):.3f}s")
        
        if best_overall:
            print(f"\n🏆 Best Overall: {best_overall['model_name']} from {best_overall['fold']} ({best_overall['test_accuracy']:.3f})")
    
    return {
        'fold_results': fold_svm_results,
        'summary_df': df_summary,
        'best_overall': best_overall,
        'all_results': all_results
    }


def neural_network_based_fold_analysis(
    train_dfs: List[pd.DataFrame],
    train_classes: List[str],
    test_dfs: List[pd.DataFrame],
    test_classes: List[str],
    train_fold_info: List[str],
    test_fold_info: List[str],
    selected_classes: List = None,
    balance_classes: bool = True,
    balance_strategy: str = "combined",
    max_samples_per_class: int = 1000,
    verbose: bool = True
) -> dict:
    """
    Run neural network fold analysis for each fold separately.
    
    Args:
        train_dfs: Training dataframes
        train_classes: Training class labels
        test_dfs: Test dataframes
        test_classes: Test class labels
        train_fold_info: Training fold information
        test_fold_info: Test fold information
        selected_classes: Classes to include in analysis
        balance_classes: Whether to balance classes
        balance_strategy: Strategy for balancing ("combined", "oversample", "undersample")
        max_samples_per_class: Maximum samples per class after balancing
        verbose: Whether to print detailed information
    
    Returns:
        dict: Results containing fold_results, summary_df, and best_overall model
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
    import time
    import pandas as pd
    
    if verbose:
        print("🧠 NEURAL NETWORK-BASED FOLD ANALYSIS")
        print("=" * 45)
    
    # Get unique folds
    unique_folds = sorted(set(test_fold_info))
    if verbose:
        print(f"📁 Processing {len(unique_folds)} folds: {unique_folds}")
    
    fold_nn_results = {}
    summary_data = []
    
    # Process each fold separately
    for fold_name in unique_folds:
        if verbose:
            print(f"\n📁 Processing Fold {fold_name}...")
        
        # Filter data for this fold
        train_fold_indices = [i for i, fold in enumerate(train_fold_info) if fold == fold_name]
        test_fold_indices = [i for i, fold in enumerate(test_fold_info) if fold == fold_name]
        
        fold_train_dfs = [train_dfs[i] for i in train_fold_indices]
        fold_train_classes = [train_classes[i] for i in train_fold_indices]
        fold_test_dfs = [test_dfs[i] for i in test_fold_indices]
        fold_test_classes = [test_classes[i] for i in test_fold_indices]
        
        # Initialize classifier for this fold
        fold_classifier = SupervisedClassifier(random_state=42, verbose=False)
        
        try:
            # Prepare data for this fold
            X_train_fold, y_train_fold, X_test_fold, y_test_fold = fold_classifier.prepare_data(
                train_dfs=fold_train_dfs,
                train_classes=fold_train_classes,
                test_dfs=fold_test_dfs,
                test_classes=fold_test_classes,
                balance_classes=balance_classes,
                balance_strategy=balance_strategy,
                max_samples_per_class=max_samples_per_class,
                selected_classes=selected_classes,
                internal_verbose=False
            )
            
            # Limit training data for efficiency (Neural Networks can be slow)
            max_nn_samples = min(1000, X_train_fold.shape[0])
            if X_train_fold.shape[0] > max_nn_samples:
                from sklearn.utils import resample
                X_train_nn, y_train_nn = resample(
                    X_train_fold, y_train_fold, 
                    n_samples=max_nn_samples, 
                    random_state=42, 
                    stratify=y_train_fold
                )
            else:
                X_train_nn, y_train_nn = X_train_fold, y_train_fold
            
            fold_results = []
            
            # 1. Simple Neural Network
            if verbose:
                print("   Training Simple NN...", end=" ")
            
            start_time = time.time()
            simple_nn = MLPClassifier(
                hidden_layer_sizes=(100,),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
            )
            simple_nn.fit(X_train_nn, y_train_nn)
            simple_train_time = time.time() - start_time
            
            simple_train_acc = accuracy_score(y_train_nn, simple_nn.predict(X_train_nn))
            simple_test_acc = accuracy_score(y_test_fold, simple_nn.predict(X_test_fold))
            
            fold_results.append({
                'model_name': 'Simple Neural Network',
                'model': simple_nn,
                'train_accuracy': simple_train_acc,
                'test_accuracy': simple_test_acc,
                'training_time': simple_train_time,
                'fold': fold_name,
                'train_samples': X_train_nn.shape[0],
                'n_iterations': simple_nn.n_iter_
            })
            
            if verbose:
                print(f"Done ({simple_test_acc:.3f})")
            
            # 2. Deep Neural Network
            if verbose:
                print("   Training Deep NN...", end=" ")
            
            start_time = time.time()
            deep_nn = MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
            )
            deep_nn.fit(X_train_nn, y_train_nn)
            deep_train_time = time.time() - start_time
            
            deep_train_acc = accuracy_score(y_train_nn, deep_nn.predict(X_train_nn))
            deep_test_acc = accuracy_score(y_test_fold, deep_nn.predict(X_test_fold))
            
            fold_results.append({
                'model_name': 'Deep Neural Network',
                'model': deep_nn,
                'train_accuracy': deep_train_acc,
                'test_accuracy': deep_test_acc,
                'training_time': deep_train_time,
                'fold': fold_name,
                'train_samples': X_train_nn.shape[0],
                'n_iterations': deep_nn.n_iter_
            })
            
            if verbose:
                print(f"Done ({deep_test_acc:.3f})")
            
            # 3. Regularized Neural Network
            if verbose:
                print("   Training Regularized NN...", end=" ")
            
            start_time = time.time()
            reg_nn = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                alpha=0.01  # L2 regularization
            )
            reg_nn.fit(X_train_nn, y_train_nn)
            reg_train_time = time.time() - start_time
            
            reg_train_acc = accuracy_score(y_train_nn, reg_nn.predict(X_train_nn))
            reg_test_acc = accuracy_score(y_test_fold, reg_nn.predict(X_test_fold))
            
            fold_results.append({
                'model_name': 'Regularized Neural Network',
                'model': reg_nn,
                'train_accuracy': reg_train_acc,
                'test_accuracy': reg_test_acc,
                'training_time': reg_train_time,
                'fold': fold_name,
                'train_samples': X_train_nn.shape[0],
                'n_iterations': reg_nn.n_iter_
            })
            
            if verbose:
                print(f"Done ({reg_test_acc:.3f})")
            
            # Store fold results
            fold_nn_results[fold_name] = fold_results
            
            # Add to summary data
            for result in fold_results:
                summary_data.append({
                    'Fold': fold_name,
                    'Model': result['model_name'],
                    'Train Acc': f"{result['train_accuracy']:.3f}",
                    'Test Acc': f"{result['test_accuracy']:.3f}",
                    'Overfitting': f"{result['train_accuracy'] - result['test_accuracy']:.3f}",
                    'Time (s)': f"{result['training_time']:.3f}",
                    'Samples': result['train_samples'],
                    'Iterations': result['n_iterations']
                })
            
            if verbose:
                best_fold = max(fold_results, key=lambda x: x['test_accuracy'])
                print(f"   Best: {best_fold['model_name']} ({best_fold['test_accuracy']:.3f})")
                print(f"   Trained on {X_train_nn.shape[0]} samples")
                
        except Exception as e:
            if verbose:
                print(f"   ❌ Error: {e}")
            fold_nn_results[fold_name] = []
    
    # Create summary DataFrame
    df_summary = pd.DataFrame(summary_data) if summary_data else pd.DataFrame()
    
    # Find best overall model
    all_results = []
    for fold_results in fold_nn_results.values():
        all_results.extend(fold_results)
    
    best_overall = max(all_results, key=lambda x: x['test_accuracy']) if all_results else None
    
    # Print concise summary
    if verbose and not df_summary.empty:
        print(f"\n📋 NEURAL NETWORK FOLD COMPARISON SUMMARY:")
        
        # Simple NN summary
        simple_results = [r for r in all_results if r['model_name'] == 'Simple Neural Network']
        if simple_results:
            simple_accs = [r['test_accuracy'] for r in simple_results]
            simple_times = [r['training_time'] for r in simple_results]
            simple_iters = [r['n_iterations'] for r in simple_results]
            print(f"🧠 Simple NN: {np.mean(simple_accs):.3f} avg (range: {min(simple_accs):.3f}-{max(simple_accs):.3f}) | Avg time: {np.mean(simple_times):.3f}s | Avg iters: {int(np.mean(simple_iters))}")
        
        # Deep NN summary
        deep_results = [r for r in all_results if r['model_name'] == 'Deep Neural Network']
        if deep_results:
            deep_accs = [r['test_accuracy'] for r in deep_results]
            deep_times = [r['training_time'] for r in deep_results]
            deep_iters = [r['n_iterations'] for r in deep_results]
            print(f"🔗 Deep NN: {np.mean(deep_accs):.3f} avg (range: {min(deep_accs):.3f}-{max(deep_accs):.3f}) | Avg time: {np.mean(deep_times):.3f}s | Avg iters: {int(np.mean(deep_iters))}")
        
        # Regularized NN summary
        reg_results = [r for r in all_results if r['model_name'] == 'Regularized Neural Network']
        if reg_results:
            reg_accs = [r['test_accuracy'] for r in reg_results]
            reg_times = [r['training_time'] for r in reg_results]
            reg_iters = [r['n_iterations'] for r in reg_results]
            print(f"⚙️ Regularized NN: {np.mean(reg_accs):.3f} avg (range: {min(reg_accs):.3f}-{max(reg_accs):.3f}) | Avg time: {np.mean(reg_times):.3f}s | Avg iters: {int(np.mean(reg_iters))}")
        
        if best_overall:
            print(f"\n🏆 Best Overall: {best_overall['model_name']} from {best_overall['fold']} ({best_overall['test_accuracy']:.3f})")
    
    return {
        'fold_results': fold_nn_results,
        'summary_df': df_summary,
        'best_overall': best_overall,
        'all_results': all_results
    }


def print_tree_analysis_results(analysis_results: dict, selected_classes: List = None) -> None:
    """
    Print detailed analysis results for tree-based algorithms.
    
    Args:
        analysis_results: Results from tree_based_fold_analysis
        selected_classes: Classes that were analyzed
    """
    import numpy as np
    
    fold_tree_results = analysis_results['fold_results']
    df_summary = analysis_results['summary_df']
    best_overall = analysis_results['best_overall']
    all_results = analysis_results['all_results']
    
    if df_summary.empty:
        print("❌ No results to analyze.")
        return
    
    # Show basic progress info
    unique_folds = sorted(fold_tree_results.keys())
    print(f"📁 Processed {len(unique_folds)} folds: {unique_folds}")
    
    if best_overall:
        print(f"🏆 Best Overall: {best_overall['model_name']} from {best_overall['fold']} ({best_overall['test_accuracy']:.3f})")
    
    print(f"\n📋 DETAILED RESULTS TABLE:")
    print("-" * 60)
    print(df_summary.to_string(index=False))
    
    # Fold comparison analysis
    print(f"\n📊 DETAILED FOLD COMPARISON:")
    print("-" * 40)
    
    # Decision Tree analysis
    dt_results = [(fold, next((r for r in results if r['model_name'] == 'Decision Tree'), None)) 
                 for fold, results in fold_tree_results.items()]
    dt_results = [(fold, result) for fold, result in dt_results if result is not None]
    
    if dt_results:
        print(f"\n🌲 Decision Tree by Fold:")
        for fold, result in dt_results:
            overfitting = result['train_accuracy'] - result['test_accuracy']
            print(f"   {fold}: {result['test_accuracy']:.3f} (overfitting: {overfitting:.3f})")
    
    # Random Forest analysis
    rf_results = [(fold, next((r for r in results if r['model_name'] == 'Random Forest'), None)) 
                 for fold, results in fold_tree_results.items()]
    rf_results = [(fold, result) for fold, result in rf_results if result is not None]
    
    if rf_results:
        print(f"\n🌳 Random Forest by Fold:")
        for fold, result in rf_results:
            overfitting = result['train_accuracy'] - result['test_accuracy']
            print(f"   {fold}: {result['test_accuracy']:.3f} (overfitting: {overfitting:.3f})")
    
    # Statistical analysis
    if dt_results and rf_results:
        dt_accuracies = [result['test_accuracy'] for _, result in dt_results]
        rf_accuracies = [result['test_accuracy'] for _, result in rf_results]
        
        print(f"\n📈 STATISTICAL COMPARISON:")
        print(f"   Decision Tree  - Avg: {sum(dt_accuracies)/len(dt_accuracies):.3f} | Range: {max(dt_accuracies)-min(dt_accuracies):.3f}")
        print(f"   Random Forest  - Avg: {sum(rf_accuracies)/len(rf_accuracies):.3f} | Range: {max(rf_accuracies)-min(rf_accuracies):.3f}")
        print(f"   Forest Advantage: +{(sum(rf_accuracies)/len(rf_accuracies)) - (sum(dt_accuracies)/len(dt_accuracies)):.3f}")
    
    # Feature importance from best Random Forest
    if best_overall and best_overall['model_name'] == 'Random Forest' and 'feature_importance' in best_overall:
        print(f"\n🔍 TOP 10 FEATURES - Best Random Forest ({best_overall['fold']}):")
        feature_importance = best_overall['feature_importance']
        top_features_idx = np.argsort(feature_importance)[-10:][::-1]
        
        for i, idx in enumerate(top_features_idx, 1):
            print(f"   {i:2d}. Feature {idx:4d}: {feature_importance[idx]:.4f}")
    
    print(f"\n🎓 KEY INSIGHTS:")
    print(f"   • Each fold represents different wells/conditions")
    print(f"   • Random Forest typically outperforms Decision Tree")
    print(f"   • Consistent performance across folds = robust algorithm")
    print(f"   • High variation = algorithm sensitive to data distribution")
    print(f"   • Classes analyzed: {selected_classes}")
    
    print(f"\n✅ Tree-based fold analysis complete!")


def print_svm_analysis_results(analysis_results: dict, selected_classes: List = None) -> None:
    """
    Print detailed analysis results for SVM algorithms.
    
    Args:
        analysis_results: Results from svm_based_fold_analysis
        selected_classes: Classes that were analyzed
    """
    import numpy as np
    
    fold_svm_results = analysis_results['fold_results']
    df_summary = analysis_results['summary_df']
    best_overall = analysis_results['best_overall']
    all_results = analysis_results['all_results']
    
    if df_summary.empty:
        print("❌ No results to analyze.")
        return
    
    # Show basic progress info
    unique_folds = sorted(fold_svm_results.keys())
    print(f"📁 Processed {len(unique_folds)} folds: {unique_folds}")
    
    if best_overall:
        print(f"🏆 Best Overall: {best_overall['model_name']} from {best_overall['fold']} ({best_overall['test_accuracy']:.3f})")
        if all_results:
            print(f"⚙️ Training limited to {all_results[0]['train_samples']} samples per fold for efficiency")
    
    print(f"\n📋 DETAILED RESULTS TABLE:")
    print("-" * 70)
    print(df_summary.to_string(index=False))
    
    # Fold comparison analysis
    print(f"\n📊 DETAILED FOLD COMPARISON:")
    print("-" * 40)
    
    # Linear SVM analysis
    linear_results = [(fold, next((r for r in results if r['model_name'] == 'Linear SVM'), None)) 
                     for fold, results in fold_svm_results.items()]
    linear_results = [(fold, result) for fold, result in linear_results if result is not None]
    
    if linear_results:
        print(f"\n📐 Linear SVM by Fold:")
        for fold, result in linear_results:
            overfitting = result['train_accuracy'] - result['test_accuracy']
            print(f"   {fold}: {result['test_accuracy']:.3f} (overfitting: {overfitting:.3f}, time: {result['training_time']:.3f}s)")
    
    # RBF SVM analysis
    rbf_results = [(fold, next((r for r in results if r['model_name'] == 'RBF SVM'), None)) 
                  for fold, results in fold_svm_results.items()]
    rbf_results = [(fold, result) for fold, result in rbf_results if result is not None]
    
    if rbf_results:
        print(f"\n🔮 RBF SVM by Fold:")
        for fold, result in rbf_results:
            overfitting = result['train_accuracy'] - result['test_accuracy']
            print(f"   {fold}: {result['test_accuracy']:.3f} (overfitting: {overfitting:.3f}, time: {result['training_time']:.3f}s)")
    
    # Statistical analysis
    if linear_results and rbf_results:
        linear_accuracies = [result['test_accuracy'] for _, result in linear_results]
        rbf_accuracies = [result['test_accuracy'] for _, result in rbf_results]
        linear_times = [result['training_time'] for _, result in linear_results]
        rbf_times = [result['training_time'] for _, result in rbf_results]
        
        print(f"\n📈 STATISTICAL COMPARISON:")
        print(f"   Linear SVM  - Avg: {sum(linear_accuracies)/len(linear_accuracies):.3f} | Range: {max(linear_accuracies)-min(linear_accuracies):.3f} | Avg Time: {sum(linear_times)/len(linear_times):.3f}s")
        print(f"   RBF SVM     - Avg: {sum(rbf_accuracies)/len(rbf_accuracies):.3f} | Range: {max(rbf_accuracies)-min(rbf_accuracies):.3f} | Avg Time: {sum(rbf_times)/len(rbf_times):.3f}s")
        print(f"   RBF Advantage: +{(sum(rbf_accuracies)/len(rbf_accuracies)) - (sum(linear_accuracies)/len(linear_accuracies)):.3f}")
        print(f"   Speed Ratio: Linear is {(sum(rbf_times)/len(rbf_times))/(sum(linear_times)/len(linear_times)):.1f}x faster than RBF")
    
    # Training efficiency analysis
    if all_results:
        print(f"\n⚡ TRAINING EFFICIENCY ANALYSIS:")
        linear_efficiency = [(r['test_accuracy'] / r['training_time']) for r in all_results if r['model_name'] == 'Linear SVM']
        rbf_efficiency = [(r['test_accuracy'] / r['training_time']) for r in all_results if r['model_name'] == 'RBF SVM']
        
        if linear_efficiency:
            print(f"   📐 Linear SVM Efficiency: {sum(linear_efficiency)/len(linear_efficiency):.1f} acc/sec")
        if rbf_efficiency:
            print(f"   🔮 RBF SVM Efficiency: {sum(rbf_efficiency)/len(rbf_efficiency):.1f} acc/sec")
    
    print(f"\n🎓 KEY INSIGHTS:")
    print(f"   • Linear SVM: Fast training, good for linearly separable data")
    print(f"   • RBF SVM: Slower training, handles complex non-linear patterns")
    print(f"   • Consistent performance across folds = robust algorithm")
    print(f"   • Classes analyzed: {selected_classes}")
    
    print(f"\n💡 SVM PERFORMANCE NOTES:")
    print(f"   • RBF kernel captures non-linear sensor relationships")
    print(f"   • Linear SVM suitable for quick baseline models")
    print(f"   • Training time scales with data complexity")
    
    print(f"\n✅ SVM-based fold analysis complete!")


def print_neural_network_analysis_results(analysis_results: dict, selected_classes: List = None) -> None:
    """
    Print detailed analysis results for neural network algorithms.
    
    Args:
        analysis_results: Results from neural_network_based_fold_analysis
        selected_classes: Classes that were analyzed
    """
    import numpy as np
    
    fold_nn_results = analysis_results['fold_results']
    df_summary = analysis_results['summary_df']
    best_overall = analysis_results['best_overall']
    all_results = analysis_results['all_results']
    
    if df_summary.empty:
        print("❌ No results to analyze.")
        return
    
    # Show basic progress info
    unique_folds = sorted(fold_nn_results.keys())
    print(f"📁 Processed {len(unique_folds)} folds: {unique_folds}")
    
    if best_overall:
        print(f"🏆 Best Overall: {best_overall['model_name']} from {best_overall['fold']} ({best_overall['test_accuracy']:.3f})")
        if all_results:
            print(f"⚙️ Training limited to {all_results[0]['train_samples']} samples per fold for efficiency")
            avg_iterations = sum(r['n_iterations'] for r in all_results) / len(all_results)
            print(f"🔄 Average training iterations: {int(avg_iterations)}")
    
    print(f"\n📋 DETAILED RESULTS TABLE:")
    print("-" * 80)
    print(df_summary.to_string(index=False))
    
    # Fold comparison analysis
    print(f"\n📊 DETAILED FOLD COMPARISON:")
    print("-" * 40)
    
    # Simple Neural Network analysis
    simple_results = [(fold, next((r for r in results if r['model_name'] == 'Simple Neural Network'), None)) 
                     for fold, results in fold_nn_results.items()]
    simple_results = [(fold, result) for fold, result in simple_results if result is not None]
    
    if simple_results:
        print(f"\n🧠 Simple Neural Network by Fold:")
        for fold, result in simple_results:
            overfitting = result['train_accuracy'] - result['test_accuracy']
            print(f"   {fold}: {result['test_accuracy']:.3f} (overfitting: {overfitting:.3f}, time: {result['training_time']:.3f}s, iters: {result['n_iterations']})")
    
    # Deep Neural Network analysis
    deep_results = [(fold, next((r for r in results if r['model_name'] == 'Deep Neural Network'), None)) 
                   for fold, results in fold_nn_results.items()]
    deep_results = [(fold, result) for fold, result in deep_results if result is not None]
    
    if deep_results:
        print(f"\n🔗 Deep Neural Network by Fold:")
        for fold, result in deep_results:
            overfitting = result['train_accuracy'] - result['test_accuracy']
            print(f"   {fold}: {result['test_accuracy']:.3f} (overfitting: {overfitting:.3f}, time: {result['training_time']:.3f}s, iters: {result['n_iterations']})")
    
    # Regularized Neural Network analysis
    reg_results = [(fold, next((r for r in results if r['model_name'] == 'Regularized Neural Network'), None)) 
                  for fold, results in fold_nn_results.items()]
    reg_results = [(fold, result) for fold, result in reg_results if result is not None]
    
    if reg_results:
        print(f"\n⚙️ Regularized Neural Network by Fold:")
        for fold, result in reg_results:
            overfitting = result['train_accuracy'] - result['test_accuracy']
            print(f"   {fold}: {result['test_accuracy']:.3f} (overfitting: {overfitting:.3f}, time: {result['training_time']:.3f}s, iters: {result['n_iterations']})")
    
    # Statistical analysis
    if simple_results and deep_results and reg_results:
        simple_accuracies = [result['test_accuracy'] for _, result in simple_results]
        deep_accuracies = [result['test_accuracy'] for _, result in deep_results]
        reg_accuracies = [result['test_accuracy'] for _, result in reg_results]
        simple_times = [result['training_time'] for _, result in simple_results]
        deep_times = [result['training_time'] for _, result in deep_results]
        reg_times = [result['training_time'] for _, result in reg_results]
        
        print(f"\n📈 STATISTICAL COMPARISON:")
        print(f"   Simple NN     - Avg: {sum(simple_accuracies)/len(simple_accuracies):.3f} | Range: {max(simple_accuracies)-min(simple_accuracies):.3f} | Avg Time: {sum(simple_times)/len(simple_times):.3f}s")
        print(f"   Deep NN       - Avg: {sum(deep_accuracies)/len(deep_accuracies):.3f} | Range: {max(deep_accuracies)-min(deep_accuracies):.3f} | Avg Time: {sum(deep_times)/len(deep_times):.3f}s")
        print(f"   Regularized NN - Avg: {sum(reg_accuracies)/len(reg_accuracies):.3f} | Range: {max(reg_accuracies)-min(reg_accuracies):.3f} | Avg Time: {sum(reg_times)/len(reg_times):.3f}s")
        
        best_avg = max([
            (sum(simple_accuracies)/len(simple_accuracies), "Simple NN"),
            (sum(deep_accuracies)/len(deep_accuracies), "Deep NN"),
            (sum(reg_accuracies)/len(reg_accuracies), "Regularized NN")
        ])
        print(f"   Best Architecture: {best_avg[1]} ({best_avg[0]:.3f})")
    
    # Training convergence analysis
    if all_results:
        print(f"\n🔄 TRAINING CONVERGENCE ANALYSIS:")
        simple_iters = [r['n_iterations'] for r in all_results if r['model_name'] == 'Simple Neural Network']
        deep_iters = [r['n_iterations'] for r in all_results if r['model_name'] == 'Deep Neural Network']
        reg_iters = [r['n_iterations'] for r in all_results if r['model_name'] == 'Regularized Neural Network']
        
        if simple_iters:
            print(f"   🧠 Simple NN Convergence: {sum(simple_iters)/len(simple_iters):.1f} avg iterations")
        if deep_iters:
            print(f"   🔗 Deep NN Convergence: {sum(deep_iters)/len(deep_iters):.1f} avg iterations")
        if reg_iters:
            print(f"   ⚙️ Regularized NN Convergence: {sum(reg_iters)/len(reg_iters):.1f} avg iterations")
    
    print(f"\n🎓 KEY INSIGHTS:")
    print(f"   • Simple NN: Fast training, good baseline performance")
    print(f"   • Deep NN: More complex patterns, risk of overfitting")
    print(f"   • Regularized NN: Balanced approach with L2 regularization")
    print(f"   • Early stopping prevents overfitting by monitoring validation loss")
    print(f"   • Classes analyzed: {selected_classes}")
    
    print(f"\n💡 NEURAL NETWORK PERFORMANCE NOTES:")
    print(f"   • Training limited to 1000 samples per fold for efficiency")
    print(f"   • Early stopping used to prevent overfitting")
    print(f"   • Convergence measured by iteration count")
    print(f"   • MLPClassifier with different architectures")
    
    print(f"\n✅ Neural network-based fold analysis complete!")


# ============================================================
# CLASS DISTRIBUTION ANALYSIS FUNCTIONS
# ============================================================

def analyze_class_distribution_by_fold(
    test_classes: List,
    test_fold_info: List,
    selected_classes: Optional[List] = None
) -> Dict:
    """
    Analyze class distribution across test folds.
    
    Args:
        test_classes: List of test class labels (can be int or str)
        test_fold_info: List of fold information for each test sample
        selected_classes: Optional list of selected classes to highlight
        
    Returns:
        Dictionary containing fold distribution analysis
    """
    from collections import Counter
    
    # Normalize classes to handle both int and str representations
    def normalize_class(cls):
        """Convert class to consistent format, handling both int and str"""
        if isinstance(cls, str):
            try:
                return int(cls)
            except ValueError:
                return cls
        return cls
    
    # Normalize test classes and selected classes
    normalized_test_classes = [normalize_class(cls) for cls in test_classes]
    normalized_selected_classes = None
    if selected_classes:
        normalized_selected_classes = [normalize_class(cls) for cls in selected_classes]
    
    # Get unique folds
    unique_folds = sorted(set(test_fold_info))
    
    # Group test data by fold
    fold_class_distribution = {}
    for fold in unique_folds:
        # Get indices for this fold
        fold_indices = [i for i, f in enumerate(test_fold_info) if f == fold]
        
        # Get classes for this fold
        fold_classes = [normalized_test_classes[i] for i in fold_indices]
        
        # Count class distribution
        class_counts = Counter(fold_classes)
        fold_class_distribution[fold] = class_counts
    
    # Get all unique classes across all folds
    all_classes = sorted(set(cls for fold_counts in fold_class_distribution.values() 
                           for cls in fold_counts.keys()))
    
    # Calculate class totals
    class_totals = {}
    grand_total = 0
    for cls in all_classes:
        total_for_class = sum(fold_class_distribution[fold].get(cls, 0) for fold in unique_folds)
        class_totals[cls] = total_for_class
        grand_total += total_for_class
    
    return {
        'fold_class_distribution': fold_class_distribution,
        'unique_folds': unique_folds,
        'all_classes': all_classes,
        'class_totals': class_totals,
        'grand_total': grand_total,
        'selected_classes': normalized_selected_classes
    }


def print_class_distribution_analysis(
    test_classes: List,
    test_fold_info: List,
    selected_classes: Optional[List] = None,
    verbose: bool = True
) -> None:
    """
    Print comprehensive class distribution analysis across test folds.
    
    Args:
        test_classes: List of test class labels (can be int or str)
        test_fold_info: List of fold information for each test sample
        selected_classes: Optional list of selected classes to highlight (can be int or str)
        verbose: Whether to print detailed output
    """
    if not verbose:
        return
        
    # Analyze distribution
    analysis = analyze_class_distribution_by_fold(test_classes, test_fold_info, selected_classes)
    
    fold_class_distribution = analysis['fold_class_distribution']
    unique_folds = analysis['unique_folds']
    all_classes = analysis['all_classes']
    class_totals = analysis['class_totals']
    grand_total = analysis['grand_total']
    normalized_selected_classes = analysis['selected_classes']
    
    print(f"\n📊 CLASS DISTRIBUTION BY TEST FOLD")
    print("=" * 45)
    
    # ============================================================
    # PER-FOLD ANALYSIS
    # ============================================================
    for fold in unique_folds:
        class_counts = fold_class_distribution[fold]
        fold_total = sum(class_counts.values())
        
        print(f"\n🗂️  {fold}:")
        print(f"   Total samples: {fold_total:,}")
        
        # Show distribution for all classes present
        all_classes_in_fold = sorted(class_counts.keys())
        for cls in all_classes_in_fold:
            count = class_counts[cls]
            percentage = (count / fold_total) * 100
            print(f"   Class {cls}: {count:,} samples ({percentage:.1f}%)")
        
        # Highlight selected classes if they exist in this fold
        if normalized_selected_classes:
            selected_in_fold = [cls for cls in normalized_selected_classes if cls in class_counts]
            if selected_in_fold:
                print(f"   Selected classes in fold: {selected_in_fold}")
                selected_total = sum(class_counts[cls] for cls in selected_in_fold)
                selected_percentage = (selected_total / fold_total) * 100
                print(f"   Selected classes total: {selected_total:,} samples ({selected_percentage:.1f}%)")
    
    # ============================================================
    # CROSS-FOLD SUMMARY TABLE
    # ============================================================
    print(f"\n📈 CROSS-FOLD SUMMARY")
    print("=" * 25)
    
    # Create summary table header
    print(f"\n{'Class':<8}", end="")
    for fold in unique_folds:
        print(f"{fold:<12}", end="")
    print("Total")
    print("-" * (8 + 12 * len(unique_folds) + 12))
    
    # Print class distribution table
    for cls in all_classes:
        print(f"{cls:<8}", end="")
        total_for_class = 0
        for fold in unique_folds:
            count = fold_class_distribution[fold].get(cls, 0)
            print(f"{count:<12,}", end="")
            total_for_class += count
        print(f"{total_for_class:<12,}")
    
    # Show totals by fold
    print("-" * (8 + 12 * len(unique_folds) + 12))
    print(f"{'Total':<8}", end="")
    for fold in unique_folds:
        fold_total = sum(fold_class_distribution[fold].values())
        print(f"{fold_total:<12,}", end="")
    print(f"{grand_total:<12,}")
    
    # ============================================================
    # SELECTED CLASSES SUMMARY
    # ============================================================
    if normalized_selected_classes:
        print(f"\n🎯 SELECTED CLASSES SUMMARY")
        print("=" * 30)
        for cls in normalized_selected_classes:
            if cls in class_totals:
                total = class_totals[cls]
                percentage = (total / grand_total) * 100
                print(f"Class {cls}: {total:,} samples ({percentage:.1f}% of total)")
                
                # Show distribution across folds
                fold_distribution = []
                for fold in unique_folds:
                    count = fold_class_distribution[fold].get(cls, 0)
                    if count > 0:
                        fold_distribution.append(f"{fold}: {count:,}")
                if fold_distribution:
                    print(f"          Folds: {', '.join(fold_distribution)}")
            else:
                print(f"Class {cls}: ❌ Not found in test data")
    
    print(f"\n✅ Class distribution analysis complete!")

