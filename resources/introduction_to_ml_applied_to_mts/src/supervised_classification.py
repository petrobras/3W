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
        fold_dirs = [d for d in os.listdir(windowed_dir) 
                    if d.startswith("fold_") and os.path.isdir(os.path.join(windowed_dir, d))]
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
                fold_train_dfs, fold_train_classes = persistence._load_dataframes(train_file, config.SAVE_FORMAT)
                all_train_windows.extend(fold_train_dfs)
                all_train_classes.extend(fold_train_classes)
                all_train_fold_info.extend([fold_name] * len(fold_train_dfs))

            # Load test data
            test_file = os.path.join(fold_path, f"test_windowed.{config.SAVE_FORMAT}")
            if os.path.exists(test_file):
                fold_test_dfs, fold_test_classes = persistence._load_dataframes(test_file, config.SAVE_FORMAT)
                all_test_windows.extend(fold_test_dfs)
                all_test_classes.extend(fold_test_classes)
                all_test_fold_info.extend([fold_name] * len(fold_test_dfs))

        if verbose:
            print(f"✅ Data loaded successfully!")
            print(f"   Training windows: {len(all_train_windows)}")
            print(f"   Test windows: {len(all_test_windows)}")
            if all_train_windows:
                print(f"   Window shape: {all_train_windows[0].shape}")

        return (all_train_windows, all_train_classes, all_train_fold_info,
                all_test_windows, all_test_classes, all_test_fold_info)

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
    tree_models = [r for r in results if "Tree" in r["model_name"] or "Forest" in r["model_name"]]
    svm_models = [r for r in results if "SVM" in r["model_name"]]
    nn_models = [r for r in results if "Neural Network" in r["model_name"]]

    # Create analysis with best performers
    analysis = {}
    
    if tree_models:
        best_tree = max(tree_models, key=lambda x: x["test_accuracy"])
        analysis['Tree-Based'] = {
            'best_algorithm': best_tree['model_name'],
            'best_accuracy': best_tree['test_accuracy'],
            'count': len(tree_models)
        }
    
    if svm_models:
        best_svm = max(svm_models, key=lambda x: x["test_accuracy"])
        analysis['Support Vector Machines'] = {
            'best_algorithm': best_svm['model_name'],
            'best_accuracy': best_svm['test_accuracy'],
            'count': len(svm_models)
        }
    
    if nn_models:
        best_nn = max(nn_models, key=lambda x: x["test_accuracy"])
        analysis['Neural Networks'] = {
            'best_algorithm': best_nn['model_name'],
            'best_accuracy': best_nn['test_accuracy'],
            'count': len(nn_models)
        }

    if verbose:
        print(f"🔧 Algorithm Analysis:")
        print(f"   • Tree-Based: {len(tree_models)} models")
        print(f"   • Support Vector Machines: {len(svm_models)} models")
        print(f"   • Neural Networks: {len(nn_models)} models")

        # Best performers by category
        if tree_models:
            best_tree = max(tree_models, key=lambda x: x["test_accuracy"])
            print(f"\n🌳 Best Tree: {best_tree['model_name']} ({best_tree['test_accuracy']:.3f})")

        if svm_models:
            best_svm = max(svm_models, key=lambda x: x["test_accuracy"])
            print(f"⚡ Best SVM: {best_svm['model_name']} ({best_svm['test_accuracy']:.3f})")

        if nn_models:
            best_nn = max(nn_models, key=lambda x: x["test_accuracy"])
            print(f"🧠 Best NN: {best_nn['model_name']} ({best_nn['test_accuracy']:.3f})")

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
                train_dfs, train_mapped_classes, strategy=balance_strategy, internal_verbose=internal_verbose
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
            print(
                f"Training: {X_train.shape[0]} samples, {X_train.shape[1]} features"
            )
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
        raise ValueError(f"train_fold_info length ({len(train_fold_info)}) != train_dfs length ({len(train_dfs)})")
    
    if len(test_fold_info) != len(test_dfs):
        raise ValueError(f"test_fold_info length ({len(test_fold_info)}) != test_dfs length ({len(test_dfs)})")
    
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
    model_names = ["Decision Tree", "Random Forest", "Linear SVM", "RBF SVM", 
                  "Simple Neural Network", "Deep Neural Network", "Regularized Neural Network"]
    
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
        temp_classifier.label_encoder = classifier.label_encoder  # Use consistent encoding
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
                internal_verbose=False
            )
            
            if verbose:
                print(f"   Data prepared - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
            
            # Train all models for this fold
            fold_model_results = temp_classifier.train_all_models(X_train, y_train, X_test, y_test)
            
            # Extract test accuracies for this fold
            for result in fold_model_results:
                model_name = result['model_name']
                test_accuracy = result['test_accuracy']
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
                internal_verbose=True
            )
            
    # Train all models for this fold
    last_fold_model_results = classifier.train_all_models(X_train, y_train, X_test, y_test)
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
        ["Regularized NN", "2 hidden", "150+100", "High (0.001)", "Medium"]
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
