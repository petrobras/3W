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

        Returns:
            tuple: (X_train, y_train, X_test, y_test) - prepared arrays
        """
        if self.verbose:
            print("Preparing Data for Supervised Classification")
            print("=" * 55)

        # Extract class labels from windows
        if self.verbose:
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
        

        if self.verbose:
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
                train_dfs, train_mapped_classes, strategy=balance_strategy
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
        if self.verbose:
            print("Encoding labels...", end=" ")

        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        self.class_names = self.label_encoder.classes_
        

        if self.verbose:
            print(f"\nData prepared successfully!")
            print(
                f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features"
            )
            print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
            print(
                f"Classes: {len(np.unique(y_train_encoded))} ({list(self.class_names)})"
            )
            print(
                f"Features per window: {X_train.shape[1]} (data already normalized)"
            )

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

        # Show detailed classification report for best model
        print(f"\n📋 Detailed Classification Report ({best_model['model_name']}):")
        print("=" * 60)

        # Get unique classes present in y_test (after class 0 filtering)
        unique_classes = sorted(np.unique(y_test))
        target_names = [f"Class {cls}" for cls in unique_classes]
        print(
            classification_report(
                y_test, best_model["predictions"], target_names=target_names
            )
        )

        # Best model identification completed
        pass

    def visualize_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Create visualization of model performance comparison.

        Args:
            results: List of model results
        """
        if not results:
            print(" No results to visualize.")
            return

        print(f"\n📈 Creating performance visualization...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        model_names = [r["model_name"] for r in results]
        train_accs = [r["train_accuracy"] for r in results]
        test_accs = [r["test_accuracy"] for r in results]
        train_times = [r["training_time"] for r in results]

        # Plot 1: Accuracy Comparison
        x = range(len(model_names))
        width = 0.35
        ax1.bar(
            [i - width / 2 for i in x],
            train_accs,
            width,
            label="Training Accuracy",
            alpha=0.8,
        )
        ax1.bar(
            [i + width / 2 for i in x],
            test_accs,
            width,
            label="Test Accuracy",
            alpha=0.8,
        )
        ax1.set_xlabel("Models")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Training vs Test Accuracy")
        ax1.set_xticks(x)
        ax1.set_xticklabels(
            [name.replace(" ", "\n") for name in model_names], rotation=45, ha="right"
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Training Time
        bars = ax2.bar(model_names, train_times, color="skyblue", alpha=0.8)
        ax2.set_xlabel("Models")
        ax2.set_ylabel("Training Time (seconds)")
        ax2.set_title("Training Time Comparison")
        ax2.set_xticklabels(
            [name.replace(" ", "\n") for name in model_names], rotation=45, ha="right"
        )
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}s",
                ha="center",
                va="bottom",
            )

        # Plot 3: Overfitting Analysis
        overfitting_gaps = [
            train_acc - test_acc for train_acc, test_acc in zip(train_accs, test_accs)
        ]
        colors = [
            "red" if gap > 0.05 else "orange" if gap > 0.02 else "green"
            for gap in overfitting_gaps
        ]
        bars = ax3.bar(model_names, overfitting_gaps, color=colors, alpha=0.8)
        ax3.set_xlabel("Models")
        ax3.set_ylabel("Train - Test Accuracy")
        ax3.set_title("Overfitting Analysis (Lower is Better)")
        ax3.set_xticklabels(
            [name.replace(" ", "\n") for name in model_names], rotation=45, ha="right"
        )
        ax3.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax3.axhline(
            y=0.05, color="red", linestyle="--", alpha=0.5, label="High Overfitting"
        )
        ax3.axhline(
            y=0.02,
            color="orange",
            linestyle="--",
            alpha=0.5,
            label="Moderate Overfitting",
        )
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Accuracy vs Training Time Scatter
        ax4.scatter(train_times, test_accs, s=100, alpha=0.7, color="purple")
        for i, name in enumerate(model_names):
            ax4.annotate(
                name.replace(" ", "\n"),
                (train_times[i], test_accs[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        ax4.set_xlabel("Training Time (seconds)")
        ax4.set_ylabel("Test Accuracy")
        ax4.set_title("Accuracy vs Training Time Trade-off")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# Enhanced analysis methods
def analyze_fold_accuracy(
    classifier: SupervisedClassifier, fold_results: Dict[str, Dict]
) -> None:
    """
    Analyze accuracy per fold for each model.

    Args:
        classifier: Trained classifier
        fold_results: Dictionary with fold results {fold_name: {model_name: accuracy}}
    """
    print("\nAccuracy Analysis Per Fold")
    print("=" * 50)

    if not fold_results:
        print(" No fold results available")
        return

    # Get all model names and fold names
    fold_names = sorted(fold_results.keys())
    model_names = set()
    for fold_data in fold_results.values():
        model_names.update(fold_data.keys())
    model_names = sorted(model_names)

    # Create summary table
    print(f"{'Fold':<10}", end="")
    for model in model_names:
        print(f"{model:<15}", end="")
    print()
    print("-" * (10 + 15 * len(model_names)))

    for fold in fold_names:
        print(f"{fold:<10}", end="")
        for model in model_names:
            accuracy = fold_results.get(fold, {}).get(model, 0.0)
            print(f"{accuracy:<15.3f}", end="")
        print()

    # Calculate average accuracy per model across folds
    print(f"\n{'Average':<10}", end="")
    for model in model_names:
        accuracies = [fold_results.get(fold, {}).get(model, 0.0) for fold in fold_names]
        avg_acc = np.mean([acc for acc in accuracies if acc > 0])
        print(f"{avg_acc:<15.3f}", end="")
    print()


def analyze_class_accuracy(
    classifier: SupervisedClassifier,
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
) -> None:
    """
    Analyze accuracy per class for each model.

    Args:
        classifier: Trained classifier
        y_true: True labels
        predictions_dict: Dictionary with model predictions {model_name: predictions}
    """
    print("\nAccuracy Analysis Per Class")
    print("=" * 50)

    class_names = (
        classifier.class_names
        if classifier.class_names is not None
        else np.unique(y_true)
    )
    model_names = sorted(predictions_dict.keys())

    # Calculate per-class accuracy for each model
    print(f"{'Class':<8}", end="")
    for model in model_names:
        print(f"{model:<15}", end="")
    print()
    print("-" * (8 + 15 * len(model_names)))

    for class_idx, class_name in enumerate(class_names):
        class_mask = y_true == class_idx
        if not np.any(class_mask):
            continue

        print(f"{class_name:<8}", end="")
        for model in model_names:
            y_pred = predictions_dict[model]
            class_accuracy = accuracy_score(y_true[class_mask], y_pred[class_mask])
            print(f"{class_accuracy:<15.3f}", end="")
        print()

    # Overall accuracy
    print(f"\n{'Overall':<8}", end="")
    for model in model_names:
        y_pred = predictions_dict[model]
        overall_accuracy = accuracy_score(y_true, y_pred)
        print(f"{overall_accuracy:<15.3f}", end="")
    print()


def analyze_accuracy_without_class0(
    classifier: SupervisedClassifier,
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    selected_classes=None,
) -> None:
    """
    Show which classes were included/excluded from analysis.

    Args:
        classifier: Trained classifier
        y_true: True labels
        predictions_dict: Dictionary with model predictions {model_name: predictions}
        selected_classes: List of classes that were selected for analysis (None = all except 0)
    """
    print("\nClass Selection Status")
    print("=" * 35)

    # Check which classes are present in the data
    unique_classes = np.unique(y_true)

    if selected_classes is None:
        # Default mode: class 0 exclusion
        has_class_0 = 0 in unique_classes
        if not has_class_0:
            print(
                "Class 0 (normal operation) successfully excluded from all analysis"
            )
            print(f"Dataset contains only fault classes: {sorted(unique_classes)}")


        else:
            print("Warning: Class 0 still present in data - check filtering logic")
    else:
        # Custom class selection mode
        print(f"Custom class selection applied")
        print(f"Selected classes: {sorted(selected_classes)}")
        print(f"Dataset contains classes: {sorted(unique_classes)}")



    print(f"Total samples in analysis: {len(y_true)}")

    # Show class distribution
    unique_classes, counts = np.unique(y_true, return_counts=True)
    print(f"\nClass distribution in analysis:")
    for cls, count in zip(unique_classes, counts):
        class_name = (
            classifier.class_names[cls] if classifier.class_names is not None else cls
        )
        print(f"   Class {class_name}: {count} samples")
        print(f"Classes found: {sorted(unique_classes)}")

        # Calculate accuracy excluding class 0 for comparison
        non_class0_mask = y_true != 0
        if np.any(non_class0_mask):
            y_true_filtered = y_true[non_class0_mask]

            print(f"\nComparison with class 0 excluded:")
            print(f"   • Total samples: {len(y_true)}")
            print(f"   • Fault-only samples: {len(y_true_filtered)}")

            for model_name in sorted(predictions_dict.keys()):
                y_pred = predictions_dict[model_name]
                y_pred_filtered = y_pred[non_class0_mask]

                overall_accuracy = accuracy_score(y_true, y_pred)
                fault_only_accuracy = accuracy_score(y_true_filtered, y_pred_filtered)

                print(
                    f"   {model_name}: Overall {overall_accuracy:.3f} vs Fault-only {fault_only_accuracy:.3f}"
                )


# Convenience function for quick usage
def quick_supervised_classification(
    train_dfs: List[pd.DataFrame],
    train_classes: List[str],
    test_dfs: List[pd.DataFrame],
    test_classes: List[str],
    balance_classes: bool = True,
    balance_strategy: str = "combined",
    max_samples_per_class: int = 300,
    verbose: bool = True,
) -> SupervisedClassifier:
    """
    Quick supervised classification with all models.

    Args:
        train_dfs, train_classes: Training data
        test_dfs, test_classes: Test data
        balance_classes: Whether to balance classes
        balance_strategy: Strategy for balancing
        max_samples_per_class: Max samples per class
        verbose: Print details

    Returns:
        SupervisedClassifier: Trained classifier with results
    """
    classifier = SupervisedClassifier(verbose=verbose)

    # Prepare data
    X_train, y_train, X_test, y_test = classifier.prepare_data(
        train_dfs,
        train_classes,
        test_dfs,
        test_classes,
        balance_classes=balance_classes,
        balance_strategy=balance_strategy,
        max_samples_per_class=max_samples_per_class,
    )

    # Train all models
    results = classifier.train_all_models(X_train, y_train, X_test, y_test)

    # Compare models
    classifier.compare_models(results, y_test)

    # Visualize results
    classifier.visualize_results(results)

    return classifier


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
    verbose: bool = True,
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
    if verbose:
        print("Enhanced Cross-Fold Classification Analysis")
        print("=" * 50)
    
    # Check if we have fold information for proper cross-validation
    if train_fold_info is None or test_fold_info is None:
        if verbose:
            print("Warning: Fold information missing - falling back to simple classification")
        
        # Fallback to original enhanced analysis without fold-wise training
        return enhanced_fold_analysis_simple(
            train_dfs, train_classes, test_dfs, test_classes,
            fold_info=test_fold_info,
            balance_classes=balance_classes,
            balance_strategy=balance_strategy,
            max_samples_per_class=max_samples_per_class,
            balance_test=balance_test,
            min_test_samples_per_class=min_test_samples_per_class,
            selected_classes=selected_classes,
            verbose=verbose
        )
    
    # Validate fold information
    if len(train_fold_info) != len(train_dfs):
        raise ValueError(f"train_fold_info length ({len(train_fold_info)}) != train_dfs length ({len(train_dfs)})")
    
    if len(test_fold_info) != len(test_dfs):
        raise ValueError(f"test_fold_info length ({len(test_fold_info)}) != test_dfs length ({len(test_dfs)})")
    
    # Get unique folds
    train_folds = sorted(set(train_fold_info))
    test_folds = sorted(set(test_fold_info))
    all_folds = sorted(set(train_folds + test_folds))
    
    if verbose:
        print(f"📁 Found folds - Train: {train_folds}, Test: {test_folds}")
        print(f"📁 Will perform cross-validation across {len(all_folds)} folds: {all_folds}")
    
    # Filter data based on selected_classes parameter
    if selected_classes is None:
        filter_message = "Warning: Filtering out class 0 (normal operation) from all data..."
        filter_condition = lambda cls: cls != 0 and cls != "0"
    else:
        selected_classes_str = [str(c) for c in selected_classes]
        filter_message = f" Filtering to include only selected classes: {selected_classes}..."
        filter_condition = lambda cls: str(cls) in selected_classes_str

    if verbose:
        print(filter_message)

    # Filter training and test data based on class selection
    filtered_train_data = []
    filtered_test_data = []
    
    for i, (df, cls, fold) in enumerate(zip(train_dfs, train_classes, train_fold_info)):
        if filter_condition(cls):
            filtered_train_data.append((df, cls, fold, i))
    
    for i, (df, cls, fold) in enumerate(zip(test_dfs, test_classes, test_fold_info)):
        if filter_condition(cls):
            filtered_test_data.append((df, cls, fold, i))
    
    if verbose:
        print(f"Class filtering completed:")
        print(f"   • Training samples: {len(train_dfs)} → {len(filtered_train_data)}")
        print(f"   • Test samples: {len(test_dfs)} → {len(filtered_test_data)}")
    
    if len(filtered_train_data) == 0 or len(filtered_test_data) == 0:
        raise ValueError("No data remaining after class filtering")
    
    # Initialize classifier for consistent label encoding across folds
    classifier = SupervisedClassifier(verbose=verbose)
    
    # Prepare all data once to establish consistent label encoding
    all_train_dfs = [item[0] for item in filtered_train_data]
    all_train_classes = [item[1] for item in filtered_train_data]
    all_test_dfs = [item[0] for item in filtered_test_data]
    all_test_classes = [item[1] for item in filtered_test_data]
    
    # Create label encoder with all classes
    all_classes = list(set(all_train_classes + all_test_classes))
    classifier.label_encoder.fit(all_classes)
    classifier.class_names = classifier.label_encoder.classes_
    
    if verbose:
        print(f"Established label encoding for classes: {sorted(all_classes)}")
    
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
        if verbose:
            print(f"\nTraining on other folds, testing on fold {test_fold}")
        
        # Separate training and test data for this fold iteration
        fold_train_dfs = []
        fold_train_classes = []
        fold_test_dfs = []
        fold_test_classes = []
        
        # Collect training data (all folds except test_fold)
        for df, cls, fold, _ in filtered_train_data:
            if fold != test_fold:
                fold_train_dfs.append(df)
                fold_train_classes.append(cls)
        
        # Collect test data (only test_fold)
        for df, cls, fold, _ in filtered_test_data:
            if fold == test_fold:
                fold_test_dfs.append(df)
                fold_test_classes.append(cls)
        
        if len(fold_train_dfs) == 0:
            if verbose:
                print(f"Warning: No training data for test fold {test_fold}, skipping")
            continue
            
        if len(fold_test_dfs) == 0:
            if verbose:
                print(f"Warning: No test data for fold {test_fold}, skipping")
            continue
        
        if verbose:
            print(f"   Training samples: {len(fold_train_dfs)}")
            print(f"   Test samples: {len(fold_test_dfs)}")
        
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
    
    # Create final classifier with all data for visualization and comparison
    if verbose:
        print(f"\n Training final models on all data for comparison...")
    
    final_X_train, final_y_train, final_X_test, final_y_test = classifier.prepare_data(
        all_train_dfs,
        all_train_classes,
        all_test_dfs,
        all_test_classes,
        balance_classes=balance_classes,
        balance_strategy=balance_strategy,
        max_samples_per_class=max_samples_per_class,
    )
    
    # Train final models
    final_results = classifier.train_all_models(final_X_train, final_y_train, final_X_test, final_y_test)
    
    # Store fold results in classifier for analysis
    classifier.fold_results = fold_results
    classifier.fold_names = all_folds
    
    if verbose:
        print("\n" + "=" * 70)
        print("CROSS-FOLD VALIDATION RESULTS")
        print("=" * 70)
        
        # Display fold results table
        print(f"{'Fold':<12}", end="")
        for model_name in model_names:
            print(f"{model_name[:12]:<15}", end="")
        print()
        print("-" * (12 + 15 * len(model_names)))
        
        for fold in all_folds:
            print(f"{fold:<12}", end="")
            for model_name in model_names:
                accuracy = fold_results[fold].get(model_name, 0.0)
                print(f"{accuracy:<15.3f}", end="")
            print()
        
        # Calculate and display averages
        print("-" * (12 + 15 * len(model_names)))
        print(f"{'Average':<12}", end="")
        for model_name in model_names:
            accuracies = [fold_results[fold].get(model_name, 0.0) for fold in all_folds]
            avg_accuracy = np.mean([acc for acc in accuracies if acc > 0])
            print(f"{avg_accuracy:<15.3f}", end="")
        print()
        
        # Show best performing model per fold
        print(f"\nBest Model Per Fold:")
        for fold in all_folds:
            fold_accs = fold_results[fold]
            if any(acc > 0 for acc in fold_accs.values()):
                best_model = max(fold_accs.items(), key=lambda x: x[1])
                print(f"   Fold {fold}: {best_model[0]} ({best_model[1]:.3f})")
    
    # Compare models and visualize (using final trained models)
    classifier.compare_models(final_results, final_y_test)
    # classifier.visualize_results(final_results)
    
    return classifier


def enhanced_fold_analysis_simple(
    train_dfs: List[pd.DataFrame],
    train_classes: List[str],
    test_dfs: List[pd.DataFrame],
    test_classes: List[str],
    fold_info: List[str] = None,
    balance_classes: bool = True,
    balance_strategy: str = "combined",
    max_samples_per_class: int = 300,
    balance_test: bool = True,
    min_test_samples_per_class: int = 300,
    selected_classes: List[str] = None,
    verbose: bool = True,
) -> SupervisedClassifier:
    """
    Enhanced supervised classification with fold-wise analysis and class-specific metrics.
    This is the original implementation that trains on all data at once.

    Args:
        train_dfs, train_classes: Training data
        test_dfs, test_classes: Test data
        fold_info: List indicating which fold each sample belongs to
        balance_classes: Whether to balance training classes
        balance_strategy: Strategy for balancing ('combined', 'oversample', 'undersample')
        max_samples_per_class: Max samples per class for training
        balance_test: Whether to balance test classes for robust evaluation
        min_test_samples_per_class: Minimum samples per class in test set
        selected_classes: List of classes to include in analysis (None = all classes except 0)
        verbose: Print details

    Returns:
        SupervisedClassifier: Trained classifier with enhanced results
    """
    classifier = SupervisedClassifier(verbose=verbose)

    if verbose:
        print("Enhanced Classification with Test Balancing and Class 0 Exclusion")
        print("=" * 65)

    # Filter data based on selected_classes parameter
    if selected_classes is None:
        # Default behavior: exclude only class 0 (normal operation)
        filter_message = "Warning: Filtering out class 0 (normal operation) from all data..."
        filter_condition = lambda cls: cls != 0 and cls != "0"
    else:
        # Custom class selection: include only specified classes
        selected_classes_str = [
            str(c) for c in selected_classes
        ]  # Convert to strings for comparison
        filter_message = (
            f" Filtering to include only selected classes: {selected_classes}..."
        )
        filter_condition = lambda cls: str(cls) in selected_classes_str

    if verbose:
        print(filter_message)

    # Filter training data
    train_filtered_dfs = []
    train_filtered_classes = []
    train_filtered_fold_info = []

    for i, (df, cls) in enumerate(zip(train_dfs, train_classes)):
        if filter_condition(cls):
            train_filtered_dfs.append(df)
            train_filtered_classes.append(cls)
            if fold_info is not None and i < len(fold_info):
                train_filtered_fold_info.append(fold_info[i])

    # Filter test data
    test_filtered_dfs = []
    test_filtered_classes = []
    test_filtered_fold_info = []

    original_fold_info = fold_info.copy() if fold_info is not None else None
    for i, (df, cls) in enumerate(zip(test_dfs, test_classes)):
        if filter_condition(cls):
            test_filtered_dfs.append(df)
            test_filtered_classes.append(cls)
            if original_fold_info is not None and i < len(original_fold_info):
                test_filtered_fold_info.append(original_fold_info[i])

    if verbose:
        if selected_classes is None:
            print(f"Class 0 filtering completed:")
        else:
            print(f"Class filtering completed:")
        print(
            f"   • Original training samples: {len(train_dfs)} → Filtered: {len(train_filtered_dfs)}"
        )
        print(
            f"   • Original test samples: {len(test_dfs)} → Filtered: {len(test_filtered_dfs)}"
        )

        # Show remaining class distribution
        train_unique, train_counts = np.unique(
            train_filtered_classes, return_counts=True
        )
        print(
            f"   • Remaining training classes: {dict(zip(train_unique, train_counts))}"
        )

        test_unique, test_counts = np.unique(test_filtered_classes, return_counts=True)
        print(f"   • Remaining test classes: {dict(zip(test_unique, test_counts))}")

    # Update data and fold info to use filtered versions
    train_dfs = train_filtered_dfs
    train_classes = train_filtered_classes
    test_dfs = test_filtered_dfs
    test_classes = test_filtered_classes

    # Update fold info to match filtered data
    if fold_info is not None:
        fold_info = test_filtered_fold_info

    if len(train_dfs) == 0 or len(test_dfs) == 0:
        if selected_classes is None:
            raise ValueError(
                "No data remaining after filtering out class 0. Check your data labels."
            )
        else:
            raise ValueError(
                f"No data remaining after filtering for classes {selected_classes}. Check your data labels and selected classes."
            )

    # Track indices for fold_info consistency
    original_test_indices = list(range(len(test_dfs)))

    # Balance test data if requested
    if balance_test:
        if verbose:
            print(
                f"Balancing test data to ensure min {min_test_samples_per_class} samples per class..."
            )

        # Import the data augmentation function
        from .data_augmentation import quick_balance_classes

        # Balance test data
        balanced_test_dfs, balanced_test_classes = quick_balance_classes(
            test_dfs,
            test_classes,
            strategy=balance_strategy,
            min_samples_per_class=min_test_samples_per_class,
        )

        if verbose:
            print(f"Test data balanced:")
            print(f"   • Original test samples: {len(test_dfs)}")
            print(f"   • Balanced test samples: {len(balanced_test_dfs)}")

            # Show class distribution after balancing
            test_unique, test_counts = np.unique(
                balanced_test_classes, return_counts=True
            )
            print(f"   • Balanced test distribution:")
            for cls, count in zip(test_unique, test_counts):
                print(f"     Class {cls}: {count} samples")

        # Update test data and fold info
        test_dfs = balanced_test_dfs
        test_classes = balanced_test_classes

        # Note: fold_info will not match perfectly after balancing test data
        # We'll skip per-fold analysis when test balancing is enabled
        if fold_info is not None and balance_test:
            if verbose:
                print("Warning: Per-fold analysis disabled when test balancing is enabled")
                print(
                    "   (Fold information becomes inconsistent after test data augmentation)"
                )
            fold_info = None

    # Prepare data (class 0 already filtered out, but ensure it stays out)
    X_train, y_train, X_test, y_test = classifier.prepare_data(
        train_dfs,
        train_classes,
        test_dfs,
        test_classes,
        balance_classes=balance_classes,
        balance_strategy=balance_strategy,
        max_samples_per_class=max_samples_per_class,
    )

    # Track test data filtering for fold consistency
    test_filter_mask = None

    # Double-check that class 0 is not in the final prepared data
    if 0 in y_train or 0 in y_test:
        if verbose:
            print("Warning: Class 0 detected in prepared data, removing...")

        # Remove class 0 from training data
        train_mask = y_train != 0
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]

        # Remove class 0 from test data and track the mask for fold_info
        test_mask = y_test != 0
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]
        test_filter_mask = test_mask  # Store the mask for fold_info filtering

        if verbose:
            print(f"Final class 0 removal completed:")
            print(f"   • Training samples after final filter: {len(X_train)}")
            print(f"   • Test samples after final filter: {len(X_test)}")
            print(f"   • Final training classes: {sorted(np.unique(y_train))}")
            print(f"   • Final test classes: {sorted(np.unique(y_test))}")

    # Handle fold_info filtering based on test data size mismatch
    if fold_info is not None:
        original_fold_info_length = len(fold_info)
        test_data_length = len(y_test)
        
        if original_fold_info_length != test_data_length:
            if verbose:
                print(f"Warning: fold_info length ({original_fold_info_length}) != test data length ({test_data_length})")
                print(f"   This suggests test data was filtered during prepare_data()")
                print(f"   Setting fold_info to None to avoid indexing errors")
            
            # Set fold_info to None rather than attempting to filter it
            # This is safer than trying to guess which samples were filtered
            fold_info = None
        else:
            if verbose:
                print(f"Fold information matches test data length ({test_data_length})")

    # Verify fold_info consistency before proceeding
    if fold_info is not None and len(fold_info) != len(y_test):
        if verbose:
            print(f"Warning: Fold info length mismatch detected:")
            print(f"   • fold_info length: {len(fold_info)}")
            print(f"   • test data length: {len(y_test)}")
            print(f"   • Disabling per-fold analysis for safety")
        fold_info = None

    # Train all models and get predictions
    results = classifier.train_all_models(X_train, y_train, X_test, y_test)

    # Extract predictions for enhanced analysis
    predictions_dict = {}
    for result in results:
        model = result["model"]  # Fixed: use 'model' key instead of 'trained_model'
        model_name = result["model_name"]
        predictions_dict[model_name] = model.predict(X_test)

    print("\n" + "=" * 70)
    print("🔍 ENHANCED CLASSIFICATION ANALYSIS (FAULT-ONLY)")
    print("=" * 70)

    # Analysis 1: Accuracy per fold (if fold info available)
    if fold_info is not None and len(fold_info) == len(test_dfs):
        fold_results = {}
        unique_folds = sorted(set(fold_info))

        print(f"\nFound {len(unique_folds)} unique folds for analysis")

        for fold in unique_folds:
            fold_mask = np.array([fold_info[i] == fold for i in range(len(fold_info))])
            if not np.any(fold_mask):
                continue

            fold_results[fold] = {}
            for model_name, predictions in predictions_dict.items():
                fold_accuracy = accuracy_score(
                    y_test[fold_mask], predictions[fold_mask]
                )
                fold_results[fold][model_name] = fold_accuracy

        analyze_fold_accuracy(classifier, fold_results)
    else:
        print("\nWarning: Fold information not available - skipping per-fold analysis")

    # Analysis 2: Accuracy per class (fault classes only)
    analyze_class_accuracy(classifier, y_test, predictions_dict)

    # Analysis 3: Show class selection status
    analyze_accuracy_without_class0(
        classifier, y_test, predictions_dict, selected_classes
    )

    # Original comparison and visualization
    classifier.compare_models(results, y_test)
    classifier.visualize_results(results)

    return classifier
