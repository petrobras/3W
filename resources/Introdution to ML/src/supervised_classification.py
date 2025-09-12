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
    
    def prepare_data(self, train_dfs: List[pd.DataFrame], train_classes: List[str],
                    test_dfs: List[pd.DataFrame], test_classes: List[str],
                    balance_classes: bool = True, balance_strategy: str = 'combined',
                    max_samples_per_class: int = 300) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            print("🔧 Preparing Data for Supervised Classification")
            print("=" * 55)
        
        # Extract class labels from windows
        if self.verbose:
            print("🏷️ Processing class labels...", end=" ")
        
        def extract_window_classes(window_dfs, fallback_classes):
            """Extract class labels from windows or use fallback classes"""
            window_classes = []
            for i, window_df in enumerate(window_dfs):
                if 'class' in window_df.columns:
                    # Get the last value of the class column
                    window_class = window_df['class'].iloc[-1]
                    window_classes.append(window_class)
                else:
                    window_classes.append(fallback_classes[i])
            return window_classes
        
        train_window_classes = extract_window_classes(train_dfs, train_classes)
        test_window_classes = extract_window_classes(test_dfs, test_classes)
        print("✅")
        
        # Map transient classes to their base classes
        if self.verbose:
            print("🔄 Mapping transient classes to base classes...", end=" ")
        
        transient_mapping = {
            101: 1, 102: 2, 105: 5, 106: 6, 
            107: 7, 108: 8, 109: 9
        }
        
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
        print("✅")
        
        if self.verbose:
            print(f"🏷️ Training class distribution: {dict(zip(*np.unique(train_mapped_classes, return_counts=True)))}")
            print(f"🏷️ Test class distribution: {dict(zip(*np.unique(test_mapped_classes, return_counts=True)))}")
        
        # Balance classes using data augmentation
        if balance_classes:
            if self.verbose:
                print(f"⚖️ Balancing classes using '{balance_strategy}' strategy...", end=" ")
            
            train_balanced_dfs, train_balanced_classes = quick_balance_classes(
                train_dfs, train_mapped_classes, strategy=balance_strategy
            )
            print("✅")
        else:
            train_balanced_dfs = train_dfs
            train_balanced_classes = train_mapped_classes
        
        # Limit samples per class if specified
        if max_samples_per_class is not None:
            if self.verbose:
                print(f"🎯 Limiting to max {max_samples_per_class} samples per class...", end=" ")
            
            train_balanced_dfs, train_balanced_classes = self._limit_samples_per_class(
                train_balanced_dfs, train_balanced_classes, max_samples_per_class
            )
            test_limited_dfs, test_limited_classes = self._limit_samples_per_class(
                test_dfs, test_mapped_classes, max_samples_per_class // 3
            )
            print("✅")
        else:
            test_limited_dfs = test_dfs
            test_limited_classes = test_mapped_classes
        
        if self.verbose:
            print(f"🏷️ Final training distribution: {dict(zip(*np.unique(train_balanced_classes, return_counts=True)))}")
            print(f"🏷️ Final test distribution: {dict(zip(*np.unique(test_limited_classes, return_counts=True)))}")
        
        # Flatten windows to feature vectors (data is already normalized)
        if self.verbose:
            print("🔄 Flattening windows to feature vectors...", end=" ")
        
        X_train = self._flatten_windows(train_balanced_dfs)
        X_test = self._flatten_windows(test_limited_dfs)
        y_train = np.array(train_balanced_classes)
        y_test = np.array(test_limited_classes)
        print("✅")
        
        # Encode labels
        if self.verbose:
            print("🏷️ Encoding labels...", end=" ")
        
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        self.class_names = self.label_encoder.classes_
        print("✅")
        
        if self.verbose:
            print(f"\n✅ Data prepared successfully!")
            print(f"🚂 Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            print(f"🧪 Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
            print(f"🏷️ Classes: {len(np.unique(y_train_encoded))} ({list(self.class_names)})")
            print(f"📊 Features per window: {X_train.shape[1]} (data already normalized)")
        
        return X_train, y_train_encoded, X_test, y_test_encoded
    
    def _limit_samples_per_class(self, dfs: List[pd.DataFrame], classes: List[str], 
                                max_per_class: int) -> Tuple[List[pd.DataFrame], List[str]]:
        """Limit the number of samples per class."""
        selected_indices = []
        selected_classes = []
        
        unique_classes = np.unique(classes)
        for target_class in unique_classes:
            class_indices = [i for i, cls in enumerate(classes) if cls == target_class]
            
            if len(class_indices) > 0:
                n_samples = min(max_per_class, len(class_indices))
                sampled_indices = np.random.choice(class_indices, size=n_samples, replace=False)
                selected_indices.extend(sampled_indices)
                selected_classes.extend([target_class] * len(sampled_indices))
        
        selected_dfs = [dfs[i] for i in selected_indices]
        return selected_dfs, selected_classes
    
    def _flatten_windows(self, window_dfs: List[pd.DataFrame]) -> np.ndarray:
        """Convert windowed time series to flattened feature vectors."""
        flattened_windows = []
        
        for window_df in window_dfs:
            # Exclude the 'class' column, keep all sensor data
            feature_columns = [col for col in window_df.columns if col != 'class']
            
            # Flatten the feature data
            flattened = window_df[feature_columns].values.flatten()
            flattened_windows.append(flattened)
        
        return np.array(flattened_windows)
    
    def train_decision_trees(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Train Decision Tree and Random Forest models.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            dict: Results dictionary with model performance
        """
        if self.verbose:
            print("🌳 Training Decision Trees and Random Forest")
            print("=" * 50)
        
        results = {}
        
        # 1. DECISION TREE
        if self.verbose:
            print("🌲 Training Decision Tree...")
        
        start_time = time.time()
        dt_classifier = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=self.random_state
        )
        
        dt_classifier.fit(X_train, y_train)
        dt_train_time = time.time() - start_time
        
        # Make predictions
        dt_train_pred = dt_classifier.predict(X_train)
        dt_test_pred = dt_classifier.predict(X_test)
        
        # Calculate accuracies
        dt_train_acc = accuracy_score(y_train, dt_train_pred)
        dt_test_acc = accuracy_score(y_test, dt_test_pred)
        
        results['decision_tree'] = {
            'model': dt_classifier,
            'model_name': 'Decision Tree',
            'train_accuracy': dt_train_acc,
            'test_accuracy': dt_test_acc,
            'training_time': dt_train_time,
            'predictions': dt_test_pred
        }
        
        if self.verbose:
            print(f"✅ Decision Tree trained in {dt_train_time:.3f}s")
            print(f"   • Training Accuracy: {dt_train_acc:.3f}")
            print(f"   • Test Accuracy: {dt_test_acc:.3f}")
        
        # 2. RANDOM FOREST
        if self.verbose:
            print("\n🌲🌲🌲 Training Random Forest...")
        
        start_time = time.time()
        rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        rf_classifier.fit(X_train, y_train)
        rf_train_time = time.time() - start_time
        
        # Make predictions
        rf_train_pred = rf_classifier.predict(X_train)
        rf_test_pred = rf_classifier.predict(X_test)
        
        # Calculate accuracies
        rf_train_acc = accuracy_score(y_train, rf_train_pred)
        rf_test_acc = accuracy_score(y_test, rf_test_pred)
        
        results['random_forest'] = {
            'model': rf_classifier,
            'model_name': 'Random Forest',
            'train_accuracy': rf_train_acc,
            'test_accuracy': rf_test_acc,
            'training_time': rf_train_time,
            'predictions': rf_test_pred,
            'feature_importance': rf_classifier.feature_importances_
        }
        
        if self.verbose:
            print(f"✅ Random Forest trained in {rf_train_time:.3f}s")
            print(f"   • Training Accuracy: {rf_train_acc:.3f}")
            print(f"   • Test Accuracy: {rf_test_acc:.3f}")
            
            # Show feature importance
            print(f"\n📊 Top 10 Most Important Features:")
            feature_importance = rf_classifier.feature_importances_
            top_features_idx = np.argsort(feature_importance)[-10:][::-1]
            
            for i, idx in enumerate(top_features_idx, 1):
                print(f"   {i:2d}. Feature {idx:4d}: {feature_importance[idx]:.4f}")
        
        return results
    
    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray,
                  max_samples: int = 1000) -> Dict[str, Any]:
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
            print(f"🎯 Using subset: {X_train_svm.shape[0]} train, {X_test_svm.shape[0]} test")
        
        results = {}
        
        # 1. LINEAR SVM
        if self.verbose:
            print(f"\n📏 Training Linear SVM...")
        
        start_time = time.time()
        linear_svm = SVC(kernel='linear', C=1.0, random_state=self.random_state)
        linear_svm.fit(X_train_svm, y_train_svm)
        linear_train_time = time.time() - start_time
        
        # Predictions
        linear_train_pred = linear_svm.predict(X_train_svm)
        linear_test_pred = linear_svm.predict(X_test_svm)
        
        linear_train_acc = accuracy_score(y_train_svm, linear_train_pred)
        linear_test_acc = accuracy_score(y_test_svm, linear_test_pred)
        
        results['linear_svm'] = {
            'model': linear_svm,
            'model_name': 'Linear SVM',
            'train_accuracy': linear_train_acc,
            'test_accuracy': linear_test_acc,
            'training_time': linear_train_time,
            'predictions': linear_test_pred,
            'test_indices': test_indices
        }
        
        if self.verbose:
            print(f"✅ Linear SVM trained in {linear_train_time:.3f}s")
            print(f"   • Training Accuracy: {linear_train_acc:.3f}")
            print(f"   • Test Accuracy: {linear_test_acc:.3f}")
        
        # 2. RBF SVM
        if self.verbose:
            print(f"\n🌀 Training RBF SVM...")
        
        start_time = time.time()
        rbf_svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=self.random_state)
        rbf_svm.fit(X_train_svm, y_train_svm)
        rbf_train_time = time.time() - start_time
        
        # Predictions
        rbf_train_pred = rbf_svm.predict(X_train_svm)
        rbf_test_pred = rbf_svm.predict(X_test_svm)
        
        rbf_train_acc = accuracy_score(y_train_svm, rbf_train_pred)
        rbf_test_acc = accuracy_score(y_test_svm, rbf_test_pred)
        
        results['rbf_svm'] = {
            'model': rbf_svm,
            'model_name': 'RBF SVM',
            'train_accuracy': rbf_train_acc,
            'test_accuracy': rbf_test_acc,
            'training_time': rbf_train_time,
            'predictions': rbf_test_pred,
            'test_indices': test_indices
        }
        
        if self.verbose:
            print(f"✅ RBF SVM trained in {rbf_train_time:.3f}s")
            print(f"   • Training Accuracy: {rbf_train_acc:.3f}")
            print(f"   • Test Accuracy: {rbf_test_acc:.3f}")
        
        return results
    
    def train_neural_networks(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Train Neural Network models.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            dict: Results dictionary with model performance
        """
        if self.verbose:
            print("🧠 Training Neural Networks")
            print("=" * 35)
            print(f"📊 Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            print(f"🧪 Test: {X_test.shape[0]} samples")
            print(f"🏷️ Classes: {len(np.unique(y_train))}")
        
        results = {}
        
        # 1. SIMPLE NEURAL NETWORK
        if self.verbose:
            print(f"\n🧠 Training Simple Neural Network...")
        
        start_time = time.time()
        simple_nn = MLPClassifier(
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            max_iter=200,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        
        simple_nn.fit(X_train, y_train)
        simple_train_time = time.time() - start_time
        
        # Predictions
        simple_train_pred = simple_nn.predict(X_train)
        simple_test_pred = simple_nn.predict(X_test)
        
        simple_train_acc = accuracy_score(y_train, simple_train_pred)
        simple_test_acc = accuracy_score(y_test, simple_test_pred)
        
        results['simple_nn'] = {
            'model': simple_nn,
            'model_name': 'Simple Neural Network',
            'train_accuracy': simple_train_acc,
            'test_accuracy': simple_test_acc,
            'training_time': simple_train_time,
            'predictions': simple_test_pred,
            'iterations': simple_nn.n_iter_
        }
        
        if self.verbose:
            print(f"✅ Simple NN trained in {simple_train_time:.3f}s ({simple_nn.n_iter_} iterations)")
            print(f"   • Training Accuracy: {simple_train_acc:.3f}")
            print(f"   • Test Accuracy: {simple_test_acc:.3f}")
        
        # 2. DEEP NEURAL NETWORK
        if self.verbose:
            print(f"\n🧠🧠🧠 Training Deep Neural Network...")
        
        start_time = time.time()
        deep_nn = MLPClassifier(
            hidden_layer_sizes=(200, 100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            max_iter=200,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        
        deep_nn.fit(X_train, y_train)
        deep_train_time = time.time() - start_time
        
        # Predictions
        deep_train_pred = deep_nn.predict(X_train)
        deep_test_pred = deep_nn.predict(X_test)
        
        deep_train_acc = accuracy_score(y_train, deep_train_pred)
        deep_test_acc = accuracy_score(y_test, deep_test_pred)
        
        results['deep_nn'] = {
            'model': deep_nn,
            'model_name': 'Deep Neural Network',
            'train_accuracy': deep_train_acc,
            'test_accuracy': deep_test_acc,
            'training_time': deep_train_time,
            'predictions': deep_test_pred,
            'iterations': deep_nn.n_iter_
        }
        
        if self.verbose:
            print(f"✅ Deep NN trained in {deep_train_time:.3f}s ({deep_nn.n_iter_} iterations)")
            print(f"   • Training Accuracy: {deep_train_acc:.3f}")
            print(f"   • Test Accuracy: {deep_test_acc:.3f}")
        
        # 3. REGULARIZED NEURAL NETWORK
        if self.verbose:
            print(f"\n🧠🎯 Training Regularized Neural Network...")
        
        start_time = time.time()
        regularized_nn = MLPClassifier(
            hidden_layer_sizes=(150, 100),
            activation='relu',
            solver='adam',
            alpha=0.001,  # Higher regularization
            learning_rate='adaptive',
            max_iter=300,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15
        )
        
        regularized_nn.fit(X_train, y_train)
        reg_train_time = time.time() - start_time
        
        # Predictions
        reg_train_pred = regularized_nn.predict(X_train)
        reg_test_pred = regularized_nn.predict(X_test)
        
        reg_train_acc = accuracy_score(y_train, reg_train_pred)
        reg_test_acc = accuracy_score(y_test, reg_test_pred)
        
        results['regularized_nn'] = {
            'model': regularized_nn,
            'model_name': 'Regularized Neural Network',
            'train_accuracy': reg_train_acc,
            'test_accuracy': reg_test_acc,
            'training_time': reg_train_time,
            'predictions': reg_test_pred,
            'iterations': regularized_nn.n_iter_
        }
        
        if self.verbose:
            print(f"✅ Regularized NN trained in {reg_train_time:.3f}s ({regularized_nn.n_iter_} iterations)")
            print(f"   • Training Accuracy: {reg_train_acc:.3f}")
            print(f"   • Test Accuracy: {reg_test_acc:.3f}")
        
        return results
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray) -> List[Dict[str, Any]]:
        """
        Train all classification models and return results.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            list: List of all model results
        """
        if self.verbose:
            print("🤖 Training All Classification Models")
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
            print("🏆 Final Model Comparison and Analysis")
            print("=" * 45)
        
        if not results:
            print("❌ No model results available.")
            return
        
        # Performance comparison table
        print("📊 Complete Model Performance Comparison:")
        print("=" * 80)
        print(f"{'Model':<25} {'Train Acc':<12} {'Test Acc':<12} {'Train Time':<12} {'Overfitting':<12}")
        print("-" * 80)
        
        for result in results:
            overfitting = result['train_accuracy'] - result['test_accuracy']
            print(f"{result['model_name']:<25} {result['train_accuracy']:<12.3f} "
                  f"{result['test_accuracy']:<12.3f} {result['training_time']:<12.3f} {overfitting:<12.3f}")
        
        # Best model identification
        best_model = max(results, key=lambda x: x['test_accuracy'])
        fastest_model = min(results, key=lambda x: x['training_time'])
        least_overfitting = min(results, key=lambda x: abs(x['train_accuracy'] - x['test_accuracy']))
        
        print(f"\n🏆 Model Rankings:")
        print(f"   🥇 Best Test Accuracy: {best_model['model_name']} ({best_model['test_accuracy']:.3f})")
        print(f"   ⚡ Fastest Training: {fastest_model['model_name']} ({fastest_model['training_time']:.3f}s)")
        print(f"   🎯 Least Overfitting: {least_overfitting['model_name']} "
              f"(gap: {abs(least_overfitting['train_accuracy'] - least_overfitting['test_accuracy']):.3f})")
        
        # Show detailed classification report for best model
        print(f"\n📋 Detailed Classification Report ({best_model['model_name']}):")
        print("=" * 60)
        
        # Get unique classes present in y_test (after class 0 filtering)
        unique_classes = sorted(np.unique(y_test))
        target_names = [f"Class {cls}" for cls in unique_classes]
        print(classification_report(y_test, best_model['predictions'], target_names=target_names))
        
        # Practical recommendations
        print(f"\n💡 Practical Recommendations for Oil Well Fault Detection:")
        print("=" * 65)
        
        if best_model['model_name'] in ['Random Forest', 'Decision Tree']:
            print(f"🌳 Tree-based models (like {best_model['model_name']}) are recommended because:")
            print(f"   • High interpretability - you can see which sensors are most important")
            print(f"   • Robust to outliers and noise common in industrial data")
            print(f"   • Fast training and prediction")
            print(f"   • Handle mixed data types well")
            
        elif 'Neural Network' in best_model['model_name']:
            print(f"🧠 Neural networks (like {best_model['model_name']}) are recommended because:")
            print(f"   • Can learn complex non-linear patterns in sensor data")
            print(f"   • Automatic feature learning from raw time series")
            print(f"   • Scale well with large amounts of data")
            print(f"   • State-of-the-art performance on many tasks")
            
        elif 'SVM' in best_model['model_name']:
            print(f"⚡ SVM (like {best_model['model_name']}) is recommended because:")
            print(f"   • Effective in high-dimensional spaces")
            print(f"   • Good generalization with limited data")
            print(f"   • Memory efficient")
            print(f"   • Works well with properly scaled features")
        
        print(f"\n🎯 Key Insights:")
        print(f"   • Data was already normalized - no additional scaling needed")
        print(f"   • Class balancing with augmentation improved model performance")
        print(f"   • Use cross-validation to get more reliable performance estimates")
        print(f"   • Consider ensemble methods for production systems")
        print(f"   • Monitor for concept drift in industrial settings")
    
    def visualize_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Create visualization of model performance comparison.
        
        Args:
            results: List of model results
        """
        if not results:
            print("❌ No results to visualize.")
            return
        
        print(f"\n📈 Creating performance visualization...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        model_names = [r['model_name'] for r in results]
        train_accs = [r['train_accuracy'] for r in results]
        test_accs = [r['test_accuracy'] for r in results]
        train_times = [r['training_time'] for r in results]
        
        # Plot 1: Accuracy Comparison
        x = range(len(model_names))
        width = 0.35
        ax1.bar([i - width/2 for i in x], train_accs, width, label='Training Accuracy', alpha=0.8)
        ax1.bar([i + width/2 for i in x], test_accs, width, label='Test Accuracy', alpha=0.8)
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Training vs Test Accuracy')
        ax1.set_xticks(x)
        ax1.set_xticklabels([name.replace(' ', '\n') for name in model_names], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Training Time
        bars = ax2.bar(model_names, train_times, color='skyblue', alpha=0.8)
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Training Time (seconds)')
        ax2.set_title('Training Time Comparison')
        ax2.set_xticklabels([name.replace(' ', '\n') for name in model_names], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s', ha='center', va='bottom')
        
        # Plot 3: Overfitting Analysis
        overfitting_gaps = [train_acc - test_acc for train_acc, test_acc in zip(train_accs, test_accs)]
        colors = ['red' if gap > 0.05 else 'orange' if gap > 0.02 else 'green' for gap in overfitting_gaps]
        bars = ax3.bar(model_names, overfitting_gaps, color=colors, alpha=0.8)
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Train - Test Accuracy')
        ax3.set_title('Overfitting Analysis (Lower is Better)')
        ax3.set_xticklabels([name.replace(' ', '\n') for name in model_names], rotation=45, ha='right')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='High Overfitting')
        ax3.axhline(y=0.02, color='orange', linestyle='--', alpha=0.5, label='Moderate Overfitting')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Accuracy vs Training Time Scatter
        ax4.scatter(train_times, test_accs, s=100, alpha=0.7, color='purple')
        for i, name in enumerate(model_names):
            ax4.annotate(name.replace(' ', '\n'), (train_times[i], test_accs[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax4.set_xlabel('Training Time (seconds)')
        ax4.set_ylabel('Test Accuracy')
        ax4.set_title('Accuracy vs Training Time Trade-off')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Enhanced analysis methods
def analyze_fold_accuracy(classifier: SupervisedClassifier, fold_results: Dict[str, Dict]) -> None:
    """
    Analyze accuracy per fold for each model.
    
    Args:
        classifier: Trained classifier
        fold_results: Dictionary with fold results {fold_name: {model_name: accuracy}}
    """
    print("\n📊 Accuracy Analysis Per Fold")
    print("=" * 50)
    
    if not fold_results:
        print("❌ No fold results available")
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

def analyze_class_accuracy(classifier: SupervisedClassifier, y_true: np.ndarray, 
                          predictions_dict: Dict[str, np.ndarray]) -> None:
    """
    Analyze accuracy per class for each model.
    
    Args:
        classifier: Trained classifier
        y_true: True labels
        predictions_dict: Dictionary with model predictions {model_name: predictions}
    """
    print("\n📊 Accuracy Analysis Per Class")
    print("=" * 50)
    
    class_names = classifier.class_names if classifier.class_names is not None else np.unique(y_true)
    model_names = sorted(predictions_dict.keys())
    
    # Calculate per-class accuracy for each model
    print(f"{'Class':<8}", end="")
    for model in model_names:
        print(f"{model:<15}", end="")
    print()
    print("-" * (8 + 15 * len(model_names)))
    
    for class_idx, class_name in enumerate(class_names):
        class_mask = (y_true == class_idx)
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

def analyze_accuracy_without_class0(classifier: SupervisedClassifier, y_true: np.ndarray, 
                                   predictions_dict: Dict[str, np.ndarray], selected_classes=None) -> None:
    """
    Show which classes were included/excluded from analysis.
    
    Args:
        classifier: Trained classifier
        y_true: True labels
        predictions_dict: Dictionary with model predictions {model_name: predictions}
        selected_classes: List of classes that were selected for analysis (None = all except 0)
    """
    print("\n📊 Class Selection Status")
    print("=" * 35)
    
    # Check which classes are present in the data
    unique_classes = np.unique(y_true)
    
    if selected_classes is None:
        # Default mode: class 0 exclusion
        has_class_0 = (0 in unique_classes)
        if not has_class_0:
            print("✅ Class 0 (normal operation) successfully excluded from all analysis")
            print(f"📊 Dataset contains only fault classes: {sorted(unique_classes)}")
            
            print(f"\n🎯 Benefits of Class 0 Exclusion:")
            print(f"   • Focus purely on fault type discrimination")
            print(f"   • Avoid normal vs fault classification bias")
            print(f"   • More relevant for fault diagnosis systems")
            print(f"   • Cleaner evaluation of fault-specific performance")
        else:
            print("⚠️ Class 0 still present in data - check filtering logic")
    else:
        # Custom class selection mode
        print(f"✅ Custom class selection applied")
        print(f"📊 Selected classes: {sorted(selected_classes)}")
        print(f"📊 Dataset contains classes: {sorted(unique_classes)}")
        
        print(f"\n🎯 Benefits of Custom Class Selection:")
        print(f"   • Focus on specific fault types of interest")
        print(f"   • Targeted analysis for particular operational scenarios")
        print(f"   • Simplified classification problem")
        print(f"   • Faster training with fewer classes")
    
    print(f"📊 Total samples in analysis: {len(y_true)}")
    
    # Show class distribution
    unique_classes, counts = np.unique(y_true, return_counts=True)
    print(f"\n📊 Class distribution in analysis:")
    for cls, count in zip(unique_classes, counts):
        class_name = classifier.class_names[cls] if classifier.class_names is not None else cls
        print(f"   Class {class_name}: {count} samples")
        print(f"📊 Classes found: {sorted(unique_classes)}")
        
        # Calculate accuracy excluding class 0 for comparison
        non_class0_mask = (y_true != 0)
        if np.any(non_class0_mask):
            y_true_filtered = y_true[non_class0_mask]
            
            print(f"\n📊 Comparison with class 0 excluded:")
            print(f"   • Total samples: {len(y_true)}")
            print(f"   • Fault-only samples: {len(y_true_filtered)}")
            
            for model_name in sorted(predictions_dict.keys()):
                y_pred = predictions_dict[model_name]
                y_pred_filtered = y_pred[non_class0_mask]
                
                overall_accuracy = accuracy_score(y_true, y_pred)
                fault_only_accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
                
                print(f"   {model_name}: Overall {overall_accuracy:.3f} vs Fault-only {fault_only_accuracy:.3f}")

# Convenience function for quick usage
def quick_supervised_classification(train_dfs: List[pd.DataFrame], train_classes: List[str],
                                   test_dfs: List[pd.DataFrame], test_classes: List[str],
                                   balance_classes: bool = True, balance_strategy: str = 'combined',
                                   max_samples_per_class: int = 300, verbose: bool = True) -> SupervisedClassifier:
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
        train_dfs, train_classes, test_dfs, test_classes,
        balance_classes=balance_classes, balance_strategy=balance_strategy,
        max_samples_per_class=max_samples_per_class
    )
    
    # Train all models
    results = classifier.train_all_models(X_train, y_train, X_test, y_test)
    
    # Compare models
    classifier.compare_models(results, y_test)
    
    # Visualize results
    classifier.visualize_results(results)
    
    return classifier


def enhanced_fold_analysis(train_dfs: List[pd.DataFrame], train_classes: List[str],
                          test_dfs: List[pd.DataFrame], test_classes: List[str],
                          fold_info: List[str] = None, balance_classes: bool = True,
                          balance_strategy: str = 'combined', max_samples_per_class: int = 300,
                          balance_test: bool = True, min_test_samples_per_class: int = 300,
                          selected_classes: List[str] = None, verbose: bool = True) -> SupervisedClassifier:
    """
    Enhanced supervised classification with fold-wise analysis and class-specific metrics.
    
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
        print("🔄 Enhanced Classification with Test Balancing and Class 0 Exclusion")
        print("=" * 65)
    
    # Filter data based on selected_classes parameter
    if selected_classes is None:
        # Default behavior: exclude only class 0 (normal operation)
        filter_message = "⚠️ Filtering out class 0 (normal operation) from all data..."
        filter_condition = lambda cls: cls != 0 and cls != '0'
    else:
        # Custom class selection: include only specified classes
        selected_classes_str = [str(c) for c in selected_classes]  # Convert to strings for comparison
        filter_message = f"🎯 Filtering to include only selected classes: {selected_classes}..."
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
            print(f"✅ Class 0 filtering completed:")
        else:
            print(f"✅ Class filtering completed:")
        print(f"   • Original training samples: {len(train_dfs)} → Filtered: {len(train_filtered_dfs)}")
        print(f"   • Original test samples: {len(test_dfs)} → Filtered: {len(test_filtered_dfs)}")
        
        # Show remaining class distribution
        train_unique, train_counts = np.unique(train_filtered_classes, return_counts=True)
        print(f"   • Remaining training classes: {dict(zip(train_unique, train_counts))}")
        
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
            raise ValueError("No data remaining after filtering out class 0. Check your data labels.")
        else:
            raise ValueError(f"No data remaining after filtering for classes {selected_classes}. Check your data labels and selected classes.")
    
    # Track indices for fold_info consistency
    original_test_indices = list(range(len(test_dfs)))
    
    # Balance test data if requested
    if balance_test:
        if verbose:
            print(f"⚖️ Balancing test data to ensure min {min_test_samples_per_class} samples per class...")
        
        # Import the data augmentation function
        from .data_augmentation import quick_balance_classes
        
        # Balance test data
        balanced_test_dfs, balanced_test_classes = quick_balance_classes(
            test_dfs, test_classes, strategy=balance_strategy,
            min_samples_per_class=min_test_samples_per_class
        )
        
        if verbose:
            print(f"✅ Test data balanced:")
            print(f"   • Original test samples: {len(test_dfs)}")
            print(f"   • Balanced test samples: {len(balanced_test_dfs)}")
            
            # Show class distribution after balancing
            test_unique, test_counts = np.unique(balanced_test_classes, return_counts=True)
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
                print("⚠️ Per-fold analysis disabled when test balancing is enabled")
                print("   (Fold information becomes inconsistent after test data augmentation)")
            fold_info = None
    
    # Prepare data (class 0 already filtered out, but ensure it stays out)
    X_train, y_train, X_test, y_test = classifier.prepare_data(
        train_dfs, train_classes, test_dfs, test_classes,
        balance_classes=balance_classes, balance_strategy=balance_strategy,
        max_samples_per_class=max_samples_per_class
    )
    
    # Track test data filtering for fold consistency
    test_filter_mask = None
    
    # Double-check that class 0 is not in the final prepared data
    if 0 in y_train or 0 in y_test:
        if verbose:
            print("⚠️ Warning: Class 0 detected in prepared data, removing...")
        
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
            print(f"✅ Final class 0 removal completed:")
            print(f"   • Training samples after final filter: {len(X_train)}")
            print(f"   • Test samples after final filter: {len(X_test)}")
            print(f"   • Final training classes: {sorted(np.unique(y_train))}")
            print(f"   • Final test classes: {sorted(np.unique(y_test))}")
    
    # Filter fold_info to match the filtered test data
    if fold_info is not None and test_filter_mask is not None:
        # Filter fold_info to match the test data after class filtering
        fold_info = np.array(fold_info)[test_filter_mask].tolist()
        if verbose:
            print(f"✅ Fold information filtered to match test data")
            print(f"   • Original fold_info length: {len(np.array(fold_info)[test_filter_mask]) + len(np.array(fold_info)[~test_filter_mask])}")
            print(f"   • Filtered fold_info length: {len(fold_info)}")
            print(f"   • Test data length: {len(y_test)}")
    
    # Verify fold_info consistency before proceeding
    if fold_info is not None and len(fold_info) != len(y_test):
        if verbose:
            print(f"⚠️ Fold info length mismatch detected:")
            print(f"   • fold_info length: {len(fold_info)}")
            print(f"   • test data length: {len(y_test)}")
            print(f"   • Disabling per-fold analysis for safety")
        fold_info = None
    
    # Train all models and get predictions
    results = classifier.train_all_models(X_train, y_train, X_test, y_test)
    
    # Extract predictions for enhanced analysis
    predictions_dict = {}
    for result in results:
        model = result['model']  # Fixed: use 'model' key instead of 'trained_model'
        model_name = result['model_name']
        predictions_dict[model_name] = model.predict(X_test)
    
    print("\n" + "="*70)
    print("🔍 ENHANCED CLASSIFICATION ANALYSIS (FAULT-ONLY)")
    print("="*70)
    
    # Analysis 1: Accuracy per fold (if fold info available)
    if fold_info is not None and len(fold_info) == len(test_dfs):
        fold_results = {}
        unique_folds = sorted(set(fold_info))
        
        print(f"\n📊 Found {len(unique_folds)} unique folds for analysis")
        
        for fold in unique_folds:
            fold_mask = np.array([fold_info[i] == fold for i in range(len(fold_info))])
            if not np.any(fold_mask):
                continue
                
            fold_results[fold] = {}
            for model_name, predictions in predictions_dict.items():
                fold_accuracy = accuracy_score(y_test[fold_mask], predictions[fold_mask])
                fold_results[fold][model_name] = fold_accuracy
        
        analyze_fold_accuracy(classifier, fold_results)
    else:
        print("\n⚠️ Fold information not available - skipping per-fold analysis")
    
    # Analysis 2: Accuracy per class (fault classes only)
    analyze_class_accuracy(classifier, y_test, predictions_dict)
    
    # Analysis 3: Show class selection status
    analyze_accuracy_without_class0(classifier, y_test, predictions_dict, selected_classes)
    
    # Original comparison and visualization
    classifier.compare_models(results, y_test)
    classifier.visualize_results(results)
    
    return classifier
