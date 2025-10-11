"""
Cross-Validation Utilities for 3W Dataset

This module provides specialized cross-validation methods for the 3W dataset,
implementing a smart train-test split strategy where:
- Training uses ALL available data (real + simulated) for better model learning
- Testing uses ONLY real data for realistic evaluation
- Cross-Validation creates multiple folds with proper isolation
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional


class CrossValidator:
    """
    Cross-validation utilities specialized for 3W dataset with real/simulated data separation.
    """

    def __init__(self, n_folds: int = 3, random_state: int = 42, verbose: bool = True):
        """
        Initialize CrossValidator.

        Args:
            n_folds (int): Number of cross-validation folds
            random_state (int): Random state for reproducibility
            verbose (bool): Whether to print detailed information
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.verbose = verbose

    def separate_real_simulated_data(
        self,
        dfs: List,
        classes: List,
        loader=None,
        fallback_real_proportion: float = 0.7,
    ) -> Tuple[List, List, List, List]:
        """
        Separate real and simulated data from the loaded datasets.

        Args:
            dfs (List): List of dataframes
            classes (List): List of corresponding classes
            loader: DataLoader instance (optional, for enhanced separation)
            fallback_real_proportion (float): Fallback proportion for real data if loader separation fails

        Returns:
            tuple: (real_dfs, real_classes, simulated_dfs, simulated_classes)
        """
        if self.verbose:
            print("🔍 Separating Real vs Simulated Data")
            print("-" * 50)

        try:
            # Try to use enhanced separation from loader
            if loader and hasattr(loader, "get_separated_data"):
                separated_data = loader.get_separated_data()

                # Extract real data info
                real_data_info = separated_data["real"]
                simulated_data_info = separated_data["simulated"]

                # Calculate proportional split
                total_real_original = len(real_data_info["dfs"])
                total_sim_original = len(simulated_data_info["dfs"])
                total_normalized = len(dfs)

                real_proportion = total_real_original / (
                    total_real_original + total_sim_original
                )
                estimated_real_count = int(total_normalized * real_proportion)

                # Split based on proportion
                real_dfs = dfs[:estimated_real_count]
                real_classes = classes[:estimated_real_count]
                simulated_dfs = dfs[estimated_real_count:]
                simulated_classes = classes[estimated_real_count:]

                if self.verbose:
                    print(f"📊 Enhanced Separation Results:")
                    print(
                        f"   Original - Real: {total_real_original}, Simulated: {total_sim_original}"
                    )
                    print(
                        f"   Processed - Real: {len(real_dfs)}, Simulated: {len(simulated_dfs)}"
                    )
                    print(f"   Real proportion: {real_proportion:.2%}")

            else:
                raise ValueError("Enhanced separation not available")

        except Exception as e:
            if self.verbose:
                print(f"⚠️ Could not use enhanced separation: {e}")
                print("   Using fallback method...")

            # Fallback: Use simple heuristic split
            total_samples = len(dfs)
            real_cutoff = int(total_samples * fallback_real_proportion)

            real_dfs = dfs[:real_cutoff] if real_cutoff > 0 else dfs
            real_classes = classes[:real_cutoff] if real_cutoff > 0 else classes
            simulated_dfs = dfs[real_cutoff:] if real_cutoff < total_samples else []
            simulated_classes = (
                classes[real_cutoff:] if real_cutoff < total_samples else []
            )

            if self.verbose:
                print(f"📊 Fallback Separation Results:")
                print(f"   Real data samples: {len(real_dfs)}")
                print(f"   Simulated data samples: {len(simulated_dfs)}")

        if self.verbose:
            print(f"   Real data classes: {dict(Counter(real_classes))}")
            print(f"   Simulated data classes: {dict(Counter(simulated_classes))}")

        return real_dfs, real_classes, simulated_dfs, simulated_classes

    def create_cv_folds(
        self,
        real_dfs: List,
        real_classes: List,
        simulated_dfs: List,
        simulated_classes: List,
    ) -> List[Dict[str, Any]]:
        """
        Create cross-validation folds with smart train-test split strategy.

        Args:
            real_dfs (List): Real data dataframes
            real_classes (List): Real data classes
            simulated_dfs (List): Simulated data dataframes
            simulated_classes (List): Simulated data classes

        Returns:
            List[Dict]: List of fold dictionaries containing train/test splits
        """
        if self.verbose:
            print(f"\n🎯 Creating {self.n_folds}-Fold Cross-Validation Splits")
            print("-" * 50)

        # Validate input
        if len(real_dfs) < self.n_folds:
            raise ValueError(
                f"Not enough real data samples for {self.n_folds}-fold cross-validation. "
                f"Available: {len(real_dfs)}, Required: {self.n_folds}"
            )

        # Set up cross-validation strategy
        real_indices = np.arange(len(real_dfs))
        real_classes_array = np.array(real_classes)

        # Check if we can use stratified splitting
        class_counts = Counter(real_classes)
        min_class_count = min(class_counts.values()) if class_counts else 0

        if min_class_count < self.n_folds:
            if self.verbose:
                print(
                    f"⚠️ Warning: Some classes have fewer than {self.n_folds} samples."
                )
                print(f"   Minimum class count: {min_class_count}")
                print("   Using regular KFold instead of StratifiedKFold")
            cv_splitter = KFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.random_state
            )
            splits = cv_splitter.split(real_indices)
        else:
            cv_splitter = StratifiedKFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.random_state
            )
            splits = cv_splitter.split(real_indices, real_classes_array)

        # Create folds
        cv_folds = []

        if self.verbose:
            print(f"🔄 Creating {self.n_folds} cross-validation folds...")
            print("=" * 40)

        for fold_idx, (train_real_idx, test_real_idx) in enumerate(splits):
            fold_num = fold_idx + 1

            if self.verbose:
                print(f"\n📁 FOLD {fold_num}:")
                print("-" * 20)

            # Test set: Only real data from this fold
            fold_test_dfs = [real_dfs[i] for i in test_real_idx]
            fold_test_classes = [real_classes[i] for i in test_real_idx]

            # Training set: Real data from other folds + ALL simulated data
            fold_train_dfs = [real_dfs[i] for i in train_real_idx] + simulated_dfs
            fold_train_classes = [
                real_classes[i] for i in train_real_idx
            ] + simulated_classes

            # Calculate statistics
            train_real_count = len(train_real_idx)
            train_sim_count = len(simulated_dfs)
            test_count = len(test_real_idx)

            # Calculate split ratios
            total_real = len(real_dfs)
            train_ratio = train_real_count / total_real * 100
            test_ratio = test_count / total_real * 100

            if self.verbose:
                print(f"   Training Set:")
                print(f"     • Real data: {train_real_count} samples")
                print(f"     • Simulated data: {train_sim_count} samples")
                print(f"     • Total training: {len(fold_train_dfs)} samples")
                print(f"     • Training classes: {dict(Counter(fold_train_classes))}")

                print(f"   Test Set (Real Only):")
                print(f"     • Real data: {test_count} samples")
                print(f"     • Test classes: {dict(Counter(fold_test_classes))}")

                print(f"   Real Data Split Ratio:")
                print(f"     • Train: {train_ratio:.1f}% of real data")
                print(f"     • Test: {test_ratio:.1f}% of real data")

            # Store fold information
            fold_info = {
                "fold_number": fold_num,
                "train_dfs": fold_train_dfs,
                "train_classes": fold_train_classes,
                "test_dfs": fold_test_dfs,
                "test_classes": fold_test_classes,
                "train_real_count": train_real_count,
                "train_sim_count": train_sim_count,
                "test_count": test_count,
                "train_ratio": train_ratio,
                "test_ratio": test_ratio,
                "train_real_indices": train_real_idx,
                "test_real_indices": test_real_idx,
            }

            cv_folds.append(fold_info)

        return cv_folds

    def print_cv_summary(
        self, cv_folds: List[Dict[str, Any]], real_dfs: List, simulated_dfs: List
    ) -> None:
        """
        Print comprehensive summary of cross-validation setup.

        Args:
            cv_folds (List[Dict]): Cross-validation folds
            real_dfs (List): Real data dataframes
            simulated_dfs (List): Simulated data dataframes
        """
        print(f"\n📊 Cross-Validation Summary:")
        print("=" * 50)

        total_real_samples = len(real_dfs)
        total_sim_samples = len(simulated_dfs)
        avg_train_real = np.mean([fold["train_real_count"] for fold in cv_folds])
        avg_test_real = np.mean([fold["test_count"] for fold in cv_folds])

        print(f"Dataset Overview:")
        print(f"   • Total real samples: {total_real_samples}")
        print(f"   • Total simulated samples: {total_sim_samples}")
        print(f"   • Number of folds: {len(cv_folds)}")

        print(f"\nPer-Fold Averages:")
        print(f"   • Avg real samples for training: {avg_train_real:.1f}")
        print(
            f"   • Avg simulated samples for training: {total_sim_samples} (constant)"
        )
        print(f"   • Avg real samples for testing: {avg_test_real:.1f}")
        print(
            f"   • Avg train/test ratio: {avg_train_real/total_real_samples*100:.1f}% / {avg_test_real/total_real_samples*100:.1f}%"
        )

        # Validation
        print(f"\nValidation:")
        all_test_indices = []
        for fold in cv_folds:
            all_test_indices.extend(fold["test_real_indices"])

        unique_test_indices = set(all_test_indices)
        print(
            f"   • Test coverage: {len(all_test_indices)} total assignments across folds"
        )
        print(f"   • Unique test samples: {len(unique_test_indices)}")
        print(
            f"   • Each real sample tested exactly once: {'✅' if len(unique_test_indices) == total_real_samples else '❌'}"
        )

        print(f"\n💾 Variables Created:")
        print(
            f"   • cv_folds: List of {len(cv_folds)} cross-validation fold dictionaries"
        )
        print(
            f"   • Each fold contains train_dfs, test_dfs, train_classes, test_classes"
        )

        print(
            f"\n🎯 Ready for {len(cv_folds)}-fold cross-validation training and evaluation!"
        )

    def get_fold_data(
        self, fold_number: int, cv_folds: List[Dict[str, Any]]
    ) -> Tuple[List, List, List, List]:
        """
        Get train/test data for a specific fold.

        Args:
            fold_number (int): Fold number (1-indexed)
            cv_folds (List[Dict]): List of cross-validation folds

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

    def print_fold_summary(
        self, fold_number: int, cv_folds: List[Dict[str, Any]]
    ) -> None:
        """
        Print detailed summary for a specific fold.

        Args:
            fold_number (int): Fold number (1-indexed)
            cv_folds (List[Dict]): List of cross-validation folds
        """
        fold = cv_folds[fold_number - 1]

        print(f"\n📁 FOLD {fold_number} SUMMARY:")
        print("-" * 30)
        print(f"Training Data:")
        print(f"  • Real samples: {fold['train_real_count']}")
        print(f"  • Simulated samples: {fold['train_sim_count']}")
        print(f"  • Total training: {len(fold['train_dfs'])}")
        print(f"  • Class distribution: {dict(Counter(fold['train_classes']))}")

        print(f"Test Data (Real Only):")
        print(f"  • Test samples: {fold['test_count']}")
        print(f"  • Class distribution: {dict(Counter(fold['test_classes']))}")

        # Calculate total data points
        train_points = sum(len(df) for df in fold["train_dfs"])
        test_points = sum(len(df) for df in fold["test_dfs"])

        print(f"Data Points:")
        print(f"  • Training points: {train_points:,}")
        print(f"  • Test points: {test_points:,}")
        print(f"  • Total points: {train_points + test_points:,}")

    def compare_all_folds(self, cv_folds: List[Dict[str, Any]]) -> None:
        """
        Compare statistics across all folds in a tabular format.

        Args:
            cv_folds (List[Dict]): List of cross-validation folds
        """
        print(f"\n📊 COMPARISON ACROSS ALL {len(cv_folds)} FOLDS:")
        print("=" * 50)

        print(
            f"{'Fold':<6} {'Train Real':<12} {'Train Sim':<12} {'Test Real':<12} {'Train%':<10} {'Test%':<10}"
        )
        print("-" * 70)

        for i, fold in enumerate(cv_folds):
            fold_num = i + 1
            train_real = fold["train_real_count"]
            train_sim = fold["train_sim_count"]
            test_real = fold["test_count"]
            train_pct = fold["train_ratio"]
            test_pct = fold["test_ratio"]

            print(
                f"{fold_num:<6} {train_real:<12} {train_sim:<12} {test_real:<12} {train_pct:<9.1f}% {test_pct:<9.1f}%"
            )

    def setup_complete_cv(
        self, dfs: List, classes: List, loader=None
    ) -> Tuple[List[Dict[str, Any]], Tuple[List, List, List, List]]:
        """
        Complete cross-validation setup in one method.

        Args:
            dfs (List): All dataframes (normalized/processed)
            classes (List): All corresponding classes
            loader: DataLoader instance (optional)

        Returns:
            tuple: (cv_folds, (real_dfs, real_classes, simulated_dfs, simulated_classes))
        """
        # Step 1: Separate real and simulated data
        real_dfs, real_classes, simulated_dfs, simulated_classes = (
            self.separate_real_simulated_data(dfs, classes, loader)
        )

        # Step 2: Create cross-validation folds
        cv_folds = self.create_cv_folds(
            real_dfs, real_classes, simulated_dfs, simulated_classes
        )

        # Step 3: Print summary
        self.print_cv_summary(cv_folds, real_dfs, simulated_dfs)

        return cv_folds, (real_dfs, real_classes, simulated_dfs, simulated_classes)


# Convenience functions for backward compatibility
def get_fold_data(
    fold_number: int, cv_folds: List[Dict[str, Any]]
) -> Tuple[List, List, List, List]:
    """Convenience function - get train/test data for a specific fold."""
    cv = CrossValidator(verbose=False)
    return cv.get_fold_data(fold_number, cv_folds)


def print_fold_summary(fold_number: int, cv_folds: List[Dict[str, Any]]) -> None:
    """Convenience function - print detailed summary for a specific fold."""
    cv = CrossValidator(verbose=False)
    cv.print_fold_summary(fold_number, cv_folds)


def compare_all_folds(cv_folds: List[Dict[str, Any]]) -> None:
    """Convenience function - compare statistics across all folds."""
    cv = CrossValidator(verbose=False)
    cv.compare_all_folds(cv_folds)
