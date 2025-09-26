"""
Data Persistence Utilities for 3W Dataset

This module provides utilities for saving and loading processed data,
cross-validation splits, and metadata for use across multiple notebooks.
"""

import os
import json
import pickle
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import warnings


class DataPersistence:
    """
    Handle saving and loading of processed 3W dataset components.
    """

    def __init__(self, base_dir: str = "processed_data", verbose: bool = True):
        """
        Initialize DataPersistence.

        Args:
            base_dir (str): Base directory for saving processed data
            verbose (bool): Whether to print detailed information
        """
        self.base_dir = base_dir
        self.verbose = verbose
        self.cv_splits_dir = os.path.join(base_dir, "cv_splits")
        self.metadata_dir = os.path.join(base_dir, "metadata")

        # Create directories
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories for data persistence."""
        directories = [
            self.base_dir,
            self.cv_splits_dir,
            self.metadata_dir,
            os.path.join(self.cv_splits_dir, "raw"),
            os.path.join(self.cv_splits_dir, "windowed"),
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        if self.verbose:
            print(f"📁 Created directory structure in: {self.base_dir}")

    def save_cv_folds(
        self,
        cv_folds: List[Dict[str, Any]],
        save_format: str = "pickle",
        compression: str = "snappy",
    ) -> str:
        """
        Save cross-validation folds to disk.

        Args:
            cv_folds (List[Dict]): Cross-validation folds
            save_format (str): File format ('pickle', 'csv', 'parquet')
            compression (str): Compression method for pickle files

        Returns:
            str: Path where data was saved
        """
        if self.verbose:
            print("💾 Saving Cross-Validation Folds")
            print("-" * 50)

        fold_dir = os.path.join(self.cv_splits_dir, "raw")
        saved_files = []

        for fold_idx, fold_info in enumerate(cv_folds):
            fold_num = fold_idx + 1
            fold_path = os.path.join(fold_dir, f"fold_{fold_num}")
            os.makedirs(fold_path, exist_ok=True)

            # Save training data
            train_file = self._save_dataframes(
                fold_info["train_dfs"],
                fold_info["train_classes"],
                os.path.join(fold_path, f"train_data.{save_format}"),
                save_format,
                compression,
            )

            # Save test data
            test_file = self._save_dataframes(
                fold_info["test_dfs"],
                fold_info["test_classes"],
                os.path.join(fold_path, f"test_data.{save_format}"),
                save_format,
                compression,
            )

            # Save fold metadata
            fold_metadata = {
                "fold_number": fold_info["fold_number"],
                "train_real_count": fold_info["train_real_count"],
                "train_sim_count": fold_info["train_sim_count"],
                "test_count": fold_info["test_count"],
                "train_ratio": fold_info["train_ratio"],
                "test_ratio": fold_info["test_ratio"],
                "train_file": train_file,
                "test_file": test_file,
                "save_timestamp": datetime.now().isoformat(),
            }

            metadata_file = os.path.join(fold_path, "fold_metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(fold_metadata, f, indent=2)

            saved_files.extend([train_file, test_file, metadata_file])

            if self.verbose:
                print(f"   📁 Fold {fold_num}:")
                print(
                    f"      • Training: {len(fold_info['train_dfs'])} samples → {train_file}"
                )
                print(
                    f"      • Testing: {len(fold_info['test_dfs'])} samples → {test_file}"
                )
                print(f"      • Metadata: {metadata_file}")

        if self.verbose:
            print(f"\n✅ Saved {len(cv_folds)} cross-validation folds")
            print(f"   Total files created: {len(saved_files)}")

        return fold_dir

    def save_windowed_data(
        self,
        windowing_results: Dict[str, Any],
        save_format: str = "pickle",
        compression: str = "snappy",
    ) -> str:
        """
        Save windowed data to disk.

        Args:
            windowing_results (Dict): Results from windowing process.
                                    Can contain either single fold data or all_windowed_folds
            save_format (str): File format ('pickle', 'csv', 'pickle')
            compression (str): Compression method

        Returns:
            str: Path where data was saved
        """
        if self.verbose:
            print("🪟 Saving Windowed Data")
            print("-" * 50)

        windowed_dir = os.path.join(self.cv_splits_dir, "windowed")

        # Check if this is the new extended format with all windowed folds
        if "all_windowed_folds" in windowing_results:
            return self._save_all_windowed_folds(
                windowing_results, save_format, compression
            )

        # Original format - save single fold windowed data
        return self._save_single_fold_windowed_data(
            windowing_results, save_format, compression
        )

    def _save_all_windowed_folds(
        self,
        windowing_results: Dict[str, Any],
        save_format: str = "pickle",
        compression: str = "snappy",
    ) -> str:
        """Save windowed data for all cross-validation folds."""
        windowed_dir = os.path.join(self.cv_splits_dir, "windowed")
        all_windowed_folds = windowing_results["all_windowed_folds"]

        if self.verbose:
            print(
                f"💾 Saving windowed data for {len(all_windowed_folds)} cross-validation folds"
            )

        # Save each windowed fold separately
        for i, fold_data in enumerate(all_windowed_folds):
            fold_num = fold_data["fold_number"]
            fold_windowed_dir = os.path.join(windowed_dir, f"fold_{fold_num}")
            os.makedirs(fold_windowed_dir, exist_ok=True)

            # Save training windows for this fold
            train_file = self._save_dataframes(
                fold_data["train_windowed_dfs"],
                fold_data["train_windowed_classes"],
                os.path.join(fold_windowed_dir, f"train_windowed.{save_format}"),
                save_format,
                compression,
            )

            # Save test windows for this fold
            test_file = self._save_dataframes(
                fold_data["test_windowed_dfs"],
                fold_data["test_windowed_classes"],
                os.path.join(fold_windowed_dir, f"test_windowed.{save_format}"),
                save_format,
                compression,
            )

            # Save fold windowing metadata
            fold_metadata = {
                "fold_number": fold_num,
                "train_statistics": fold_data["train_statistics"],
                "test_statistics": fold_data["test_statistics"],
                "train_metadata": fold_data.get("train_metadata", []),
                "test_metadata": fold_data.get("test_metadata", []),
                "train_windows_count": len(fold_data["train_windowed_dfs"]),
                "test_windows_count": len(fold_data["test_windowed_dfs"]),
                "save_timestamp": datetime.now().isoformat(),
            }

            fold_metadata_file = os.path.join(
                fold_windowed_dir, "windowing_metadata.json"
            )
            with open(fold_metadata_file, "w") as f:
                json.dump(fold_metadata, f, indent=2)

            if self.verbose:
                train_count = len(fold_data["train_windowed_dfs"])
                test_count = len(fold_data["test_windowed_dfs"])
                print(
                    f"   • Fold {fold_num}: {train_count} train, {test_count} test windows"
                )

        # Save overall windowing summary
        overall_metadata = {
            "total_folds": len(all_windowed_folds),
            "total_train_windows": windowing_results.get("total_train_windows", 0),
            "total_test_windows": windowing_results.get("total_test_windows", 0),
            "windowing_config": windowing_results.get("windowing_config", {}),
            "save_timestamp": datetime.now().isoformat(),
            "fold_details": [
                {
                    "fold_number": fold["fold_number"],
                    "train_windows": len(fold["train_windowed_dfs"]),
                    "test_windows": len(fold["test_windowed_dfs"]),
                }
                for fold in all_windowed_folds
            ],
        }

        overall_metadata_file = os.path.join(
            windowed_dir, "overall_windowing_summary.json"
        )
        with open(overall_metadata_file, "w") as f:
            json.dump(overall_metadata, f, indent=2)

        if self.verbose:
            total_windows = windowing_results.get(
                "total_train_windows", 0
            ) + windowing_results.get("total_test_windows", 0)
            print(
                f"   ✅ Total: {total_windows} windows across {len(all_windowed_folds)} folds"
            )
            print(f"   📊 Summary: {overall_metadata_file}")

        return windowed_dir

    def _save_single_fold_windowed_data(
        self,
        windowing_results: Dict[str, Any],
        save_format: str = "pickle",
        compression: str = "snappy",
    ) -> str:
        """Save windowed data for a single fold (original format)."""
        windowed_dir = os.path.join(self.cv_splits_dir, "windowed")

        # If we get the extended format but need to save as single fold, use default_fold_results
        if "default_fold_results" in windowing_results:
            single_fold_data = windowing_results["default_fold_results"]
        else:
            single_fold_data = windowing_results

        # Ensure we have the required keys
        if "train_windowed_dfs" not in single_fold_data:
            if self.verbose:
                print("⚠️ No windowed data found to save in single fold format")
            return windowed_dir

        # Save windowed training data
        train_windowed_file = self._save_dataframes(
            single_fold_data["train_windowed_dfs"],
            single_fold_data["train_windowed_classes"],
            os.path.join(windowed_dir, f"train_windowed.{save_format}"),
            save_format,
            compression,
        )

        # Save windowed test data
        test_windowed_file = self._save_dataframes(
            single_fold_data["test_windowed_dfs"],
            single_fold_data["test_windowed_classes"],
            os.path.join(windowed_dir, f"test_windowed.{save_format}"),
            save_format,
            compression,
        )

        # Save windowing metadata
        windowing_metadata = {
            "train_statistics": single_fold_data["train_statistics"],
            "test_statistics": single_fold_data["test_statistics"],
            "train_metadata": single_fold_data.get("train_metadata", []),
            "test_metadata": single_fold_data.get("test_metadata", []),
            "windowing_params": single_fold_data.get("windowing_params", {}),
            "save_timestamp": datetime.now().isoformat(),
        }

        metadata_file = os.path.join(windowed_dir, "windowing_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(windowing_metadata, f, indent=2)

        if self.verbose:
            print(
                f"   • Training windows: {len(single_fold_data['train_windowed_dfs'])} → {train_windowed_file}"
            )
            print(
                f"   • Test windows: {len(single_fold_data['test_windowed_dfs'])} → {test_windowed_file}"
            )
            print(f"   • Metadata: {metadata_file}")

        return windowed_dir

    def save_processing_summary(self, summary_data: Dict[str, Any]) -> str:
        """
        Save complete processing summary and metadata.

        Args:
            summary_data (Dict): Summary of all processing steps

        Returns:
            str: Path to summary file
        """
        summary_file = os.path.join(self.metadata_dir, "processing_summary.json")

        # Add timestamp and system info
        summary_data.update(
            {
                "processing_timestamp": datetime.now().isoformat(),
                "save_location": self.base_dir,
                "format_version": "1.0",
            }
        )

        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=2)

        if self.verbose:
            print(f"📄 Processing summary saved: {summary_file}")

        return summary_file

    def _save_dataframes(
        self,
        dfs: List[pd.DataFrame],
        classes: List[str],
        filepath: str,
        save_format: str,
        compression: str,
    ) -> str:
        """
        Save list of dataframes with their classes to a single file.

        Args:
            dfs (List[pd.DataFrame]): List of dataframes
            classes (List[str]): List of corresponding classes
            filepath (str): Output file path
            save_format (str): File format
            compression (str): Compression method

        Returns:
            str: Path to saved file
        """
        if save_format.lower() == "pickle":
            return self._save_as_pickle(dfs, classes, filepath)
        elif save_format.lower() == "csv":
            return self._save_as_csv(dfs, classes, filepath)
        else:
            raise ValueError(f"Unsupported save format: {save_format}")


    def _save_as_csv(
        self, dfs: List[pd.DataFrame], classes: List[str], filepath: str
    ) -> str:
        """Save as CSV format."""
        combined_data = []
        for i, (df, class_label) in enumerate(zip(dfs, classes)):
            df_copy = df.copy()
            df_copy["sample_id"] = i
            df_copy["class_label"] = class_label
            combined_data.append(df_copy)

        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            combined_df.to_csv(filepath, index=False)
        else:
            empty_df = pd.DataFrame({"sample_id": [], "class_label": []})
            empty_df.to_csv(filepath, index=False)

        return filepath

    def _save_as_pickle(
        self, dfs: List[pd.DataFrame], classes: List[str], filepath: str
    ) -> str:
        """Save as pickle format."""
        data = {"dataframes": dfs, "classes": classes}
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        return filepath

    def load_cv_folds(
        self, fold_dir: Optional[str] = None, save_format: str = "pickle"
    ) -> List[Dict[str, Any]]:
        """
        Load cross-validation folds from disk.

        Args:
            fold_dir (str, optional): Directory containing folds. If None, uses default.
            save_format (str): File format to load

        Returns:
            List[Dict]: Loaded cross-validation folds
        """
        if fold_dir is None:
            fold_dir = os.path.join(self.cv_splits_dir, "raw")

        if not os.path.exists(fold_dir):
            raise FileNotFoundError(f"Fold directory not found: {fold_dir}")

        cv_folds = []
        fold_dirs = [d for d in os.listdir(fold_dir) if d.startswith("fold_")]
        fold_dirs.sort(key=lambda x: int(x.split("_")[1]))

        for fold_dirname in fold_dirs:
            fold_path = os.path.join(fold_dir, fold_dirname)

            # Load metadata
            metadata_file = os.path.join(fold_path, "fold_metadata.json")
            with open(metadata_file, "r") as f:
                fold_metadata = json.load(f)

            # Load data
            train_dfs, train_classes = self._load_dataframes(
                os.path.join(fold_path, f"train_data.{save_format}"), save_format
            )
            test_dfs, test_classes = self._load_dataframes(
                os.path.join(fold_path, f"test_data.{save_format}"), save_format
            )

            fold_info = {
                "fold_number": fold_metadata["fold_number"],
                "train_dfs": train_dfs,
                "train_classes": train_classes,
                "test_dfs": test_dfs,
                "test_classes": test_classes,
                "train_real_count": fold_metadata["train_real_count"],
                "train_sim_count": fold_metadata["train_sim_count"],
                "test_count": fold_metadata["test_count"],
                "train_ratio": fold_metadata["train_ratio"],
                "test_ratio": fold_metadata["test_ratio"],
            }

            cv_folds.append(fold_info)

        if self.verbose:
            print(f"📂 Loaded {len(cv_folds)} cross-validation folds")

        return cv_folds

    def _load_dataframes(
        self, filepath: str, save_format: str
    ) -> Tuple[List[pd.DataFrame], List[str]]:
        """Load dataframes from file."""
        if save_format.lower() == "pickle":
            return self._load_from_pickle(filepath)
        elif save_format.lower() == "csv":
            return self._load_from_csv(filepath)
        elif save_format.lower() == "pickle":
            return self._load_from_pickle(filepath)
        else:
            raise ValueError(f"Unsupported load format: {save_format}")

    def _load_from_pickle(self, filepath: str) -> Tuple[List[pd.DataFrame], List[str]]:
        """Load from pickle format."""
        combined_df = pd.read_pickle(filepath)

        if combined_df.empty:
            return [], []

        dfs = []
        classes = []

        for sample_id in combined_df["sample_id"].unique():
            sample_data = combined_df[combined_df["sample_id"] == sample_id].copy()
            class_label = sample_data["class_label"].iloc[0]
            sample_data = sample_data.drop(["sample_id", "class_label"], axis=1)

            dfs.append(sample_data)
            classes.append(class_label)

        return dfs, classes

    def _load_from_csv(self, filepath: str) -> Tuple[List[pd.DataFrame], List[str]]:
        """Load from CSV format."""
        combined_df = pd.read_csv(filepath)

        if combined_df.empty:
            return [], []

        dfs = []
        classes = []

        for sample_id in combined_df["sample_id"].unique():
            sample_data = combined_df[combined_df["sample_id"] == sample_id].copy()
            class_label = sample_data["class_label"].iloc[0]
            sample_data = sample_data.drop(["sample_id", "class_label"], axis=1)

            dfs.append(sample_data)
            classes.append(class_label)

        return dfs, classes

    def _load_from_pickle(self, filepath: str) -> Tuple[List[pd.DataFrame], List[str]]:
        """Load from pickle format."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return data["dataframes"], data["classes"]

    def _load_all_windowed_folds(self, windowed_dir: str) -> Dict[str, Any]:
        """Load windowed data for all cross-validation folds."""
        # Load overall summary
        overall_summary_file = os.path.join(
            windowed_dir, "overall_windowing_summary.json"
        )
        with open(overall_summary_file, "r") as f:
            overall_summary = json.load(f)

        all_windowed_folds = []
        total_train_windows = 0
        total_test_windows = 0

        # Load each fold's windowed data
        for fold_detail in overall_summary["fold_details"]:
            fold_num = fold_detail["fold_number"]
            fold_windowed_dir = os.path.join(windowed_dir, f"fold_{fold_num}")

            if os.path.exists(fold_windowed_dir):
                # Load windowed data for this fold
                train_file = os.path.join(fold_windowed_dir, "train_windowed.pickle")
                test_file = os.path.join(fold_windowed_dir, "test_windowed.pickle")
                metadata_file = os.path.join(
                    fold_windowed_dir, "windowing_metadata.json"
                )

                train_dfs, train_classes = self._load_from_pickle(train_file)
                test_dfs, test_classes = self._load_from_pickle(test_file)

                # Load fold metadata
                with open(metadata_file, "r") as f:
                    fold_metadata = json.load(f)

                fold_data = {
                    "fold_number": fold_num,
                    "train_windowed_dfs": train_dfs,
                    "train_windowed_classes": train_classes,
                    "test_windowed_dfs": test_dfs,
                    "test_windowed_classes": test_classes,
                    "train_statistics": fold_metadata["train_statistics"],
                    "test_statistics": fold_metadata["test_statistics"],
                    "train_metadata": fold_metadata.get("train_metadata", []),
                    "test_metadata": fold_metadata.get("test_metadata", []),
                }

                all_windowed_folds.append(fold_data)
                total_train_windows += len(train_dfs)
                total_test_windows += len(test_dfs)

        # Prepare windowing results structure
        windowing_results = {
            "all_windowed_folds": all_windowed_folds,
            "total_train_windows": total_train_windows,
            "total_test_windows": total_test_windows,
            "windowing_config": overall_summary.get("windowing_config", {}),
            "overall_summary": overall_summary,
        }

        # Add default fold results for compatibility (first fold)
        if all_windowed_folds:
            first_fold = all_windowed_folds[0]
            windowing_results["default_fold_results"] = {
                "train_windowed_dfs": first_fold["train_windowed_dfs"],
                "train_windowed_classes": first_fold["train_windowed_classes"],
                "test_windowed_dfs": first_fold["test_windowed_dfs"],
                "test_windowed_classes": first_fold["test_windowed_classes"],
                "train_statistics": first_fold["train_statistics"],
                "test_statistics": first_fold["test_statistics"],
            }

        if self.verbose:
            print(f"   ✅ Loaded windowed data for {len(all_windowed_folds)} folds")
            print(
                f"   📊 Total: {total_train_windows} train, {total_test_windows} test windows"
            )

        return windowing_results

    def _load_single_fold_windowing(self, windowed_dir: str) -> Dict[str, Any]:
        """Load legacy single-fold windowed data."""
        windowing_results = {}

        # Try to load windowed data files
        train_windows_file = os.path.join(windowed_dir, "train_windowed.pickle")
        test_windows_file = os.path.join(windowed_dir, "test_windowed.pickle")

        if os.path.exists(train_windows_file):
            train_dfs, train_classes = self._load_from_pickle(train_windows_file)
            windowing_results["train_windowed_dfs"] = train_dfs
            windowing_results["train_windowed_classes"] = train_classes

        if os.path.exists(test_windows_file):
            test_dfs, test_classes = self._load_from_pickle(test_windows_file)
            windowing_results["test_windowed_dfs"] = test_dfs
            windowing_results["test_windowed_classes"] = test_classes

        # Load windowed statistics if available
        stats_file = os.path.join(windowed_dir, "windowing_metadata.json")
        if os.path.exists(stats_file):
            with open(stats_file, "r") as f:
                metadata = json.load(f)
                windowing_results["train_statistics"] = metadata.get(
                    "train_statistics", {}
                )
                windowing_results["test_statistics"] = metadata.get(
                    "test_statistics", {}
                )

        if self.verbose:
            train_count = len(windowing_results.get("train_windowed_dfs", []))
            test_count = len(windowing_results.get("test_windowed_dfs", []))
            print(
                f"   ✅ Loaded legacy windowing: {train_count} train, {test_count} test windows"
            )

        return windowing_results

    def get_save_summary(self) -> Dict[str, Any]:
        """Get summary of saved data."""
        summary = {
            "base_directory": self.base_dir,
            "directories": {
                "cv_splits": self.cv_splits_dir,
                "metadata": self.metadata_dir,
            },
            "files": {},
        }

        # Count saved files
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                rel_path = os.path.relpath(file_path, self.base_dir)
                summary["files"][rel_path] = {
                    "size_bytes": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 2),
                }

        total_size = sum(info["size_bytes"] for info in summary["files"].values())
        summary["total_size_mb"] = round(total_size / (1024 * 1024), 2)
        summary["total_files"] = len(summary["files"])

        return summary


# Convenience functions
def save_complete_pipeline(
    cv_folds: List[Dict[str, Any]],
    windowing_results: Dict[str, Any],
    processing_config: Dict[str, Any],
    base_dir: str = "processed_data",
) -> str:
    """
    Save complete data processing pipeline results.

    Args:
        cv_folds: Cross-validation folds
        windowing_results: Windowing results
        processing_config: Processing configuration
        base_dir: Base directory for saving

    Returns:
        str: Base directory where data was saved
    """
    persistence = DataPersistence(base_dir=base_dir, verbose=True)

    # Save CV folds
    cv_dir = persistence.save_cv_folds(cv_folds, save_format="pickle")

    # Save windowed data
    windowed_dir = persistence.save_windowed_data(
        windowing_results, save_format="pickle"
    )

    # Save processing summary
    summary_data = {
        "processing_config": processing_config,
        "cv_folds_count": len(cv_folds),
        "cv_splits_dir": cv_dir,
        "windowed_dir": windowed_dir,
    }

    # Add windowing counts based on structure
    if "all_windowed_folds" in windowing_results:
        # New all-folds structure
        summary_data["windowed_folds_count"] = len(
            windowing_results["all_windowed_folds"]
        )
        summary_data["total_train_windows"] = windowing_results.get(
            "total_train_windows", 0
        )
        summary_data["total_test_windows"] = windowing_results.get(
            "total_test_windows", 0
        )
        summary_data["windowing_applied_to_all_folds"] = True
    else:
        # Legacy single-fold structure
        summary_data["windowed_train_count"] = len(
            windowing_results.get("train_windowed_dfs", [])
        )
        summary_data["windowed_test_count"] = len(
            windowing_results.get("test_windowed_dfs", [])
        )
        summary_data["windowing_applied_to_all_folds"] = False

    persistence.save_processing_summary(summary_data)

    return base_dir


def load_complete_pipeline(base_dir: str = "processed_data") -> Dict[str, Any]:
    """
    Load complete data processing pipeline results.

    Args:
        base_dir: Base directory where data was saved

    Returns:
        Dict containing:
            - cv_folds: Cross-validation folds
            - windowing_results: Windowing results
            - processing_config: Processing configuration
            - metadata: Loading metadata
    """
    persistence = DataPersistence(base_dir=base_dir, verbose=True)

    # Load processing summary to get metadata
    try:
        summary_file = os.path.join(persistence.metadata_dir, "processing_summary.json")
        with open(summary_file, "r") as f:
            processing_summary = json.load(f)
    except FileNotFoundError:
        print("⚠️ Processing summary not found. Loading with default parameters.")
        processing_summary = {"processing_config": {}}

    # Load CV folds
    print("📂 Loading cross-validation folds...")
    cv_folds = persistence.load_cv_folds(save_format="pickle")

    # Load windowed data (if available)
    windowed_dir = os.path.join(persistence.cv_splits_dir, "windowed")
    windowing_results = {}

    if os.path.exists(windowed_dir):
        print("📂 Loading windowed data...")

        # Check if we have the new all-folds structure
        overall_summary_file = os.path.join(
            windowed_dir, "overall_windowing_summary.json"
        )

        if os.path.exists(overall_summary_file):
            # Load new all-folds windowing structure
            windowing_results = persistence._load_all_windowed_folds(windowed_dir)
        else:
            # Load legacy single-fold windowing structure
            windowing_results = persistence._load_single_fold_windowing(windowed_dir)
    else:
        print("ℹ️ No windowed data found.")

    # Prepare return data
    results = {
        "cv_folds": cv_folds,
        "windowing_results": windowing_results,
        "processing_config": processing_summary.get("processing_config", {}),
        "metadata": {
            "load_timestamp": datetime.now().isoformat(),
            "base_dir": base_dir,
            "cv_folds_count": len(cv_folds),
            "has_all_windowed_folds": "all_windowed_folds" in windowing_results,
            "total_windowed_folds": len(
                windowing_results.get("all_windowed_folds", [])
            ),
            "total_train_windows": windowing_results.get("total_train_windows", 0),
            "total_test_windows": windowing_results.get("total_test_windows", 0),
            "processing_summary": processing_summary,
        },
    }

    print("✅ Complete pipeline data loaded successfully!")
    print(f"   • CV folds: {len(cv_folds)}")

    if "all_windowed_folds" in windowing_results:
        windowed_folds_count = len(windowing_results["all_windowed_folds"])
        total_train = windowing_results.get("total_train_windows", 0)
        total_test = windowing_results.get("total_test_windows", 0)
        print(f"   • Windowed folds: {windowed_folds_count}")
        print(f"   • Total training windows: {total_train}")
        print(f"   • Total test windows: {total_test}")
    elif windowing_results:
        train_count = len(windowing_results.get("train_windowed_dfs", []))
        test_count = len(windowing_results.get("test_windowed_dfs", []))
        print(f"   • Training windows: {train_count}")
        print(f"   • Test windows: {test_count}")

    return results
