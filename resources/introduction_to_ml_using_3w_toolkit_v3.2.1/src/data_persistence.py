"""
Save and load the processed data shared by the notebooks.

Notebook 1 saves, for every cross-validation fold, the scaled events
(`complete/`) and their windows (`windowed/`). Notebooks 2 to 5 read them back.
Each pickle holds `{"dataframes": [...], "classes": [...]}`.
"""

import json
import os
import pickle
from typing import Any, Dict, List, Tuple

import pandas as pd

from . import config


class DataPersistence:
    """Directory layout and pickle files of the processed 3W data."""

    # kind -> (file name of the training split, file name of the test split)
    FILES = {
        "complete": ("train_data.pickle", "test_data.pickle"),
        "windowed": ("train_windowed.pickle", "test_windowed.pickle"),
    }

    def __init__(self, base_dir: str = config.PROCESSED_DATA_DIR, verbose: bool = True):
        self.base_dir = base_dir
        self.verbose = verbose
        self.cv_splits_dir = os.path.join(base_dir, "cv_splits")
        self.metadata_dir = os.path.join(base_dir, "metadata")
        for kind in self.FILES:
            os.makedirs(os.path.join(self.cv_splits_dir, kind), exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

    def save_fold(
        self,
        kind: str,
        fold_number: int,
        train: Tuple[List[pd.DataFrame], List[str]],
        test: Tuple[List[pd.DataFrame], List[str]],
    ) -> str:
        """Save the (dataframes, classes) of the training and test split of a fold."""
        fold_dir = os.path.join(self.cv_splits_dir, kind, f"fold_{fold_number}")
        os.makedirs(fold_dir, exist_ok=True)
        for file_name, (dfs, classes) in zip(self.FILES[kind], (train, test)):
            with open(os.path.join(fold_dir, file_name), "wb") as f:
                pickle.dump({"dataframes": dfs, "classes": classes}, f)
        if self.verbose:
            print(
                f"   {kind} fold {fold_number}: {len(train[0])} train, {len(test[0])} test"
            )
        return fold_dir

    def save_summary(self, summary: Dict[str, Any]) -> str:
        """Write the processing summary (parameters and counts) as JSON."""
        path = os.path.join(self.metadata_dir, "processing_summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        return path

    def _load_dataframes(
        self, filepath: str, save_format: str = "pickle"
    ) -> Tuple[List[pd.DataFrame], List[str]]:
        """Load the (dataframes, classes) saved by `save_fold`."""
        if save_format.lower() != "pickle":
            raise ValueError(f"Unsupported load format: {save_format}")
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return data["dataframes"], data["classes"]

    def get_save_summary(self) -> Dict[str, Any]:
        """Number and size of the files saved so far."""
        sizes = [
            os.path.getsize(os.path.join(root, name))
            for root, _, names in os.walk(self.base_dir)
            for name in names
        ]
        return {
            "total_files": len(sizes),
            "total_size_mb": round(sum(sizes) / (1024 * 1024), 2),
        }
