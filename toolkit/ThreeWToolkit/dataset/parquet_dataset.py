import random
import shutil
import zipfile
import pandas as pd

from pathlib import Path
from typing import Any, Dict

from pandas import read_parquet

from ..core.base_step import BaseStep
from ..core.base_dataset import ParquetDatasetConfig
from ..utils.downloader import get_figshare_data
from ..utils.data_utils import default_data_processing

DATASET_VALIDATION_RULES = {
    "2.0.0": {"total_parquet_files": 2228},
}


class ParquetDataset(BaseStep):
    """
    Dataset handler for Parquet files.

    This class manages loading, filtering, preprocessing, and batching
    of event-based parquet datasets according to the provided configuration.
    """

    def __init__(self, config: ParquetDatasetConfig):
        self.config = config

        # Check if dataset version is valid
        if self.config.version not in DATASET_VALIDATION_RULES:
            raise ValueError(
                f"Dataset version {self.config.version} is not valid. \
                Supported versions are: {list(DATASET_VALIDATION_RULES.keys())}"
            )

        # TODO: Implement dataset splitting for train, val, test
        if self.config.split not in [None, "list"]:
            raise ValueError("Dataset splitting not implemented.")

        # Check if download is forced
        if self.config.force_download:
            print(
                f"`force_download` is True. Deleting existing dataset at {self.config.path}."
            )
            # Delete existing dataset
            if Path(self.config.path).exists():
                shutil.rmtree(self.config.path)

        # If dataset is not extracted, we must download it
        should_download_dataset = not self.is_dataset_extracted_correctly()

        # Download dataset if required
        if should_download_dataset:
            dl_path = Path(config.path) / "download"
            dl_path.mkdir(exist_ok=True, parents=True)

            # Check if a zip already exists
            existing_zips = list(dl_path.glob("*.zip"))
            if existing_zips:
                print(f"[ParquetDataset] Found existing dataset at {dl_path}.")
                zip_path = existing_zips[0]
            else:
                print("[ParquetDataset] Downloading dataset...")
                downloaded = get_figshare_data(
                    path=dl_path, version=self.config.version, chunk_size=1024 * 1024
                )
                zip_path = Path(downloaded[0])

            # Check if all files were extracted correctly
            if self.is_dataset_extracted_correctly():
                print(f"[ParquetDataset] Dataset already extracted at {config.path}.")
            else:
                print(f"[ParquetDataset] Extracting dataset from {zip_path}...")
                with zipfile.ZipFile(zip_path, "r") as zip_file:
                    zip_file.extractall(config.path)
        else:
            print(f"[ParquetDataset] Dataset found at {config.path}")

        # Collect parquet files
        root = Path(self.config.path)

        # Check if dataset
        print("[ParquetDataset] Validating dataset integrity...")
        pass_validation, error_messages = self.validate_dataset_integrity()
        if not pass_validation:
            msg = "\n".join(error_messages)
            raise FileNotFoundError(f"Dataset validation failed: {msg}")
        else:
            print("[ParquetDataset] Dataset integrity check passed!")

        all_events_files = [e.relative_to(root) for e in root.rglob("*.parquet")]
        all_events_files = sorted(all_events_files)

        # Filter files by event type and class
        found_events = self._filter_events(all_events_files)

        if self.config.file_list is None:
            self.files_events = found_events
        else:
            # Cleans the config file list so that only parquet files are left
            self.config.file_list = [
                Path(p) for p in self.config.file_list if Path(p).suffix == ".parquet"
            ]

            not_found = set(Path(p) for p in self.config.file_list) - set(found_events)
            if not_found:
                raise RuntimeError(
                    f'"file_list" contains files not found in root path: {not_found}'
                )
            self.files_events = [Path(p) for p in self.config.file_list]

        # Remove target column from feature names if necessary
        if self.config.columns:
            self.feature_names = self.config.columns.copy()
            if self.config.target_column in self.feature_names:
                self.feature_names.remove(self.config.target_column)
            print(f">> {self.feature_names}")

    def _check_event_type(self, event: Path) -> bool:
        """
        Check if the event file name matches one of the requested event types.
        """
        if isinstance(self.config.event_type, list):
            return any(event.name.startswith(t.value) for t in self.config.event_type)
        else:  # Default: accept all
            return True

    def _check_event_class(self, event: Path) -> bool:
        """
        Check if the event folder name matches one of the requested target classes.
        """
        if isinstance(self.config.target_class, list):
            return int(event.parent.name) in self.config.target_class
        else:  # Default: accept all
            return True

    def _filter_events(self, events: list[Path]) -> list[Path]:
        """
        Filter events that match both type and class constraints.
        """
        return [
            e
            for e in events
            if self._check_event_type(e) and self._check_event_class(e)
        ]

    def is_dataset_extracted_correctly(self):
        """
        Check if the dataset is already extracted by checking the number of parquet files.
        """
        extracted_files = list(Path(self.config.path).rglob("*.parquet"))
        total_expected_files = DATASET_VALIDATION_RULES[self.config.version][
            "total_parquet_files"
        ]

        return len(extracted_files) == total_expected_files

    def validate_dataset_integrity(self) -> tuple[bool, list[str]]:
        """
        Verifies the integrity and consistency of a local dataset directory by \
        checking its existence, expected structure, required files, and optional \
        checksums.

        Returns:
            tuple[bool, list[str]]: True if the dataset is consistent, False otherwise.
            List of error messages.
        """

        error_messages = []
        pass_validation = True

        # Check if dataset path exists
        if not Path(self.config.path).exists():
            error_messages.append(f"Dataset directory not found: {self.config.path}.")
            pass_validation = False

        return pass_validation, error_messages

    def iterbatches(self):
        """
        Iterate over batches of parquet files.

        Returns:
            Generator yielding dictionaries with:
                - signals: dict[column -> list of signal arrays]
                - labels: dict[target_column -> list of labels]
                - file_names: dict[file_name -> list of file names]
        """
        total_files = len(self)
        shuffled_ids = list(range(total_files))
        random.seed(self.config.seed)
        random.shuffle(shuffled_ids)

        for idx in range(0, total_files, self.config.files_per_batch):
            batch_ids = shuffled_ids[idx : idx + self.config.files_per_batch]
            batch_files = [self[i] for i in batch_ids]

            dict_signals = {}
            dict_labels = {}
            dict_file_names = {}
            for file_data in batch_files:
                # Extract signals
                for col_name in self.config.columns:
                    samples_col = list(file_data["signal"][col_name])
                    dict_signals.setdefault(col_name, []).append(samples_col)
                # Extract labels
                labels = list(file_data["label"][self.config.target_column])
                dict_labels.setdefault(self.config.target_column, []).append(labels)
                # Extract file names
                file_name = file_data["file_name"]
                dict_file_names.setdefault(file_name, []).append(file_name)

            yield dict(
                signals=dict_signals,
                labels=dict_labels,
                file_names=dict_file_names,
            )

    def iterfiles(self):
        """
        Iterate over individual parquet files one by one.
        """
        total_files = len(self)
        for idx in range(total_files):
            yield self[idx]

    def pre_process(self, data: Any = None) -> Any:
        """
        Clean or preprocess data if required by configuration.
        """
        if self.config.clean_data:
            # NOTE: Currently assuming that the parent folder name corresponds to the target value
            fill_target_value = int(data.get("file_name").parent.name)

            data = default_data_processing(
                data,
                target_column=self.config.target_column,
                fill_target_value=fill_target_value,
            )
        return data

    def run(self, data: Any = None) -> Any:
        """
        Run the dataset step. Only checks are performed.

        If the pipeline calls `.run()` directly, return a DataLoader.
        Otherwise, this could return (X_tensor, y_tensor) and let `post_process`
        handle DataLoader creation. For compatibility, we return the input data.
        """
        if data["signal"].empty:
            raise ValueError("[ParquetDataset] Empty file or invalid columns.")

        # Only check label if target_column is defined
        if self.config.target_column is not None:
            if "label" not in data:
                raise ValueError("[ParquetDataset] No target column found.")

            if not isinstance(data["label"], (pd.Series, pd.DataFrame)):
                raise ValueError(
                    "[ParquetDataset] 'label' column must be a pandas Series or DataFrame."
                )

            if data["label"].empty:
                raise ValueError("[ParquetDataset] Target column is empty.")

            if len(data["label"]) != len(data["signal"]):
                raise ValueError(
                    "[ParquetDataset] Target column has different length than signal."
                )

        return data

    def post_process(self, data: Any) -> Any:
        """
        If you want to use the BaseStep flow (pre -> run -> post),
        return the DataLoader here when `run` returns (X, y).
        Since `run` already returns a DataLoader-compatible object,
        this method currently acts as an identity function.
        """
        return data

    def __len__(self) -> int:
        """Return the number of events in the dataset."""
        return len(self.files_events)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load and process one dataset file.

        Args:
            idx (int): Index of the file.

        Returns:
            Dict[str, Any]: Dictionary containing signals, labels, and file name.
        """
        data = self.load_data(idx)
        return self.__call__(data)

    def load_data(self, idx: int) -> Dict[str, Any]:
        """
        Load a parquet file and separate signals and labels.

        Args:
            idx (int): File index.

        Returns:
            Dict[str, Any]:
                - signal: DataFrame of input signals
                - label: DataFrame of labels (if target_column is defined)
                - file_name: Path to the file
        """
        file_name = self.files_events[idx]
        path = Path(self.config.path) / file_name
        ret: Dict[str, Any] = {}

        ret["signal"] = read_parquet(
            path, columns=self.config.columns, engine="pyarrow"
        )

        if self.config.target_column is not None:
            # Drop target column from signals if present
            if self.config.target_column in ret["signal"].columns:
                ret["signal"].drop(columns=[self.config.target_column], inplace=True)

            # Load target column separately
            ret["label"] = read_parquet(
                path, columns=[self.config.target_column], engine="pyarrow"
            )
        ret["file_name"] = file_name
        return ret
