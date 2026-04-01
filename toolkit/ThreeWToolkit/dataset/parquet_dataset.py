import shutil
import zipfile
from pathlib import Path
from pandas import read_parquet
from pydantic import Field, field_validator, PrivateAttr
from typing import Literal
from ThreeWToolkit.core.enums import EventPrefixEnum
from ThreeWToolkit.core.dataset_outputs import DatasetOutputs
from ..utils.downloader import get_figshare_data

from ..core.base_dataset import BaseDataset, BaseDatasetConfig

DATASET_VALIDATION_RULES = {
    "2.0.0": {"total_parquet_files": 2228},
}


class ParquetDatasetConfig(BaseDatasetConfig):
    """
    Configuration schema for loading a Parquet dataset.
    Defines the dataset location, splits, filtering options, and preprocessing behavior.
    """

    path: Path | str = Field(..., description="Path to the dataset directory or file.")

    split: Literal["train", "val", "test", "list"] | None = Field(
        default=None,
        description="Dataset split to load. If None, load all available files.",
    )

    file_list: list[str] | list[Path] | None = Field(
        default=None,
        description='List of files to load if split=="list".',
    )

    event_type: list[EventPrefixEnum] | None = Field(
        default=None,
        description="Event types to include. (e.g., simulated, real, ...)",
    )

    target_class: list[int] | None = Field(
        default=None,
        description="Event classes to include. (e.g., 0, 1, 2, ...). Loads all classes if `None`.",
    )

    columns: list[str] | None = Field(
        default=None,
        description="Specific data columns to load. Loads all columns if `None`.",
    )

    target_column: str | None = Field(
        default="class",
        description="Target column used for supervised tasks.",
    )

    force_download: bool = Field(
        default=False,
        description="If True, dataset is downloaded even if it already exists. In this case, \
            existing files will be overwritten.",
    )

    version: Literal["2.0.0"] = Field(
        default="2.0.0",
        description="Dataset version to load. (e.g., 2.0.0, 2.0.1, ...)",
    )

    _target: type = PrivateAttr(default_factory=lambda: ParquetDataset)

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: Path | str) -> Path:
        """Convert str to Path if needed."""
        return Path(v) if isinstance(v, str) else v

    @field_validator("file_list")
    @classmethod
    def validate_file_list(cls, file_list, info):
        """
        Ensure that `file_list` is only provided when `split=="list"`.
        Raise a ValueError otherwise.
        """
        split = info.data.get("split")
        if split == "list" and file_list is None:
            raise ValueError('file_list must be provided if split is "list".')
        elif split != "list" and file_list is not None:
            raise ValueError(f'file_list must not be provided if split="{split}".')
        return file_list


class ParquetDataset(BaseDataset):
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

        # Check dataset integrity
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

    def _get_event_type_from_filename(self, event: Path) -> str:
        """
        Extract the event type prefix from the file name. Assumes the event type is indicated by a prefix in the file name.
        """
        for prefix in [
            ("WELL", "real"),
            ("SIMULATED", "simulated"),
            ("DRAWN", "drawn"),
        ]:
            if event.name.startswith(prefix[0]):
                return prefix[1]
        return "unknown"

    def _get_event_class_from_folder(self, event: Path) -> int:
        """
        Extract the event class from the parent folder name. Assumes the parent folder is named with the class number.
        """
        try:
            return int(event.parent.name)
        except ValueError:
            return -1  # Return -1 if the parent folder name is not an integer

    def _check_event_type(self, event: Path) -> bool:
        """
        Check if the event file name matches one of the requested event types.
        """
        if self.config.event_type is not None:
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

    def __len__(self) -> int:
        """Return the number of events in the dataset."""
        return len(self.files_events)

    def __getitem__(self, idx: int) -> DatasetOutputs:
        """
        Load and process one dataset file.

        Args:
            idx (int): Index of the file.

        Returns:
            DatasetOutputs: Structured object containing signals, labels, and metadata.
        """
        return self.load_file(idx)

    def load_file(self, idx: int) -> DatasetOutputs:
        """
        Load a parquet file and separate signals and labels.

        Args:
            idx (int): File index.

        Returns:
            DatasetOutputs: Structured object containing:
                - signal: DataFrame of input signals
                - label: Series of labels (if target_column is defined)
                - metadata: Dict with file_name and other info
        """
        file_name = self.files_events[idx]

        event_class = self._get_event_class_from_folder(file_name)
        event_type = self._get_event_type_from_filename(file_name)

        path = Path(self.config.path) / file_name

        # Read single parquet file
        parquet_file = read_parquet(path, engine="pyarrow")

        # extract label column if defined, otherwise set to None
        if self.config.target_column is not None:
            if self.config.target_column not in parquet_file.columns:
                raise ValueError(
                    f"Target column '{self.config.target_column}' not found in file {file_name}."
                )

            label_series = parquet_file[self.config.target_column]
            parquet_file = parquet_file.drop(columns=[self.config.target_column])
        else:
            label_series = None

        if self.config.columns is not None:
            if not set(self.config.columns).issubset(set(parquet_file.columns)):
                missing_cols = set(self.config.columns) - set(parquet_file.columns)
                raise ValueError(
                    f"Some specified columns are not found in file {file_name}: {missing_cols}"
                )

            signal_df = parquet_file[self.config.columns]
        else:  # load the remaining columns as signal if no specific columns are defined
            signal_df = parquet_file

        return DatasetOutputs(
            signal=signal_df,
            label=label_series,
            metadata={
                "file_name": file_name,
                "event_class": event_class,
                "event_type": event_type,
            },
        )
