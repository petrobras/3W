from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import Literal
from pathlib import Path

from .enums import EventPrefixEnum


class ParquetDatasetConfig(BaseModel):
    """
    Configuration schema for loading a Parquet dataset.
    Defines the dataset location, splits, filtering options, and preprocessing behavior.
    """

    path: str | Path = Field(..., description="Path to the dataset directory or file.")
    split: Literal[None, "train", "val", "test", "list"] = Field(
        default=None,
        description="Dataset split to load. If None, load all available files.",
    )
    file_list: (list[str] | list[Path]) | None = Field(
        default=None,
        description='List of files to load if split=="list". Must be explicitly provided.',
    )
    event_type: list[EventPrefixEnum] | list[str] | None = Field(
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
    files_per_batch: int = Field(
        default=10,
        description="Number of files to load per batch during iteration.",
    )
    shuffle: bool = Field(
        default=False,
        description="If True, shuffle dataset files before loading.",
    )
    clean_data: bool = Field(
        default=True,
        description="If True, apply basic cleaning to the loaded data.",
    )
    seed: int = Field(
        default=2025,
        description="Random seed for reproducibility in shuffling and splits.",
    )
    version: str = Field(
        default="2.0.0",
        description="Dataset version to load. (e.g., 2.0.0, 2.0.1, ...)",
    )

    @field_validator("file_list")
    @classmethod
    def validate_file_list(
        cls: type["ParquetDatasetConfig"],
        file_list: list[str] | list[Path] | None,
        info: ValidationInfo,
    ) -> list[str] | list[Path] | None:
        """
        Ensure that `file_list` is only provided when `split=="list"`.

        Args:
            cls (ParquetDatasetConfig): The class reference.
            file_list (list[str] | list[Path] | None): List of files to load.
            info (ValidationInfo): Validation info containing the split.

        Returns:
            list[str] | list[Path] | None: Validated file list.

        Raises:
            ValueError: If file_list is provided incorrectly based on split.
        """
        split = info.data.get("split")
        if split == "list" and file_list is None:
            raise ValueError('file_list must be provided if split is "list".')
        elif split != "list" and file_list is not None:
            raise ValueError(f'file_list must not be provided if split="{split}".')
        return file_list

    @field_validator("event_type")
    @classmethod
    def validate_event_type(
        cls: type["ParquetDatasetConfig"],
        event_type: list[EventPrefixEnum] | list[str] | None,
    ) -> list[EventPrefixEnum] | list[str] | None:
        """
        Ensure that all event types are valid string values of EventPrefixEnum.

        Args:
            cls (ParquetDatasetConfig): The class reference.
            event_type (list[EventPrefixEnum] | list[str] | None): List of event types.

        Returns:
            list[EventPrefixEnum] | list[str] | None: Validated event types.

        Raises:
            ValueError: If an unknown type is provided.
            TypeError: If event_type is not a list of strings.
        """
        if event_type is not None:
            valid_strs = {e.value for e in EventPrefixEnum}
            if not isinstance(event_type, list) or not all(
                isinstance(event_type_item, str) for event_type_item in event_type
            ):
                raise TypeError("event_type must be a list of str.")
            for event_type_item in event_type:
                if event_type_item not in valid_strs:
                    raise ValueError(f"Unknown event_type: {event_type_item}")
        return event_type
