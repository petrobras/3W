from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal, Any, Callable

from pathlib import Path
from .enums import EventPrefixEnum


class DatasetConfig(BaseModel):
    path: str | Path = Field(..., description="Path to the dataset.")
    split: Literal[None, "train", "val", "test", "list"] = Field(
        default=None, description="Split to load. Load all files if None."
    )
    file_list: Optional[list[str] | list[Path]] = Field(
        default=None, description='List of files to load if split=="list".'
    )

    file_type: Literal["csv", "parquet", "json"] = Field(
        default="parquet", description="File type. (e.g. csv, parquet, ...)"
    )

    event_type: Optional[list[EventPrefixEnum]] = Field(
        default=None, description="Event types to load. (e.g. simulated, real, ...)"
    )
    target_class: Optional[list[int]] = Field(
        default=None, description="Event classes to load. (e.g. 0, 1, 2, ...)"
    )

    columns: Optional[list[str]] = Field(
        default=None, description="Data columns to be loaded. Loads all if None."
    )
    target_column: Optional[str] = Field(
        default="class", description="Target column for supervised tasks."
    )

    @field_validator("file_list")
    @classmethod
    def validate_file_list(cls, v, info):
        split = info.data.get("split")
        if split == "list" and v is None:
            raise ValueError('File list must be provided if split is "list".')
        elif split != "list" and v is not None:
            raise ValueError(f'file_list must not be provided if split is "{split}".')
        return v

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v):
        if v is not None:
            for t in v:
                if t not in list(EventPrefixEnum):
                    raise ValueError(f"Unknown event_type: {t}")
        return v


class BaseDataset(ABC):
    def __init__(self, config: DatasetConfig):
        self.config = config

    def describe_config(self):
        print("DataLoader configuration:")
        for k, v in self.config.model_dump().items():
            print(f"{k}: {v}")

    def check_event_type(self, event: Path) -> bool:
        """
        Check if event prefix matches requested types.
        """
        if isinstance(self.config.event_type, list):
            return any(event.name.startswith(t.value) for t in self.config.event_type)
        else:  # default
            return True

    def check_event_class(self, event: Path) -> bool:
        """
        Check if event class matches requested targets.
        """
        if isinstance(self.config.target_class, list):
            return int(event.parent.name) in self.config.target_class
        else:  # default
            return True

    def filter_events(self, events: list[Path]) -> list[Path]:
        """
        Filter events matching type and target classes.
        """
        return [
            e for e in events if self.check_event_type(e) and self.check_event_class(e)
        ]

    def transform(self, transforms: dict[str, Callable]):
        """Returns a wrapper applying transforms to this dataset"""
        return TransformedDataset(self, transforms)

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns number of items in the dataset.
        """
        pass

    @abstractmethod
    def load_data(self, idx) -> dict[str, Any]:
        """
        Loads data based on the configuration.
        Should return (X, y) or just X.
        """
        pass

    def __getitem__(self, idx) -> dict[str, Any]:
        """
        Alias to load_data(idx).
        """
        return self.load_data(idx)


class TransformedDataset(BaseDataset):
    """Apply transformations to an inner dataset.

    Given a dict of callables, transforms = {"k1": f1, "k2": f2, ...},
    will return
        transformed[idx] == {"k1": f1(**inner[idx]), "k2": f2(**inner), ...}.
    Functions will default to identity if not provided.
    Will raise if "kn" not in inner[idx].
    """

    def __init__(self, dataset: BaseDataset, transforms: dict[str, Callable]):
        self.dataset = dataset
        self.config = dataset.config
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.dataset)

    def load_data(self, idx: int) -> dict[str, Any]:
        # load from base dataset
        item = self.dataset.load_data(idx)

        # check that keys match
        missing_keys = set(self.transforms.keys()).difference(item.keys())
        if len(missing_keys) > 0:
            raise RuntimeError(
                f"Error: {', '.join(missing_keys)} not present in base item."
            )

        transformed = {}
        for key in item.keys():
            if key in self.transforms:  # apply only to keys which are in transforms
                transformed[key] = self.transforms[key](**item)
            else:
                transformed[key] = item[key]

        return transformed
