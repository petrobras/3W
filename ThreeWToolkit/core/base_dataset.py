from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal, Union, Any, List, Dict

from pathlib import Path


class DatasetConfig(BaseModel):
    # Fixed properties
    path: Union[str, Path] = Field(..., description="Path to the dataset.")
    split: Literal[None, "train", "val", "test", "list"] = Field(default=None, description="Split to load. Load all files if None.")
    file_list: Optional[List[str] | List[Path]] = Field(default=None, description="List of files to load if split==\"list\".")

    file_type:  Literal["csv", "parquet", "json"] = Field(..., description="File type.")
    event_type: Optional[List[str]] = Field(default=None, description="Event types to load.")

    columns: Optional[List[str]] = Field(default=None, description="Data columns to be loaded. Loads all if None.")
    target_column: Optional[str] = Field(default="class", description="Target column for supervised tasks.")


    @field_validator("file_list")
    @classmethod
    def validate_file_list(cls, v, info):
        split = info.data.get("split")
        if split == "list" and v is None:
            raise ValueError("File list must be provided if split is \"list\".")
        elif split != "list" and v is not None:
            raise ValueError(f"file_list must not be provided if split is \"{split}\".")
        return v

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v):
        if v is not None:
            for t in v:
                if t not in BaseDataset.EVENT_PREFIX:
                    raise ValueError(f"Unknown event_type: {t}")
        return v

class BaseDataset(ABC):
    EVENT_PREFIX = {
        "real":      "WELL",
        "simulated": "SIMULATED",
        "drawn":     "DRAWN",
    }

    def __init__(self, config: DatasetConfig):
        self.config = config

    def describe_config(self):
        print("DataLoader configuration:")
        for k, v in self.config.model_dump().items():
            print(f"{k}: {v}")

    def check_event_type(self, event: Path) -> bool:
        if isinstance(self.config.event_type, list):
            return any(event.name.startswith(self.EVENT_PREFIX[t]) for t in self.config.event_type)
        else:
            return True

    def filter_events(self, events: List[Path]) -> List[Path]:
        return [e for e in events if self.check_event_type(e)]

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns number of items in the dataset.
        """
        pass
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        Alias to load_data(idx).
        """
        return self.load_data(idx)

    @abstractmethod
    def load_data(self, idx) -> Dict[str, Any]:
        """
        Loads data based on the configuration.
        Should return (X, y) or just X.
        """
        pass

