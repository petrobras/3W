from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal, Union, Tuple, Any, List, Dict

from pathlib import Path


class DatasetConfig(BaseModel):
    path: Union[str, Path] = Field(..., description="Path to the dataset.")
    split: Optional[Literal["train", "val", "test", "list"]] = Field(default=None, description="Split to load. Load all files if None.")
    file_list: Optional[Union[List[str], List[Path]]] = Field(default=None, description="List of files to load if split==\"list\"")

    file_type: Literal["csv", "parquet", "json"] = Field(..., description="File type.")

    columns: Optional[List[str]] = Field(default=None, description="Data columns to be loaded. Loads all if None.")
    target_column: Optional[str] = Field(default="class", description="Target column for supervised tasks.")

    @field_validator("file_list")
    @classmethod
    def validate_file_list(cls, v, info):
        split = info.data.get("split")
        if split == "list" and v is None:
            raise ValueError("File list must be provided if split is \"list\".")
        return v

class BaseDataset(ABC):
    def __init__(self, config: DatasetConfig):
        self.config = config

    def describe_config(self):
        print("DataLoader configuration:")
        for k, v in self.config.model_dump().items():
            print(f"{k}: {v}")

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns number of items in the dataset.
        """
        pass
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        Loads data based on the configuration.
        Should return (X, y) or just X.
        """
        return self.load_data(idx)

    @abstractmethod
    def load_data(self, idx) -> Dict[str, Any]:
        """
        Loads data based on the configuration.
        Should return (X, y) or just X.
        """
        pass

