from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal, Union, Tuple, Any


class DataLoaderConfig(BaseModel):
    source_path: str = Field(..., description="Path to the data file.")
    file_type: Literal["csv", "parquet", "json"] = Field(..., description="File type.")
    has_header: bool = True
    separator: Optional[str] = Field(",", description="Field separator (for CSV).")
    target_column: Optional[str] = Field(
        None, description="Target column for supervised tasks."
    )
    shape: Optional[Tuple[int, int]] = None

    @field_validator("source_path")
    @classmethod
    def path_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("source_path cannot be empty.")
        return v

    @field_validator("separator")
    @classmethod
    def validate_separator(cls, v, info):
        file_type = info.data.get("file_type")
        if file_type == "csv" and not v:
            raise ValueError("Separator must be provided for CSV files.")
        return v


class BaseDataLoader(ABC):
    def __init__(self, config: DataLoaderConfig):
        self.config = config

    @abstractmethod
    def load_data(self) -> Union[Any, Tuple[Any, Any]]:
        """
        Loads data based on the configuration.
        Should return (X, y) or just X.
        """
        pass

    def describe_config(self):
        print("DataLoader configuration:")
        for k, v in self.config.model_dump().items():
            print(f"{k}: {v}")
