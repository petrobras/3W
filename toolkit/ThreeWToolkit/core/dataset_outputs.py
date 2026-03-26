"""DatasetOutputs class for structured dataset return values."""

from typing import Any
import pandas as pd
from pydantic import BaseModel, Field, field_validator, ConfigDict


class DatasetOutputs(BaseModel):
    """
    Structured output from dataset loading operations.

    Provides type-safe dataset outputs with backward compatibility for
    dict-like access patterns.

    Attributes:
        signal (pd.DataFrame): Input signal data
        label (pd.Series | None): Target labels for supervised learning
        metadata (dict[str, Any]): Additional metadata (file_name, etc.)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    signal: pd.DataFrame = Field(..., description="Input signal data")
    label: pd.Series | None = Field(default=None, description="Target labels")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("signal")
    @classmethod
    def validate_signal(cls, v):
        if not isinstance(v, pd.DataFrame):
            raise ValueError(f"signal must be a pandas DataFrame, got {type(v)}")
        if v.empty:
            raise ValueError("signal DataFrame cannot be empty")
        return v

    @field_validator("label")
    @classmethod
    def validate_label(cls, v):
        if v is not None and not isinstance(v, pd.Series):
            raise ValueError(f"label must be a pandas Series or None, got {type(v)}")
        return v

    def __getitem__(self, key: str) -> Any:
        """Enable dict-like access for backward compatibility."""
        if key == "signal":
            return self.signal
        elif key == "label":
            return self.label
        elif key in self.metadata:
            return self.metadata[key]
        else:
            raise KeyError(
                f"'{key}' not found. Available: signal, label, metadata keys: {list(self.metadata.keys())}"
            )

    def __setitem__(self, key: str, value: Any) -> None:
        """Enable dict-like assignment for backward compatibility."""
        if key == "signal":
            self.signal = value
        elif key == "label":
            self.label = value
        else:
            self.metadata[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default fallback for backward compatibility."""
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> list[str]:
        """Return all available keys for dict-like iteration."""
        keys = ["signal"]
        if self.label is not None:
            keys.append("label")
        keys.extend(self.metadata.keys())
        return keys

    def __repr__(self) -> str:
        label_info = f"shape={self.label.shape}" if self.label is not None else "None"
        return (
            f"DatasetOutputs(signal: {self.signal.shape}, "
            f"label: {label_info}, metadata: {list(self.metadata.keys())})"
        )
