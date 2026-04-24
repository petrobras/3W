"""DatasetOutputs class for structured dataset return values."""

from typing import Any
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict


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

    label: pd.Series | None = Field(..., description="Optional target labels")

    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
