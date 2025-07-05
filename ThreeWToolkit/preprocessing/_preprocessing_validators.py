import pandas as pd

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import Literal, Optional, Union, List


class ImputeMissingArgsValidator(BaseModel):
    data: Union[pd.DataFrame, pd.Series]
    strategy: Literal["mean", "median", "constant"]
    fill_value: Optional[Union[int, float]] = None
    columns: Optional[List[str]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("fill_value")
    def check_fill_value_for_constant(cls, v, info):
        strategy = info.data.get("strategy")
        if strategy == "constant" and v is None:
            raise ValueError("You must provide `fill_value` when strategy='constant'")
        return v


class NormalizeArgsValidator(BaseModel):
    X: Union[pd.DataFrame, pd.Series]
    norm: Literal["l1", "l2", "max"] = "l2"
    axis: Optional[Literal[0, 1]] = 1
    copy_values: Optional[bool] = True
    return_norm_values: Optional[bool] = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("X")
    def validate_numeric(cls, v):
        if isinstance(v, pd.Series):
            if not pd.api.types.is_numeric_dtype(v):
                raise TypeError("Series must be numeric.")

        else:
            non_numeric = [
                col for col in v.columns if not pd.api.types.is_numeric_dtype(v[col])
            ]
            if non_numeric:
                raise TypeError(
                    f"All columns must be numeric. Non-numeric columns: {non_numeric}"
                )

        return v


class WindowingArgsValidator(BaseModel):
    X: Union[pd.Series, pd.DataFrame]
    window: Literal[
        "boxcar",
        "triang",
        "blackman",
        "hamming",
        "hann",
        "bartlett",
        "flattop",
        "parzen",
        "bohman",
        "blackmanharris",
        "nuttall",
        "barthann",
        "cosine",
        "exponential",
        "tukey",
        "taylor",
        "lanczos",
    ] = "hann"
    window_size: int = Field(..., gt=1)
    overlap: float = Field(0.0, ge=0.0, lt=1.0)
    normalize: bool = False
    fftbins: bool = True
    pad_last_window: bool = False
    pad_value: float = 0.0

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("X")
    def validate_numeric(cls, v):
        if not pd.api.types.is_numeric_dtype(v):
            raise TypeError("Series must be numeric.")

        return v

    @model_validator(mode="after")
    def check_window_size_vs_data(self):
        n_samples = len(self.X)
        if self.window_size > n_samples:
            raise ValueError(
                "`window_size` must be smaller than or equal to the length of X."
            )
        return self
