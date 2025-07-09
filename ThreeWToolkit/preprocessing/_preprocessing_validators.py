import pandas as pd

from pydantic import BaseModel, ConfigDict, field_validator
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
