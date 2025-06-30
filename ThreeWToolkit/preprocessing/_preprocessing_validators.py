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