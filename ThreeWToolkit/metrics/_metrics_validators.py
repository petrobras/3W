import numpy as np
import pandas as pd

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from typing import Union, Optional

class BaseScoreArgsValidator(BaseModel):
    y_true: Union[np.ndarray, pd.Series, list]
    y_pred: Union[np.ndarray, pd.Series, list]
    sample_weight: Optional[Union[np.ndarray, pd.Series, list]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator('y_true', 'y_pred', 'sample_weight', mode='before')
    @classmethod
    def check_array_like(cls, v, info):
        if info.field_name == 'sample_weight' and v is None:
            return v
        if not isinstance(v, (np.ndarray, pd.Series, list)):
            raise TypeError(f"'{info.field_name}' must be a np.ndarray, pd.Series or list, got {type(v)}")
        return v

    @model_validator(mode='after')
    def check_shapes(self):
        if len(self.y_true) != len(self.y_pred):
            raise ValueError(
                f"'y_true' and 'y_pred' must have the same number of elements "
                f"(received: {len(self.y_true)} and {len(self.y_pred)})"
            )
        if self.sample_weight is not None and len(self.sample_weight) != len(self.y_true):
            raise ValueError(
                f"'sample_weight' must have the same number of elements as 'y_true' "
                f"(received: {len(self.sample_weight)} and {len(self.y_true)})"
            )
        return self


class AccuracyScoreArgsValidator(BaseScoreArgsValidator):
    normalize: bool = True

    @field_validator('normalize', mode='before')
    @classmethod
    def check_bool(cls, v):
        if not isinstance(v, bool):
            raise TypeError(f"'normalize' must be a boolean, got {type(v)} with value '{v}'")
        return v
    
class BalancedAccuracyScoreArgsValidator(BaseScoreArgsValidator):
    adjusted: bool = False

    @field_validator('adjusted', mode='before')
    @classmethod
    def check_bool(cls, v):
        if not isinstance(v, bool):
            raise TypeError(f"'adjusted' must be a boolean, got {type(v)} with value '{v}'")
        return v
    
class AveragePrecisionScoreArgsValidator(BaseScoreArgsValidator):
    average: Optional[str] = "binary"
    pos_label: Optional[int] = 1

    @field_validator('average', mode='before')
    @classmethod
    def check_average(cls, v):
        if v not in {"binary", "micro", "macro", "samples", "weighted", None}:
            raise ValueError(
                f"'average' must be one of "
                f"['binary', 'micro', 'macro', 'samples', 'weighted', None], got '{v}'"
            )
        return v

    @field_validator('pos_label', mode='before')
    @classmethod
    def check_pos_label(cls, v):
        if not isinstance(v, (int, float)) and v is not None:
            raise TypeError(f"'pos_label' must be a number or None, got {type(v)}")
        return v
