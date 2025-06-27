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
    
class LabelsValidator(BaseModel):
    labels: Optional[list] = None

    @field_validator("labels", mode="before")
    @classmethod
    def check_labels(cls, v):
        if v is not None and not isinstance(v, list):
            raise TypeError(f"'labels' must be a list or None, got {type(v)}")
        return v
    
class PosLabelValidator(BaseModel):
    pos_label: Optional[int] = 1

    @field_validator("pos_label", mode="before")
    @classmethod
    def check_pos_label(cls, v):
        if not isinstance(v, (int, float)) and v is not None:
            raise TypeError(f"'pos_label' must be a number or None, got {type(v)}")
        return v
    
class AverageValidator(BaseModel):
    average: Optional[str] = "binary"

    @field_validator("average", mode="before")
    @classmethod
    def check_average(cls, v):
        allowed = {"micro", "macro", "samples", "weighted", "binary", None}
        if v not in allowed:
            raise ValueError(f"'average' must be one of {allowed}, got '{v}'")
        return v

class ZeroDivisionValidator(BaseModel):
    zero_division: Union[str, int] = "warn"

    @field_validator("zero_division", mode="before")
    @classmethod
    def check_zero_division(cls, v):
        if v not in {"warn", 0, 1}:
            raise ValueError(f"'zero_division' must be one of ['warn', 0, 1], got '{v}'")
        return v

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
    
class AveragePrecisionScoreArgsValidator(BaseScoreArgsValidator,
                                         AverageValidator,
                                         PosLabelValidator):
        
    @field_validator("average", mode="before")
    @classmethod
    def override_average_options(cls, v):
        allowed = {"micro", "macro", "samples", "weighted", None}
        if v not in allowed:
            raise ValueError(f"'average' must be one of {allowed}, got '{v}'")
        return v
    
class MultiClassValidator(BaseModel):
    multi_class: str = "raise"

    @field_validator("multi_class", mode="before")
    @classmethod
    def check_multi_class(cls, v):
        allowed = {"raise", "ovr", "ovo"}
        if v not in allowed:
            raise ValueError(f"'multi_class' must be one of {allowed}, got '{v}'")
        return v
    
class MaxFprValidator(BaseModel):
    max_fpr: Optional[float] = None

    @field_validator("max_fpr", mode="before")
    @classmethod
    def check_max_fpr(cls, v):
        if v is not None and (not isinstance(v, (int, float)) or v <= 0 or v > 1):
            raise ValueError(f"'max_fpr' must be a float in the range (0, 1], got {v}")
        return v
    
class PrecisionScoreArgsValidator(BaseScoreArgsValidator,
                                  LabelsValidator,
                                  PosLabelValidator,
                                  AverageValidator,
                                  ZeroDivisionValidator):
    
    pass

class RecallScoreArgsValidator(BaseScoreArgsValidator,
                               LabelsValidator,
                               PosLabelValidator,
                               AverageValidator,
                               ZeroDivisionValidator):
    
    pass

class F1ScoreArgsValidator(BaseScoreArgsValidator,
                           LabelsValidator,
                           PosLabelValidator,
                           AverageValidator,
                           ZeroDivisionValidator):
    
    pass

class RocAucScoreArgsValidator(BaseScoreArgsValidator,
                               AverageValidator,
                               MaxFprValidator,
                               MultiClassValidator,
                               LabelsValidator
):
        
    @field_validator("average", mode="before")
    @classmethod
    def override_average_options(cls, v):
        allowed = {"micro", "macro", "samples", "weighted", None}
        if v not in allowed:
            raise ValueError(f"'average' must be one of {allowed}, got '{v}'")
        return v
    
class MultiOutputValidator(BaseModel):
    multioutput: str = "uniform_average"

    @field_validator("multioutput", mode="before")
    @classmethod
    def check_multioutput(cls, v):
        allowed = {"raw_values", "uniform_average", "variance_weighted"}
        if v not in allowed:
            raise ValueError(
                f"'multioutput' must be one of {allowed}, got '{v}'"
            )
        return v

class ForceFiniteValidator(BaseModel):
    force_finite: bool = True

    @field_validator("force_finite", mode="before")
    @classmethod
    def check_force_finite(cls, v):
        if not isinstance(v, bool):
            raise TypeError(f"'force_finite' must be a boolean, got {type(v)}")
        return v
    
class ExplainedVarianceScoreArgsValidator(BaseScoreArgsValidator,
                                          MultiOutputValidator,
                                          ForceFiniteValidator
):
    
    pass