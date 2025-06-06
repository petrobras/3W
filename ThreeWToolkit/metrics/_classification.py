from typing import Optional, Union
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score as sk_acc

from ..utils.general_utils import GeneralUtils
from ._metrics_validators import AccuracyScoreArgsValidator


@GeneralUtils.validate_func_args_with_pydantic(AccuracyScoreArgsValidator)
def accuracy_score(y_true: Union[np.ndarray, pd.Series, list], 
                   y_pred: Union[np.ndarray, pd.Series, list], 
                   normalize: bool = True, 
                   sample_weight: Optional[Union[np.ndarray, pd.Series, list]] =  None) -> float:
    """
    Calculates the accuracy between true and predicted values.

    This function validates the arguments using Pydantic to ensure
    correct types and compatible sizes before calculating the metric.

    Args:
        y_true (np.ndarray | pd.Series | list): True values.
        y_pred (np.ndarray | pd.Series | list): Predicted values.
        normalize (bool, optional): If True, returns the fraction correct.
                                    If False, returns the number of correct samples.
                                    Default is True.
        sample_weight (np.ndarray | pd.Series | list | None, optional):
            Sample weights. Default is None.

    Returns:
        float: The calculated accuracy.

    Raises:
        TypeError: If any argument has an invalid type.
        ValueError: If array dimensions are incompatible.
    """

    return sk_acc(y_true = y_true,
                  y_pred = y_pred,
                  normalize = normalize,
                  sample_weight = sample_weight)