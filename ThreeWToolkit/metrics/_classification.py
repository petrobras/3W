import numpy as np
import pandas as pd

from typing import Optional, Union
from sklearn.metrics import (
        accuracy_score as sk_acc,
        balanced_accuracy_score as sk_balanced_acc
)

from ..utils.general_utils import GeneralUtils
from ._metrics_validators import AccuracyScoreArgsValidator, BalancedAccuracyScoreArgsValidator


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

@GeneralUtils.validate_func_args_with_pydantic(BalancedAccuracyScoreArgsValidator)
def balanced_accuracy_score(y_true: Union[np.ndarray, pd.Series, list],
                            y_pred: Union[np.ndarray, pd.Series, list],
                            sample_weight: Optional[Union[np.ndarray, pd.Series, list]] = None,
                            adjusted: bool = False) -> float:
    """
    Computes the balanced accuracy between true and predicted values.

    This metric is robust to imbalanced datasets by averaging recall obtained on each class.

    Args:
        y_true (np.ndarray | pd.Series | list): True class labels.
        y_pred (np.ndarray | pd.Series | list): Predicted class labels.
        sample_weight (np.ndarray | pd.Series | list | None, optional): Optional sample weights.
        adjusted (bool, optional): Whether to use the adjusted balanced accuracy score.
                                   Default is False.

    Returns:
        float: The balanced accuracy score.

    Raises:
        TypeError: If argument types are invalid.
        ValueError: If lengths of inputs are inconsistent.
    """
    return sk_balanced_acc(
        y_true = y_true,
        y_pred = y_pred,
        sample_weight = sample_weight,
        adjusted = adjusted
    )


