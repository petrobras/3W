import numpy as np
import pandas as pd

from typing import Optional, Union
from sklearn.metrics import (
        accuracy_score as sk_acc,
        balanced_accuracy_score as sk_balanced_acc,
        average_precision_score as sk_avg_precision,
        precision_score as sk_precision
)

from ..utils.general_utils import GeneralUtils
from ._metrics_validators import (
        AccuracyScoreArgsValidator, 
        BalancedAccuracyScoreArgsValidator,
        AveragePrecisionScoreArgsValidator,
        PrecisionScoreArgsValidator 
)


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

@GeneralUtils.validate_func_args_with_pydantic(AveragePrecisionScoreArgsValidator)
def average_precision_score(y_true: Union[np.ndarray, pd.Series, list],
                            y_pred: Union[np.ndarray, pd.Series, list],
                            average: Optional[str] = "macro",
                            pos_label: Optional[int] = 1,
                            sample_weight: Optional[Union[np.ndarray, pd.Series, list]] = None) -> float:
    """
    Compute average precision (AP) from prediction scores.

    Args:
        y_true (np.ndarray | pd.Series | list): Ground truth (true binary labels).
        y_pred (np.ndarray | pd.Series | list): Estimated probabilities or decision function.
        average (str, optional): Averaging method - {'weighted', 'micro', 'macro', 'samples', None}.
                                 Default is 'macro'.
        pos_label (int or float, optional): The label of the positive class. Only applied to binary y_true. For multilabel-indicator y_true, pos_label is fixed to 1.
        sample_weight (np.ndarray | pd.Series | list | None, optional): Optional sample weights.

    Returns:
        float: The average precision score.

    Raises:
        TypeError: If input types are invalid.
        ValueError: If inputs are inconsistent or average value is invalid.
    """
    return sk_avg_precision(
        y_true = y_true,
        y_score = y_pred,
        average = average,
        pos_label = pos_label,
        sample_weight = sample_weight
    )

@GeneralUtils.validate_func_args_with_pydantic(PrecisionScoreArgsValidator)
def precision_score(y_true: Union[np.ndarray, pd.Series, list],
                    y_pred: Union[np.ndarray, pd.Series, list],
                    labels: Optional[list] = None,
                    pos_label: Optional[int] = 1,
                    average: Optional[str] = "binary",
                    sample_weight: Optional[Union[np.ndarray, pd.Series, list]] = None,
                    zero_division: Union[str, int] = "warn") -> float:
    """
    Compute the precision score.

    Args:
        y_true (np.ndarray | pd.Series | list): Ground truth (correct) labels.
        y_pred (np.ndarray | pd.Series | list): Predicted labels.
        labels (list | None, optional): The set of labels to include when average != 'binary'.
        pos_label (int | None, optional): Label to report as positive class in binary classification.
        average (str | None, optional): {'binary', 'micro', 'macro', 'samples', 'weighted'} or None.
        sample_weight (np.ndarray | pd.Series | list | None, optional): Sample weights.
        zero_division (str | int, optional): 'warn', 0 or 1. Sets the value to return when there is a zero division.

    Returns:
        float: Precision score.

    Raises:
        TypeError: If input types are invalid.
        ValueError: If input values are invalid.
    """
    return sk_precision(
        y_true = y_true,
        y_pred = y_pred,
        labels = labels,
        pos_label = pos_label,
        average = average,
        sample_weight = sample_weight,
        zero_division = zero_division
    )
