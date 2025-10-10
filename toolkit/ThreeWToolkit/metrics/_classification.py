import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score as sk_acc,
    balanced_accuracy_score as sk_balanced_acc,
    average_precision_score as sk_avg_precision,
    precision_score as sk_precision,
    recall_score as sk_recall,
    f1_score as sk_f1,
    roc_auc_score as sk_roc_auc,
)

from ..utils.general_utils import GeneralUtils
from ..core.base_metrics import (
    AccuracyScoreConfig,
    BalancedAccuracyScoreConfig,
    AveragePrecisionScoreConfig,
    PrecisionScoreConfig,
    RecallScoreConfig,
    F1ScoreConfig,
    RocAucScoreConfig,
)


@GeneralUtils.validate_func_args_with_pydantic(AccuracyScoreConfig)
def accuracy_score(
    y_true: np.ndarray | pd.Series | list,
    y_pred: np.ndarray | pd.Series | list,
    normalize: bool = True,
    sample_weight: np.ndarray | pd.Series | list | None = None,
) -> float:
    """
    Calculates the accuracy between true and predicted values.

    This function validates the arguments using Pydantic to ensure
    correct types and compatible sizes before calculating the metric.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        normalize: If True, returns the fraction correct.
                   If False, returns the number of correct samples.
                   Default is True.
        sample_weight: Sample weights. Default is None.

    Returns:
        The calculated accuracy.

    Raises:
        TypeError: If any argument has an invalid type.
        ValueError: If array dimensions are incompatible.
    """
    return sk_acc(
        y_true=y_true, y_pred=y_pred, normalize=normalize, sample_weight=sample_weight
    )


@GeneralUtils.validate_func_args_with_pydantic(BalancedAccuracyScoreConfig)
def balanced_accuracy_score(
    y_true: np.ndarray | pd.Series | list,
    y_pred: np.ndarray | pd.Series | list,
    sample_weight: np.ndarray | pd.Series | list | None = None,
    adjusted: bool = False,
) -> float:
    """
    Computes the balanced accuracy between true and predicted values.

    This metric is robust to imbalanced datasets by averaging recall obtained on each class.

    Args:
        y_true: True class labels.
        y_pred: Predicted class labels.
        sample_weight: Optional sample weights.
        adjusted: Whether to use the adjusted balanced accuracy score.
                  Default is False.

    Returns:
        The balanced accuracy score.

    Raises:
        TypeError: If argument types are invalid.
        ValueError: If lengths of inputs are inconsistent.
    """
    return sk_balanced_acc(
        y_true=y_true, y_pred=y_pred, sample_weight=sample_weight, adjusted=adjusted
    )


@GeneralUtils.validate_func_args_with_pydantic(AveragePrecisionScoreConfig)
def average_precision_score(
    y_true: np.ndarray | pd.Series | list,
    y_pred: np.ndarray | pd.Series | list,
    average: str | None = "macro",
    pos_label: int | None = 1,
    sample_weight: np.ndarray | pd.Series | list | None = None,
) -> float:
    """
    Compute average precision (AP) from prediction scores.

    Args:
        y_true: Ground truth (true binary labels).
        y_pred: Estimated probabilities or decision function.
        average: Averaging method - {'weighted', 'micro', 'macro', 'samples', None}.
                 Default is 'macro'.
        pos_label: The label of the positive class. Only applied to binary y_true.
                   For multilabel-indicator y_true, pos_label is fixed to 1.
        sample_weight: Optional sample weights.

    Returns:
        The average precision score.

    Raises:
        TypeError: If input types are invalid.
        ValueError: If inputs are inconsistent or average value is invalid.
    """
    return sk_avg_precision(
        y_true=y_true,
        y_score=y_pred,
        average=average,
        pos_label=pos_label,
        sample_weight=sample_weight,
    )


@GeneralUtils.validate_func_args_with_pydantic(PrecisionScoreConfig)
def precision_score(
    y_true: np.ndarray | pd.Series | list,
    y_pred: np.ndarray | pd.Series | list,
    labels: list | None = None,
    pos_label: int = 1,
    average: str = "binary",
    sample_weight: np.ndarray | pd.Series | list | None = None,
    zero_division: str | int = "warn",
) -> float:
    """
    Compute the precision score.

    Args:
        y_true: Ground truth (correct) labels.
        y_pred: Predicted labels.
        labels: The set of labels to include when average != 'binary'.
        pos_label: Label to report as positive class in binary classification.
        average: {'binary', 'micro', 'macro', 'samples', 'weighted'} or None.
        sample_weight: Sample weights.
        zero_division: 'warn', 0 or 1. Sets the value to return when there is a zero division.

    Returns:
        Precision score.

    Raises:
        TypeError: If input types are invalid.
        ValueError: If input values are invalid.
    """
    return sk_precision(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )


@GeneralUtils.validate_func_args_with_pydantic(RecallScoreConfig)
def recall_score(
    y_true: np.ndarray | pd.Series | list,
    y_pred: np.ndarray | pd.Series | list,
    labels: list | None = None,
    pos_label: int | None = 1,
    average: str | None = "binary",
    sample_weight: np.ndarray | pd.Series | list | None = None,
    zero_division: str | int = "warn",
) -> float:
    """
    Compute the recall score.

    Args:
        y_true: Ground truth (correct) labels.
        y_pred: Predicted labels.
        labels: The set of labels to include when average != 'binary'.
        pos_label: Label to report as positive class in binary classification.
        average: {'binary', 'micro', 'macro', 'samples', 'weighted'} or None.
        sample_weight: Sample weights.
        zero_division: 'warn', 0 or 1. Sets the value to return when there is a zero division.

    Returns:
        Recall score.

    Raises:
        TypeError: If input types are invalid.
        ValueError: If input values are invalid.
    """
    return sk_recall(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )


@GeneralUtils.validate_func_args_with_pydantic(F1ScoreConfig)
def f1_score(
    y_true: np.ndarray | pd.Series | list,
    y_pred: np.ndarray | pd.Series | list,
    labels: list | None = None,
    pos_label: int | None = 1,
    average: str | None = "binary",
    sample_weight: np.ndarray | pd.Series | list | None = None,
    zero_division: str | int = "warn",
) -> float:
    """
    Compute the F1 score.

    Args:
        y_true: Ground truth (correct) labels.
        y_pred: Predicted labels.
        labels: The set of labels to include when average != 'binary'.
        pos_label: Label to report as positive class in binary classification.
        average: {'binary', 'micro', 'macro', 'samples', 'weighted'} or None.
        sample_weight: Sample weights.
        zero_division: 'warn', 0 or 1. Value to return when there is a zero division.

    Returns:
        F1 score.

    Raises:
        TypeError: If input types are invalid.
        ValueError: If input values are invalid.
    """
    return sk_f1(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )


@GeneralUtils.validate_func_args_with_pydantic(RocAucScoreConfig)
def roc_auc_score(
    y_true: np.ndarray | pd.Series | list,
    y_pred: np.ndarray | pd.Series | list,
    average: str | None = "macro",
    sample_weight: np.ndarray | pd.Series | list | None = None,
    max_fpr: float | None = None,
    multi_class: str = "raise",
    labels: list | None = None,
) -> float:
    """
    Compute the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    Args:
        y_true: True binary or multiclass labels.
        y_pred: Target scores, can either be probability estimates or confidence values.
        average: One of ['micro', 'macro', 'samples', 'weighted', None]. Default is 'macro'.
        sample_weight: Sample weights.
        max_fpr: If not None, the standardized partial AUC over the range [0, max_fpr].
        multi_class: {'raise', 'ovr', 'ovo'}. Only used for multiclass targets.
        labels: List of labels to index the classes in y_true and y_pred.

    Returns:
        ROC AUC score.

    Raises:
        ValueError, TypeError: For invalid inputs or arguments.
    """
    return sk_roc_auc(
        y_true=y_true,
        y_score=y_pred,
        average=average,
        sample_weight=sample_weight,
        max_fpr=max_fpr,
        multi_class=multi_class,
        labels=labels,
    )
