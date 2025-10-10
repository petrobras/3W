import numpy as np
import pandas as pd

from sklearn.metrics import explained_variance_score as sk_explained_variance

from ..utils.general_utils import GeneralUtils
from ..core.base_metrics import ExplainedVarianceScoreConfig


@GeneralUtils.validate_func_args_with_pydantic(ExplainedVarianceScoreConfig)
def explained_variance_score(
    y_true: np.ndarray | pd.Series | list,
    y_pred: np.ndarray | pd.Series | list,
    sample_weight: np.ndarray | pd.Series | list | None = None,
    multioutput: str = "uniform_average",
    force_finite: bool = True,
) -> float:
    """
    Compute the explained variance regression score.

    Args:
        y_true: Ground truth target values.
        y_pred: Estimated target values.
        sample_weight: Sample weights.
        multioutput: {'raw_values', 'uniform_average', 'variance_weighted'}.
        force_finite: If True, only finite values are allowed in the output.

    Returns:
        Explained variance score.

    Raises:
        ValueError, TypeError: For invalid inputs or arguments.
    """
    return sk_explained_variance(
        y_true=y_true,
        y_pred=y_pred,
        sample_weight=sample_weight,
        multioutput=multioutput,
        force_finite=force_finite,
    )
