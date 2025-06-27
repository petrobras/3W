import numpy as np
import pandas as pd

from typing import Optional, Union
from sklearn.metrics import (
        explained_variance_score as sk_explained_variance
)

from ..utils.general_utils import GeneralUtils
from ._metrics_validators import (
    ExplainedVarianceScoreArgsValidator
)

@GeneralUtils.validate_func_args_with_pydantic(ExplainedVarianceScoreArgsValidator)
def explained_variance_score(y_true: Union[np.ndarray, pd.Series, list],
                             y_pred: Union[np.ndarray, pd.Series, list],
                             sample_weight: Optional[Union[np.ndarray, pd.Series, list]] = None,
                             multioutput: str = "uniform_average",
                             force_finite: bool = True) -> float:
    """
    Compute the explained variance regression score.

    Args:
        y_true (np.ndarray | pd.Series | list): Ground truth target values.
        y_pred (np.ndarray | pd.Series | list): Estimated target values.
        sample_weight (np.ndarray | pd.Series | list | None): Sample weights.
        multioutput (str): {'raw_values', 'uniform_average', 'variance_weighted'}.
        force_finite (bool): If True, only finite values are allowed in the output.

    Returns:
        float: Explained variance score.

    Raises:
        ValueError, TypeError: For invalid inputs or arguments.
    """
    return sk_explained_variance(
        y_true = y_true,
        y_pred = y_pred,
        sample_weight = sample_weight,
        multioutput = multioutput,
        force_finite = force_finite
    )