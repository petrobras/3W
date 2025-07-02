import pandas as pd
import numpy as np

from typing import Literal, Optional, Union

from ..utils.general_utils import GeneralUtils
from ._preprocessing_validators import ImputeMissingArgsValidator, NormalizeArgsValidator

from sklearn.preprocessing import (
    normalize as sk_normalize
)

@GeneralUtils.validate_func_args_with_pydantic(ImputeMissingArgsValidator)
def impute_missing_data(data: Union[pd.DataFrame, pd.Series],
                        strategy: Literal["mean", "median", "constant"],
                        fill_value: Optional[Union[int, float]] = None,
                        columns: Optional[list[str]] = None) -> Union[pd.DataFrame, pd.Series]:
    """
    Imputes missing values (NaNs) in specified columns of a DataFrame or Series
    using the given strategy.

    Args:
        data (pd.DataFrame | pd.Series): Input data containing missing values to impute.
        strategy (str): Imputation strategy. Must be one of 'mean', 'median', or 'constant'.
        fill_value (int | float, optional): Constant value to use if strategy is 'constant'.
            Must be provided in that case. Default is None.
        columns (list[str], optional): List of columns to impute. If None, all columns are imputed.
            Applicable only if `data` is a DataFrame.

    Returns:
        pd.DataFrame | pd.Series: Data with missing values imputed according to the strategy.
            Returns a Series if input was a Series; otherwise, returns a DataFrame.

    Raises:
        ValueError: If any column in `columns` does not exist in the DataFrame.
        TypeError: If any target column is not numeric.
        ValueError: If strategy is 'constant' and `fill_value` is not provided.
    """

    is_series = isinstance(data, pd.Series)
    if is_series:
        data = data.to_frame(name = "__temp__")

    cols_to_impute = columns if columns is not None else data.columns.tolist()

    missing = [col for col in cols_to_impute if col not in data.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")

    non_numeric = [col for col in cols_to_impute if not pd.api.types.is_numeric_dtype(data[col])]
    if non_numeric:
        raise TypeError(f"Only numeric columns can be imputed. Non-numeric columns: {non_numeric}")

    data_copy = data.copy()
    for col in cols_to_impute:
        if strategy == "mean":
            data_copy[col] = data_copy[col].fillna(data_copy[col].mean())
        elif strategy == "median":
            data_copy[col] = data_copy[col].fillna(data_copy[col].median())
        else:
            data_copy[col] = data_copy[col].fillna(fill_value)

    return data_copy["__temp__"] if is_series else data_copy

@GeneralUtils.validate_func_args_with_pydantic(NormalizeArgsValidator)
def normalize(X: Union[pd.DataFrame, pd.Series],
              norm: Literal["l1", "l2", "max"] = "l2",
              axis: Optional[Literal[0, 1]] = 1,
              copy_values: Optional[bool] = True,
              return_norm_values: Optional[bool] = False) -> Union[pd.DataFrame, pd.Series, tuple]:
    """
    Normalize input data using L1, L2 or max norm.

    Args:
        X (pd.DataFrame | pd.Series): Input data to normalize.
        norm (str): Norm to use ('l1', 'l2', or 'max').
        axis (int): Axis along which to normalize (0 = columns, 1 = rows).
        copy_values (bool): If True, perform normalization on a copy_values.
        return_norm_values (bool): If True, also return the computed norm values.

    Returns:
        pd.DataFrame | pd.Series | tuple: Normalized data. If `return_norm_values=True`,
        returns a tuple (normalized_data, norms).
    """

    is_series = isinstance(X, pd.Series)
    X_array = X.values.reshape(-1, 1) if is_series else X.values

    normalized = sk_normalize(X_array, norm = norm, axis = axis, copy = copy_values)
    norms = np.linalg.norm(X_array, ord = {"l1": 1, "l2": 2, "max": np.inf}[norm], axis = axis, keepdims = True)

    if is_series:
        normalized = pd.Series(normalized.flatten(), index = X.index, name = X.name)
    else:
        normalized = pd.DataFrame(normalized, index = X.index, columns = X.columns)

    if return_norm_values:
        return normalized, norms
    return normalized