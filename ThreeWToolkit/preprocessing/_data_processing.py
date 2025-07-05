import pandas as pd
import numpy as np

from typing import Literal, Optional, Union

from ..utils.general_utils import GeneralUtils
from ._preprocessing_validators import (
    ImputeMissingArgsValidator,
    NormalizeArgsValidator,
    WindowingArgsValidator,
)

from sklearn.preprocessing import normalize as sk_normalize
from scipy.signal import get_window


@GeneralUtils.validate_func_args_with_pydantic(ImputeMissingArgsValidator)
def impute_missing_data(
    data: Union[pd.DataFrame, pd.Series],
    strategy: Literal["mean", "median", "constant"],
    fill_value: Optional[Union[int, float]] = None,
    columns: Optional[list[str]] = None,
) -> Union[pd.DataFrame, pd.Series]:
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
        data = data.to_frame(name="__temp__")

    cols_to_impute = columns if columns is not None else data.columns.tolist()

    missing = [col for col in cols_to_impute if col not in data.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")

    non_numeric = [
        col for col in cols_to_impute if not pd.api.types.is_numeric_dtype(data[col])
    ]
    if non_numeric:
        raise TypeError(
            f"Only numeric columns can be imputed. Non-numeric columns: {non_numeric}"
        )

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
def normalize(
    X: Union[pd.DataFrame, pd.Series],
    norm: Literal["l1", "l2", "max"] = "l2",
    axis: Optional[Literal[0, 1]] = 1,
    copy_values: Optional[bool] = True,
    return_norm_values: Optional[bool] = False,
) -> Union[pd.DataFrame, pd.Series, tuple]:
    """
    Normalize input data using L1, L2 or max norm.

    Args:
        X (pd.DataFrame | pd.Series): Input data to normalize.
        norm (str): Norm to use ('l1', 'l2', or 'max').
        axis (int): Axis along which to normalize (0 = columns, 1 = rows).
        copy_values (bool): If True, perform normalization on a copy of the input data `X`.
        return_norm_values (bool): If True, also return the computed norm values.

    Returns:
        pd.DataFrame | pd.Series | tuple: Normalized data. If `return_norm_values=True`,
        returns a tuple with normalized data and norms.
    """

    is_series = isinstance(X, pd.Series)
    X_array = X.values.reshape(-1, 1) if is_series else X.values

    normalized = sk_normalize(X_array, norm=norm, axis=axis, copy=copy_values)
    norms = np.linalg.norm(
        X_array, ord={"l1": 1, "l2": 2, "max": np.inf}[norm], axis=axis, keepdims=True
    )

    if is_series:
        normalized = pd.Series(normalized.flatten(), index=X.index, name=X.name)
    else:
        normalized = pd.DataFrame(normalized, index=X.index, columns=X.columns)

    if return_norm_values:
        return normalized, norms
    return normalized


@GeneralUtils.validate_func_args_with_pydantic(WindowingArgsValidator)
def windowing(
    X: pd.Series,
    window: str = "hann",
    window_size: int = 4,
    overlap: float = 0.0,
    normalize: bool = False,
    fftbins: bool = True,
    pad_last_window: bool = False,
    pad_value: float = 0.0,
) -> pd.DataFrame:
    """
    Segment a 1D time-series into overlapping windows and apply a specified windowing function.

    This function divides a 1D signal into segments (windows) of a fixed size,
    applies a window function (e.g., Hann, Hamming), optionally normalizes it,
    and returns the windowed segments in a structured DataFrame format.

    Args:
        X (pd.Series): Input 1D signal to be segmented.
        window (str): Name of the window function to apply (e.g., 'hann', 'hamming', 'boxcar').
        window_size (int): Number of samples in each window.
        overlap (float): Overlap ratio between consecutive windows. Must be in [0, 1).
        normalize (bool): Whether to normalize the window function to have unit area.
        fftbins (bool): Whether to generate the window in FFT-compatible form (True by default).
        pad_last_window (bool): If True, pads the last window to include all remaining samples.
        pad_value (float): Value used to pad the final window if `pad_last_window` is True.

    Returns:
        pd.DataFrame: A DataFrame where each row is a windowed segment of the original signal,
                      columns are named as 'val_1', ..., 'val_N', and an additional column 'win'
                      indicates the window index.
    """
    # Convert Series to NumPy array
    values = X.to_numpy()
    n_samples = len(values)

    # Calculate step size between windows
    step = int(window_size * (1 - overlap))

    # Create the desired window function using scipy.signal.get_window
    win = get_window(window, window_size, fftbins=fftbins)
    # Optionally normalize the window to sum to 1
    if normalize:
        win = win / win.sum()

    windows = []  # List to store each windowed segment
    win_id = 1  # Counter to label window index

    # Slide through the signal with step size, extract window-sized chunks
    for start in range(0, n_samples, step):
        end = start + window_size
        window_vals = values[start:end]

        # If the segment is smaller than window size (last part)
        if len(window_vals) < window_size:
            if pad_last_window:
                # Pad with constant value if requested
                pad = np.full(window_size - len(window_vals), pad_value)
                window_vals = np.concatenate([window_vals, pad])
            else:
                break  # Discard incomplete window if no padding allowe

        # Apply the window function (element-wise multiplication)
        windowed = window_vals * win
        # Append the result along with the window index
        windows.append(np.append(windowed, win_id))
        win_id += 1

    # Define column names: val_1, val_2, ..., val_N, and 'win'
    col_names = [f"val_{i + 1}" for i in range(window_size)] + ["win"]
    # Create DataFrame from all windows
    _temp = pd.DataFrame(windows, columns=col_names)
    # Ensure window index is of integer type
    _temp["win"] = _temp["win"].astype(int)

    return _temp
