import numpy as np
import pandas as pd


"""Signals in dataset with few samples (or no samples at all)."""
UNUSED_TAGS = [
    "P-JUS-BS",  # zero instances
    "P-MON-SDV-P",  # zero instances
    "PT-P",  # zero instances
    "QBS",  # zero instances
    "P-MON-CKGL",  # only two events have this tag non-NA
    "state",
]

"""Faulty sensors may give wrong readings. Values outside this range should be discarded."""
VALID_RANGE = {
    # choke states are between 0 and 100
    "ABER-CKGL": (0, 100),  # Gas-lift choke value
    "ABER-CKP": (0, 100),  # Production choke value
    # state variables must assume 0, 0.5 or 1.0 only
    "ESTADO-DHSV": (0, 1),
    "ESTADO-M1": (0, 1),
    "ESTADO-M2": (0, 1),
    "ESTADO-PXO": (0, 1),
    "ESTADO-SDV-GL": (0, 1),
    "ESTADO-SDV-P": (0, 1),
    "ESTADO-SDV-W1": (0, 1),
    "ESTADO-SDV-W2": (0, 1),
    "ESTADO-XO": (0, 1),
    # pressure readings
    "P-ANULAR": (0, 10e7),  # highest reliable reading is 4.35e+07
    "P-JUS-CKGL": (
        -1e7,
        10e7,
    ),  # highest reliable reading is 4.52e+07, TODO: check negatives
    "P-JUS-CKP": (0, 10e7),  # highest reliable reading is 1.01e+07
    "P-PDG": (0, 10e7),  # highest reliable reading is 4.91e+07
    "P-TPT": (0, 10e7),  # highest reliable reading is 8.17e+07
    # temperature readings
    "T-JUS-CKP": (-15, 150),  # from data analysis
    "T-MON-CKP": (-10, 150),
    "T-PDG": (0, 300),  # highest reliable reading is 2.76e2
    "T-TPT": (-5, 150),
}


"""Faulty sensors may give frozen readings inside the valid range. If sensor variation is below threshold,
   we should discard the data."""
DEVIATION_THRESHOLD = {
    # unclear what to do with state variables
    # pressure readings
    "P-ANULAR": 1e2,
    "P-JUS-CKGL": 1e0,
    "P-JUS-CKP": 1e0,
    "P-PDG": 1e-3,
    "P-TPT": 1e-3,
    # temperature readings
    "T-JUS-CKP": 1e-5,
    "T-MON-CKP": 1e-2,
    "T-PDG": 1e-5,
    "T-TPT": 1e-7,
}

"""Averages and standard deviations taken from cleaned up data.
   Cleaned up data means (in order):
       * tags in UNUSED_TAGS are removed from the dataset
       * all values outside the ranges `VALID_RANGE` are removed
       * all values within an event where the sensor appears frozen (standard deviation below threshold) are removed
"""
GLOBAL_AVERAGES = {  # computed from cleaned up data
    "ABER-CKGL": 1.931843e01,
    "ABER-CKP": 4.239023e01,
    "ESTADO-DHSV": 5.703575e-01,
    "ESTADO-M1": 8.532889e-01,
    "ESTADO-M2": 3.172376e-01,
    "ESTADO-PXO": 8.822910e-03,
    "ESTADO-SDV-GL": 5.448040e-01,
    "ESTADO-SDV-P": 9.110019e-01,
    "ESTADO-W1": 7.026502e-01,
    "ESTADO-W2": 2.407879e-01,
    "ESTADO-XO": 3.095789e-03,
    "P-ANULAR": 1.543509e07,
    "P-JUS-CKGL": 1.633026e07,
    "P-JUS-CKP": 1.848287e06,
    "P-MON-CKP": 3.373246e06,
    "P-PDG": 2.178213e07,
    "P-TPT": 1.397929e07,
    "QGL": 1.304284e00,
    "T-JUS-CKP": 6.926409e01,
    "T-MON-CKP": 7.168256e01,
    "T-PDG": 7.412067e01,
    "T-TPT": 9.690458e01,
}

GLOBAL_STDS = {  # computed from cleaned up data
    "ABER-CKGL": 3.040066e01,
    "ABER-CKP": 2.847642e01,
    "ESTADO-DHSV": 4.950251e-01,
    "ESTADO-M1": 3.538177e-01,
    "ESTADO-M2": 4.653943e-01,
    "ESTADO-PXO": 9.351506e-02,
    "ESTADO-SDV-GL": 4.979886e-01,
    "ESTADO-SDV-P": 2.847410e-01,
    "ESTADO-W1": 4.570918e-01,
    "ESTADO-W2": 4.275588e-01,
    "ESTADO-XO": 5.555363e-02,
    "P-ANULAR": 6.251561e06,
    "P-JUS-CKGL": 1.227916e07,
    "P-JUS-CKP": 2.069025e06,
    "P-MON-CKP": 3.175275e06,
    "P-PDG": 8.057268e06,
    "P-TPT": 5.887995e06,
    "QGL": 1.541538e00,
    "T-JUS-CKP": 1.903404e01,
    "T-MON-CKP": 2.898848e01,
    "T-PDG": 2.331257e01,
    "T-TPT": 3.053168e01,
}


def default_data_cleanup(
    data: pd.DataFrame, target_column: str | None = None, *args, **kwargs
) -> pd.DataFrame:
    """Apply default cleanup for signal dataframes.

    Removes unused tags, frozen sensors and out-of-range sensor readings.

    Args:
        data (pd.DataFrame): Raw signal data to be cleaned.
        target_column (str | None, optional): Name of the target column to exclude
            from cleanup. Defaults to None.
        *args: Additional positional arguments (unused).
        **kwargs: Additional keyword arguments (unused).

    Returns:
        pd.DataFrame: Cleaned dataframe with invalid readings replaced by NaN.

    Notes:
        - Drops columns listed in UNUSED_TAGS
        - Replaces frozen sensor readings (std < threshold) with NaN
        - Replaces out-of-range values with NaN based on VALID_RANGE
    """

    # Drop unused columns
    columns_to_drop = UNUSED_TAGS.copy()
    if target_column is not None:
        columns_to_drop += [target_column]

    # Make sure columns to drop are actually in the data
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]

    data = data.drop(columns_to_drop, axis=1)

    # Per-column cleanup
    for tag in data.columns:
        # Clean stuck sensors
        if tag in DEVIATION_THRESHOLD:
            if data[tag].std() < DEVIATION_THRESHOLD[tag]:
                data[tag] = np.nan

        # Remove values outside valid range
        if tag in VALID_RANGE:
            lower, upper = VALID_RANGE[tag]
            data[tag] = data[tag].where(data[tag].between(lower, upper), other=np.nan)

    return data


def default_data_normalization(
    data: pd.DataFrame, target_column: str | None = None, *args, **kwargs
) -> pd.DataFrame:
    """Apply default scaling on data.

    Uses global averages and standard deviations computed from canonical cleaned values
    to perform z-score normalization.

    Args:
        data (pd.DataFrame): Data to be normalized.
        target_column (str | None, optional): Name of the target column to exclude
            from normalization. Defaults to None.
        *args: Additional positional arguments (unused).
        **kwargs: Additional keyword arguments (unused).

    Returns:
        pd.DataFrame: Normalized dataframe with z-score scaling applied.

    Notes:
        - Uses (data - mean) / std normalization
        - Global statistics are pre-computed from cleaned training data
        - Target column is excluded from normalization if specified
    """

    # Filter loaded signals
    selected_columns = data.columns

    # Remove target column
    if target_column is not None:
        selected_columns = [col for col in selected_columns if col != target_column]

    # Remove "class" column from normalization calculations
    selected_columns = [col for col in selected_columns if col != "class"]

    avg = pd.Series({tag: GLOBAL_AVERAGES[tag] for tag in selected_columns})
    std = pd.Series({tag: GLOBAL_STDS[tag] for tag in selected_columns})
    return (data - avg) / std


def default_label_handling(data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """Apply default label processing.

    Args:
        data (pd.DataFrame): labels to be adjusted.
        *args: Additional positional arguments (unused).
        **kwargs: Additional keyword arguments (unused).

    Returns:
        pd.DataFrame: Adjusted dataframe with modified "class" column.

    Notes:
        - Maps transient labels to corresponding faults
        - Annotation gaps are filled with adjacent valid labels, forward first, then backwards
    """
    s = data["class"]
    s = s % 100  # map transients to faults
    s = s.ffill()  # forward fill of gaps in annotations
    s = s.bfill()  # backward fill of holes in annotations
    data["class"] = s
    return data


def default_data_processing(
    data: dict[str, pd.DataFrame],
    fillna: bool = True,
    target_column: str | None = None,
    fill_target_value: int | None = None,
    *args,
    **kwargs,
) -> pd.DataFrame:
    """Apply default cleaning and scaling on data and labels.

    Performs a complete data processing pipeline including cleanup, normalization, and missing value imputation.
    Maps transient labels to the corresponding fault labels, and fills annotation gaps with the nearest labels in the series.

    Args:
        data (dict[str, pd.DataFrame]): Dictionary containing 'signal' and 'label' dataframes.
        fillna (bool, optional): Whether to fill missing values with 0. Defaults to True.
        target_column (str | None, optional): Name of the target column. Defaults to None.
        fill_target_value (int | None, optional): Value to fill in the target class column.
            Defaults to None.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        pd.DataFrame: Processed data dictionary with cleaned and normalized signals.

    Notes:
        - Applies cleanup to remove invalid sensor readings
        - Applies z-score normalization using global statistics
        - Fills missing values with 0 (which corresponds to the global mean after normalization)
        - TODO: Implement normalization for labels in regression tasks
    """

    # Signal processing
    data["signal"] = default_data_cleanup(data["signal"], target_column)
    data["signal"] = default_data_normalization(data["signal"], target_column)
    if fillna:
        data["signal"] = data["signal"].fillna(0)

    # Label processing
    # Obs: When target_column is None, "label" is not in data
    if "label" in data:
        data["label"] = default_label_handling(data["label"])

    # TODO: Implement normalization labels for regression tasks

    return data
