import numpy as np
import pandas as pd

from typing import Union
from sklearn.preprocessing import normalize as sk_normalize
from scipy.signal import get_window

from ..core.base_step import BaseStep
from ..core.base_preprocessing import (
    ImputeMissingConfig,
    NormalizeConfig,
    RenameColumnsConfig,
    WindowingConfig,
)


class ImputeMissing(BaseStep):
    """
    A data processing step that handles missing values in numeric columns using various imputation strategies.

    This class supports different imputation methods including mean, median, and constant value filling.
    It can work with both pandas Series and DataFrame inputs, automatically handling the conversion
    and restoration of the original data format.

    Attributes:
        config (ImputeMissingConfig): Configuration object containing imputation parameters
        is_series (bool): Flag to track if input was originally a Series
    """

    def __init__(
        self,
        config: ImputeMissingConfig,
    ):
        """
        Initialize the ImputeMissing step with the provided configuration.

        Args:
            config (ImputeMissingConfig): Configuration containing strategy, columns, and fill_value
        """
        self.config = config

    def pre_process(self, data: pd.DataFrame | pd.Series) -> pd.Series | pd.DataFrame:
        """
        Standardize Series input to DataFrame format for consistent processing.

        This method converts pandas Series to a single-column DataFrame with a temporary
        column name to enable uniform processing in the run method.

        Args:
            data (pd.DataFrame | pd.Series): Input data to be processed

        Returns:
            pd.DataFrame: Data in DataFrame format (original DataFrame or converted Series)
        """
        self.is_series = isinstance(data, pd.Series)
        return data.to_frame(name="__temp__") if self.is_series else data

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the missing value imputation on the specified columns.

        This method performs the core imputation logic, applying the configured strategy
        to fill missing values. It validates column existence, checks for numeric data types,
        and applies the appropriate imputation method.

        Args:
            data (pd.DataFrame): Input DataFrame with potential missing values

        Returns:
            pd.DataFrame: DataFrame with imputed missing values

        Raises:
            TypeError: If non-numeric columns are specified for imputation
            ValueError: If strategy is 'constant' but no fill_value is provided
        """
        # Determine which columns to impute (all columns if none specified)
        cols_to_impute = (
            self.config.columns
            if self.config.columns is not None
            else data.columns.tolist()
        )

        # Check if any specified columns exist in the data
        all_missing = all([col not in data.columns for col in cols_to_impute])
        if all_missing:
            print(f"No column was found to be imputed: {all_missing}")
            return data

        # Filter to only valid columns that exist in the DataFrame
        only_valid_cols_to_impute = [
            col for col in cols_to_impute if col in data.columns
        ]
        if len(only_valid_cols_to_impute) == 0:
            return data

        # Validate that all columns to impute are numeric
        non_numeric = [
            col
            for col in only_valid_cols_to_impute
            if not pd.api.types.is_numeric_dtype(data[col])
        ]
        if non_numeric:
            raise TypeError(
                f"Only numeric columns can be imputed. Non-numeric columns: {non_numeric}"
            )

        # Create a copy to avoid modifying the original data
        data_copy = data.copy()
        # Apply the imputation strategy to each valid column
        for col in only_valid_cols_to_impute:
            if self.config.strategy == "mean":
                # Fill missing values with column mean
                data_copy[col] = data_copy[col].fillna(data_copy[col].mean())
            elif self.config.strategy == "median":
                # Fill missing values with column median
                data_copy[col] = data_copy[col].fillna(data_copy[col].median())
            else:
                # Fill missing values with constant value
                if self.config.fill_value is None:
                    raise ValueError(
                        "You must provide `fill_value` when strategy='constant'"
                    )
                data_copy[col] = data_copy[col].fillna(self.config.fill_value)

        return data_copy

    def post_process(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Restore the original data format (Series or DataFrame).

        If the input was originally a Series, this method extracts the temporary column
        and returns it as a Series with the original structure.

        Args:
            data (pd.DataFrame): Processed DataFrame

        Returns:
            Union[pd.DataFrame, pd.Series]: Data in its original format
        """
        return data["__temp__"] if self.is_series else data


class Normalize(BaseStep):
    """
    A data processing step that normalizes data using different normalization strategies.

    This class applies sklearn's normalize function to scale data according to specified norms
    (l1, l2, or max). It preserves the original pandas structure and can optionally return
    the computed norms along with the normalized data.

    Attributes:
        config (NormalizeConfig): Configuration object containing normalization parameters
        is_series (bool): Flag to track if input was originally a Series
        index (pd.Index): Original index for data reconstruction
        columns (list): Original column names for data reconstruction
    """

    def __init__(
        self,
        config: NormalizeConfig,
    ):
        """
        Initialize the Normalize step with the provided configuration.

        Args:
            config (NormalizeConfig): Configuration containing norm type, axis, and other parameters
        """
        self.config = config

    def pre_process(self, data: pd.DataFrame | pd.Series) -> np.ndarray:
        """
        Convert pandas data to numpy array format and store metadata for reconstruction.

        This method extracts the underlying numpy array from pandas objects while preserving
        the necessary metadata (index, columns) to reconstruct the original structure later.

        Args:
            data (pd.DataFrame | pd.Series): Input data to be normalized

        Returns:
            np.ndarray: Numpy array ready for normalization (2D for DataFrame, reshaped for Series)
        """
        self.is_series = isinstance(data, pd.Series)
        self.index = data.index
        self.columns = data.columns if isinstance(data, pd.DataFrame) else [data.name]

        if self.is_series:
            if not pd.api.types.is_numeric_dtype(data):
                raise TypeError("Series must be numeric")
            return data.values.reshape(-1, 1)

        non_numeric_cols = [
            col for col in data.columns if not pd.api.types.is_numeric_dtype(data[col])
        ]
        if non_numeric_cols:
            raise TypeError("Non-numeric columns")

        return data.values

    def run(self, X_array: np.ndarray) -> tuple:
        """
        Apply normalization to the input array and compute norms.

        This method uses sklearn's normalize function to scale the data according to the
        specified norm and axis. It also computes the norms of the original data for
        potential later use or analysis.

        Args:
            X_array (np.ndarray): Input array to be normalized

        Returns:
            tuple: (normalized_array, norms_array) containing the normalized data and computed norms
        """
        # Apply sklearn normalization
        normalized = sk_normalize(
            X_array,
            norm=self.config.norm,
            axis=self.config.axis,
            copy=self.config.copy_values,
        )
        # Compute norms of the original data
        norms = np.linalg.norm(
            X_array,
            ord={"l1": 1, "l2": 2, "max": np.inf}[self.config.norm],
            axis=self.config.axis,
            keepdims=True,
        )

        return normalized, norms

    def post_process(self, result: tuple) -> pd.DataFrame | pd.Series | tuple:
        """
        Reconstruct pandas objects from numpy arrays and return results based on configuration.

        This method converts the normalized numpy arrays back to the original pandas format
        (Series or DataFrame) using the stored metadata. It can return either just the
        normalized data or a tuple including the computed norms.

        Args:
            result (tuple): Tuple containing (normalized_array, norms_array)

        Returns:
            pd.DataFrame | pd.Series | tuple: Normalized data in original format,
                                                   optionally with norms if configured
        """
        normalized, norms = result

        if self.is_series:
            # Reconstruct Series from flattened array
            normalized = pd.Series(
                normalized.flatten(), index=self.index, name=self.columns[0]
            )
        else:
            # Reconstruct DataFrame with original structure
            normalized = pd.DataFrame(
                normalized, index=self.index, columns=self.columns
            )

        # Return normalized data only, or tuple with norms based on configuration
        return (normalized, norms) if self.config.return_norm_values else normalized


class Windowing(BaseStep):
    """
    A data processing step that applies windowing techniques to time series data.

    This class creates overlapping or non-overlapping windows from time series data,
    applying window functions for signal processing. It supports multiple variables
    and various window types from scipy.signal.

    Attributes:
        config (WindowingConfig): Configuration object containing windowing parameters
    """

    def __init__(
        self,
        config: WindowingConfig,
    ):
        """
        Initialize the Windowing step with the provided configuration.

        Args:
            config (WindowingConfig): Configuration containing window parameters like size,
                                    overlap, type, and padding options
        """
        self.config = config

    def pre_process(self, x: pd.DataFrame | pd.Series) -> np.ndarray:
        """
        Convert input data to numpy array format for windowing operations.

        This method standardizes the input format by converting pandas objects to numpy arrays.
        Series are reshaped to column vectors, while DataFrames maintain their 2D structure.

        Args:
            x (pd.DataFrame | pd.Series): Input time series data

        Returns:
            np.ndarray: 2D array with shape (samples, variables)

        Raises:
            ValueError: If input is neither pandas Series nor DataFrame
        """
        if isinstance(x, pd.Series):
            return x.to_numpy().reshape(-1, 1)
        elif isinstance(x, pd.DataFrame):
            return x.to_numpy()
        else:
            raise ValueError("Input must be either pandas Series or DataFrame")

    def _check_window_size_vs_data(self, values: np.ndarray):
        """
        Validate that the window size is appropriate for the data length.

        This private method ensures that the configured window size does not exceed
        the available data length, which would make windowing impossible.

        Args:
            values (np.ndarray): Input data array

        Raises:
            ValueError: If window_size exceeds the number of samples in the data
        """
        n_samples = values.shape[0]
        if self.config.window_size > n_samples:
            raise ValueError(
                "`window_size` must be smaller than or equal to the length of X."
            )

    def run(self, values: np.ndarray) -> pd.DataFrame:
        """
        Apply windowing to the input data and create a structured DataFrame output.

        This method performs the core windowing operation by:
        1. Validating window size against data length
        2. Creating window function from scipy.signal
        3. Extracting overlapping windows with specified step size
        4. Applying window function to each extracted window
        5. Flattening multivariate windows and adding window IDs

        Args:
            values (np.ndarray): Input time series data (samples x variables)

        Returns:
            pd.DataFrame: DataFrame with columns for each time step of each variable
                         plus a 'win' column containing window IDs
        """
        self._check_window_size_vs_data(values)

        # Ensure data is 2D (samples x variables)
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        n_samples, n_variables = values.shape
        # Calculate step size based on overlap configuration
        step = int(self.config.window_size * (1 - self.config.overlap))

        ## Create window function using scipy.signal
        win = get_window(
            self.config.window, self.config.window_size, fftbins=self.config.fftbins
        )
        # Normalize window if requested
        if self.config.normalize:
            win = win / win.sum()

        windows = []
        win_id = 1

        # Extract windows with specified step size
        for start in range(0, n_samples, step):
            end = start + self.config.window_size
            window_vals = values[start:end]  # Shape: (window_size, n_variables)

            # Handle the last window if it's shorter than window_size
            if len(window_vals) < self.config.window_size:
                if self.config.pad_last_window:
                    # Pad the last window to maintain consistent size
                    pad_size = self.config.window_size - len(window_vals)
                    pad = np.full(
                        (pad_size, n_variables),
                        self.config.pad_value,
                    )
                    window_vals = np.concatenate([window_vals, pad], axis=0)
                else:
                    # Skip incomplete windows if padding is disabled
                    break

            # Apply window function to all variables (broadcasting)
            windowed = window_vals * win.reshape(-1, 1)
            # Flatten windowed data: [var1_t0, var1_t1, ..., var2_t0, var2_t1, ...]
            # Transpose to get variables as rows, then flatten for desired column order
            windowed_flat = (
                windowed.T.flatten()
            )  # Transpose and flatten to obtain the desired order

            # Add window ID to the flattened data
            window_row = np.append(windowed_flat, win_id)
            windows.append(window_row)
            win_id += 1

        # Generate column names for the output DataFrame
        col_names = []
        for var_idx in range(n_variables):
            var_name = f"var{var_idx + 1}"
            for t in range(self.config.window_size):
                col_names.append(f"{var_name}_t{t}")
        col_names.append("win")  # Window ID column

        return pd.DataFrame(windows, columns=col_names)

    def post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply final formatting to the windowed DataFrame.

        This method ensures that the window ID column has the correct integer data type
        for proper indexing and analysis.

        Args:
            df (pd.DataFrame): DataFrame with windowed data

        Returns:
            pd.DataFrame: DataFrame with properly formatted window IDs
        """
        df["win"] = df["win"].astype(int)
        return df


class RenameColumns(BaseStep):
    """
    A simple data processing step that renames DataFrame columns according to a mapping.

    This class provides a clean interface for renaming columns in a pandas DataFrame
    using a dictionary mapping from old names to new names.

    Attributes:
        config (RenameColumnsConfig): Configuration object containing the column mapping
    """

    def __init__(self, config: RenameColumnsConfig):
        """
        Initialize the RenameColumns step with the provided configuration.

        Args:
            config (RenameColumnsConfig): Configuration containing the columns_map dictionary
        """
        self.config = config

    def pre_process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create a copy of the input DataFrame to avoid modifying the original.

        Args:
            data (pd.DataFrame): Input DataFrame to be processed

        Returns:
            pd.DataFrame: Copy of the input DataFrame
        """
        return data.copy()

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns according to the configured mapping.

        This method applies the column renaming using pandas' rename method with the
        mapping provided in the configuration.

        Args:
            df (pd.DataFrame): DataFrame with columns to be renamed

        Returns:
            pd.DataFrame: DataFrame with renamed columns
        """
        if df.columns.duplicated().any():
            duplicated = df.columns[df.columns.duplicated()].unique().tolist()
            raise ValueError(f"Duplicate column names found in DataFrame: {duplicated}")

        missing = [col for col in self.config.columns_map if col not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")

        return df.rename(columns=self.config.columns_map)

    def post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return the processed DataFrame without additional modifications.

        Args:
            df (pd.DataFrame): DataFrame with renamed columns

        Returns:
            pd.DataFrame: The same DataFrame (no additional processing needed)
        """
        return df
