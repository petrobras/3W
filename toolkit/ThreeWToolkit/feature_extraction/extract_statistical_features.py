import numpy as np
import pandas as pd

from scipy import stats
from ..core.base_feature_extractor import StatisticalConfig
from ..core.base_step import BaseStep


class ExtractStatisticalFeatures(BaseStep):
    """
    Extracts statistical features from windowed time series data.

    Supports both univariate and multivariate analysis.

    IMPORTANT: Data must be already windowed. Each row should represent a window.
    If data is not windowed, use the Windowing class first.

    Input format: DataFrame with windowed data where each row is a window
    Output format:
    - Univariate: [var1_feature1, var1_feature2, ..., label]
    - Multivariate: [var1_feature1, var2_feature1, ..., var1_feature2, var2_feature2, ..., label]
    """

    FEATURES = ["mean", "std", "skew", "kurt", "min", "1qrt", "med", "3qrt", "max"]

    def __init__(self, config: StatisticalConfig):
        """
        Initialize the feature extractor.

        Args:
            config: Configuration object with feature extraction parameters
        """
        super().__init__()

        self.window_size = config.window_size
        self.overlap = config.overlap
        self.offset = config.offset
        self.eps = config.eps

        self.selected_features = (
            config.selected_features if config.selected_features else self.FEATURES
        )
        self.multivariate = getattr(config, "multivariate", True)
        self.is_windowed = getattr(config, "is_windowed", False)
        self.label_column = getattr(config, "label_column", None)

    def _identify_variables(self, data: pd.DataFrame) -> dict:
        """Identifies variables in the data based on naming pattern."""
        columns = data.columns.tolist()
        has_label = self.label_column is not None and self.label_column in columns

        # Remove label from columns if it exists
        if has_label:
            columns = [col for col in columns if col != self.label_column]

        if not columns:
            raise ValueError("No variable columns found in the data")

        # Extract variable numbers using string manipulation
        var_numbers = set()

        for col in columns:
            if col.startswith("var") and "_" in col:
                # Extract the part between "var" and "_"
                try:
                    var_part = col[3:]  # Remove "var"
                    underscore_pos = var_part.find("_")
                    if underscore_pos > 0:
                        var_number_str = var_part[:underscore_pos]
                        if var_number_str.isdigit():
                            var_numbers.add(int(var_number_str))
                except (ValueError, IndexError):
                    continue  # Ignore columns with invalid format

        if not var_numbers:
            raise ValueError("No variables with pattern 'varX_' found in columns")

        # Organize variables by number
        variables = {}
        for var_num in sorted(var_numbers):
            var_cols = [col for col in columns if col.startswith(f"var{var_num}_")]
            if var_cols:  # Only add if there are columns for this variable
                variables[var_num] = var_cols

        return variables

    def _calculate_statistics(self, data_array: np.ndarray) -> dict:
        """
        Calculate statistics using numpy/scipy in an optimized way.

        Args:
            data_array: 2D array where each row is a window

        Returns:
            Dictionary with calculated statistics
        """
        if data_array.size == 0:
            return {}

        # If 1D, convert to 2D (1 window)
        if data_array.ndim == 1:
            data_array = data_array.reshape(1, -1)

        stats_dict = {}

        # Basic statistics - vectorized
        if "mean" in self.selected_features:
            stats_dict["mean"] = np.mean(data_array, axis=1)

        if "std" in self.selected_features:
            stats_dict["std"] = np.std(data_array, axis=1, ddof=0)

        # For skew and kurtosis, we need to handle special cases
        std_values = (
            np.std(data_array, axis=1, ddof=0)
            if "skew" in self.selected_features or "kurt" in self.selected_features
            else None
        )

        # For skew and kurtosis, we need to handle special cases
        if "skew" in self.selected_features or "kurt" in self.selected_features:
            std_values = np.std(data_array, axis=1, ddof=0)

            if "skew" in self.selected_features:
                skew_values = np.full(data_array.shape[0], 0.0)
                valid_mask = std_values > self.eps  # Use eps to avoid division by zero
                if np.any(valid_mask):
                    valid_data = data_array[valid_mask]
                    skew_values[valid_mask] = stats.skew(valid_data, axis=1)
                stats_dict["skew"] = skew_values

            if "kurt" in self.selected_features:
                kurt_values = np.full(data_array.shape[0], 0.0)
                valid_mask = std_values > self.eps
                if np.any(valid_mask):
                    valid_data = data_array[valid_mask]
                    kurt_values[valid_mask] = stats.kurtosis(valid_data, axis=1)
                stats_dict["kurt"] = kurt_values

        # Quantiles - all vectorized
        if "min" in self.selected_features:
            stats_dict["min"] = np.min(data_array, axis=1)

        if "1qrt" in self.selected_features:
            stats_dict["1qrt"] = np.percentile(data_array, 25, axis=1)

        if "med" in self.selected_features:
            stats_dict["med"] = np.median(data_array, axis=1)

        if "3qrt" in self.selected_features:
            stats_dict["3qrt"] = np.percentile(data_array, 75, axis=1)

        if "max" in self.selected_features:
            stats_dict["max"] = np.max(data_array, axis=1)

        return stats_dict

    def _extract_features_from_array(
        self, data_array: np.ndarray, var_idx: int
    ) -> dict:
        """
        Extract statistical features from a window array.

        Args:
            data_array: Array with variable data
            var_idx: Variable index

        Returns:
            Dictionary with extracted features
        """
        if data_array.size == 0:
            return {}

        # Calculate statistics
        stats_dict = self._calculate_statistics(data_array)

        # Format with variable name
        features_dict = {}
        for stat_name, values in stats_dict.items():
            col_name = f"var{var_idx}_{stat_name}"
            # Ensure values are 1D arrays
            if isinstance(values, np.ndarray):
                features_dict[col_name] = values
            else:
                features_dict[col_name] = np.array([values])

        return features_dict

    def pre_process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply initial preprocessing.

        Args:
            data: Input DataFrame

        Returns:
            Processed DataFrame
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")

        if data.empty:
            raise ValueError("Input data is empty")

        # Apply offset if specified
        data_size = len(data)
        if self.offset > 0:
            if self.offset >= data_size:
                raise ValueError(
                    f"Offset ({self.offset}) is larger than data length ({data_size})"
                )
            data = data.iloc[self.offset :].copy()

        return data

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Main step logic - statistical feature extraction.

        Args:
            data: DataFrame with windowed data

        Returns:
            DataFrame with extracted features
        """
        # Check if data is windowed
        if not self.is_windowed:
            raise ValueError(
                "Data is not windowed. Please use the Windowing class to window your data first, "
                "then set is_windowed=True in the config when initializing ExtractStatisticalFeatures."
            )

        # Identify variables
        variables = self._identify_variables(data)

        if not variables:
            raise ValueError("No variables found in the data")

        # Extract labels if they exist
        labels = None
        if self.label_column and self.label_column in data.columns:
            labels = data[self.label_column].values

        # Extract features for each variable
        all_features = {}

        for var_idx, var_columns in variables.items():
            if not var_columns:
                continue

            # Get variable data
            var_data = data[var_columns].values

            # Extract features
            var_features = self._extract_features_from_array(var_data, var_idx)

            # Add to result
            all_features.update(var_features)

        if not all_features:
            raise ValueError("No features were extracted")

        # Create result DataFrame
        result_df = pd.DataFrame(all_features)

        # Add labels if they existA
        if labels is not None:
            result_df["label"] = labels

        return result_df

    def post_process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Post-process the data.

        Args:
            data: DataFrame with extracted features

        Returns:
            Final DataFrame
        """
        if data.empty:
            raise ValueError("No data to post-process")

        # Check for NaN or infinite values
        if data.select_dtypes(include=[np.number]).isnull().any().any():
            print("Warning: NaN values detected in extracted features")

        if np.isinf(data.select_dtypes(include=[np.number]).values).any():
            print("Warning: Infinite values detected in extracted features")
            # Replace infinities with finite extreme values
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].replace(
                [np.inf, -np.inf], [np.finfo(np.float64).max, np.finfo(np.float64).min]
            )

        return data
