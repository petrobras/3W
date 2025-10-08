import numpy as np
import pandas as pd

from ..core.base_feature_extractor import EWStatisticalConfig
from ..core.base_step import BaseStep


class ExtractEWStatisticalFeatures(BaseStep):
    """
    Extracts exponentially weighted statistical features from windowed time series data.

    Applies exponential decay weights to calculate weighted statistics, giving more
    importance to recent observations within each window.

    IMPORTANT: Data must be already windowed. Each row should represent a window.
    If data is not windowed, use the Windowing class first.

    Input format: DataFrame with windowed data where each row is a window
    Output format:
    - Univariate: [var1_feature1, var1_feature2, ..., label]
    - Multivariate: [var1_feature1, var2_feature1, ..., var1_feature2, var2_feature2, ..., label]
    """

    FEATURES = [
        "ew_mean",
        "ew_std",
        "ew_skew",
        "ew_kurt",
        "ew_min",
        "ew_1qrt",
        "ew_med",
        "ew_3qrt",
        "ew_max",
    ]

    def __init__(self, config: EWStatisticalConfig):
        """
        Initialize the exponentially weighted statistical feature extractor.

        Args:
            config: Configuration object with exponential weighting parameters
        """
        super().__init__()

        self.window_size = config.window_size
        self.overlap = config.overlap
        self.offset = config.offset
        self.eps = config.eps
        self.decay = config.decay

        self.selected_features = (
            config.selected_features if config.selected_features else self.FEATURES
        )
        self.is_windowed = getattr(config, "is_windowed", False)
        self.label_column = getattr(config, "label_column", None)

    def _initialize_weights(self):
        """
        Initialize exponential decay weights for the window.
        More recent values get higher weights.
        """
        # Create exponential decay weights (recent values have higher weights)
        h = self.decay ** np.arange(self.window_size, 0, step=-1, dtype=np.float64)
        # Normalize weights so they sum to 1
        self.weights = h / (np.abs(h).sum() + self.eps)

    def _apply_weights(self, data_array: np.ndarray, axis=-1) -> np.ndarray:
        """
        Apply exponential weights to data using dot product.

        Args:
            data_array: Input array to weight
            axis: Axis along which to apply weights

        Returns:
            Weighted array
        """
        if axis == -1 and data_array.ndim == 2:
            # For 2D array: each row is a window, apply weights across columns
            return np.dot(data_array, self.weights)
        if axis == -1:
            return np.sum(data_array * self.weights, axis=axis)
        else:
            # Handle other axes by moving the target axis to the end
            data_moved = np.moveaxis(data_array, axis, -1)
            result = np.sum(data_moved * self.weights, axis=-1)
            return result

    def _identify_variables(self, data: pd.DataFrame) -> dict:
        """
        Identifies variables in the data based on naming pattern.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary mapping variable numbers to their column names
        """
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

    def _calculate_ew_statistics(self, data_array: np.ndarray) -> dict:
        """
        Calculate exponentially weighted statistics.

        Args:
            data_array: 2D array where each row is a window

        Returns:
            Dictionary with calculated exponentially weighted statistics
        """
        if data_array.size == 0:
            return {}

        # If 1D, convert to 2D (1 window)
        if data_array.ndim == 1:
            data_array = data_array.reshape(1, -1)

        stats_dict = {}

        # Exponentially weighted mean
        if "ew_mean" in self.selected_features:
            ew_mean = self._apply_weights(data_array, axis=-1)
            stats_dict["ew_mean"] = ew_mean
        else:
            # Calculate mean anyway as it's needed for other statistics
            ew_mean = self._apply_weights(data_array, axis=-1)

        # Exponentially weighted standard deviation
        if "ew_std" in self.selected_features:
            # Calculate weighted variance
            mean_expanded = np.expand_dims(ew_mean, axis=-1)
            variance = self._apply_weights(
                np.power(data_array - mean_expanded, 2), axis=-1
            )
            ew_std = np.sqrt(variance)
            stats_dict["ew_std"] = ew_std
        else:
            # Calculate std anyway as it's needed for standardization
            mean_expanded = np.expand_dims(ew_mean, axis=-1)
            variance = self._apply_weights(
                np.power(data_array - mean_expanded, 2), axis=-1
            )
            ew_std = np.sqrt(variance)

        # Standardized data for skewness and kurtosis
        mean_expanded = np.expand_dims(ew_mean, axis=-1)
        std_expanded = np.expand_dims(ew_std, axis=-1)
        standardized_data = (data_array - mean_expanded) / (std_expanded + self.eps)

        # Exponentially weighted skewness
        if "ew_skew" in self.selected_features:
            ew_skew = self._apply_weights(np.power(standardized_data, 3), axis=-1)
            stats_dict["ew_skew"] = ew_skew

        # Exponentially weighted kurtosis
        if "ew_kurt" in self.selected_features:
            ew_kurt = self._apply_weights(np.power(standardized_data, 4), axis=-1)
            stats_dict["ew_kurt"] = ew_kurt

        # Exponentially weighted quantiles
        # For quantiles, we use the standardized data and calculate weighted quantiles
        quantile_features = ["ew_min", "ew_1qrt", "ew_med", "ew_3qrt", "ew_max"]
        quantile_values = [0.0, 0.25, 0.5, 0.75, 1.0]

        for feat_name, q_val in zip(quantile_features, quantile_values):
            if feat_name in self.selected_features:
                # Calculate weighted quantiles for each window
                quantile_results = []
                for i in range(standardized_data.shape[0]):
                    window_data = standardized_data[i, :]
                    # Use numpy percentile as approximation
                    quantile_result = np.percentile(window_data, q_val * 100)
                    quantile_results.append(quantile_result)
                stats_dict[feat_name] = np.array(quantile_results)

        return stats_dict

    def _extract_features_from_array(
        self, data_array: np.ndarray, var_idx: int
    ) -> dict:
        """
        Extract exponentially weighted statistical features from a window array.

        Args:
            data_array: Array with variable data
            var_idx: Variable index

        Returns:
            Dictionary with extracted features
        """
        if data_array.size == 0:
            return {}

        # Calculate exponentially weighted statistics
        stats_dict = self._calculate_ew_statistics(data_array)

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

        # Initialize exponential weights
        self._initialize_weights()

        # Apply offset if specified
        if self.offset > 0:
            if self.offset >= len(data):
                raise ValueError(
                    f"Offset ({self.offset}) is larger than data length ({len(data)})"
                )
            data = data.iloc[self.offset :].copy()

        return data

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Main step logic - exponentially weighted statistical feature extraction.

        Args:
            data: DataFrame with windowed data

        Returns:
            DataFrame with extracted exponentially weighted features
        """
        # Check if data is windowed
        if not self.is_windowed:
            raise ValueError(
                "Data is not windowed. Please use the Windowing class to window your data first, "
                "then set is_windowed=True in the config when initializing ExtractEWStatisticalFeatures."
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

            # Extract exponentially weighted features
            var_features = self._extract_features_from_array(var_data, var_idx)

            # Add to result
            all_features.update(var_features)

        if not all_features:
            raise ValueError("No features were extracted")

        # Create result DataFrame
        result_df = pd.DataFrame(all_features)

        # Add labels if they exist
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
            print(
                "Warning: NaN values detected in extracted exponentially weighted features"
            )

        if np.isinf(data.select_dtypes(include=[np.number]).values).any():
            print(
                "Warning: Infinite values detected in extracted exponentially weighted features"
            )
            # Replace infinities with finite extreme values
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].replace(
                [np.inf, -np.inf], [np.finfo(np.float64).max, np.finfo(np.float64).min]
            )

        return data
