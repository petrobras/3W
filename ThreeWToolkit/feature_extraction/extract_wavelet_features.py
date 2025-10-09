import numpy as np
import pandas as pd
import pywt

from ..core.base_feature_extractor import WaveletConfig
from ..core.base_step import BaseStep


class ExtractWaveletFeatures(BaseStep):
    """
    Extracts wavelet features from windowed time series data using Stationary Wavelet Transform (SWT).

    Supports both univariate and multivariate analysis using PyWavelets library.

    IMPORTANT: Data must be already windowed. Each row should represent a window.
    If data is not windowed, use the Windowing class first.

    Input format: DataFrame with windowed data where each row is a window
    Output format:
    - Univariate: [var1_feature1, var1_feature2, ..., label]
    - Multivariate: [var1_feature1, var2_feature1, ..., var1_feature2, var2_feature2, ..., label]
    """

    def __init__(self, config: WaveletConfig):
        """
        Initialize the wavelet feature extractor.

        Args:
            config: Configuration object with wavelet parameters
        """
        super().__init__()

        self.level = config.level
        # Calculate window_size based on level (power of 2)
        self.window_size = 2**self.level
        self.overlap = config.overlap
        self.offset = config.offset
        self.wavelet = config.wavelet

        self.is_windowed = getattr(config, "is_windowed", False)
        self.label_column = getattr(config, "label_column", None)

        # Initialize wavelet filter matrix
        self._initialize_wavelet_filters()

    def _initialize_wavelet_filters(self):
        """
        Initialize the wavelet filter matrix using SWT decomposition.
        Creates filter matrix for efficient batch processing.
        """
        # Create impulse response for filter matrix generation
        impulse = np.zeros(self.window_size)
        impulse[-1] = 1

        # Perform SWT decomposition to get coefficients
        swt_coefficients = pywt.swt(impulse, self.wavelet, level=self.level)

        # Stack coefficients to create filter matrix
        filter_components = []
        for level_coeffs in swt_coefficients:
            for coeff in level_coeffs:  # approximation and detail coefficients
                filter_components.append(coeff)

        # Add original signal as the last component (A0 level)
        filter_components.append(impulse)

        # Create filter matrix for matrix multiplication
        self.wt_filter_matrix = np.stack(filter_components, axis=-1)

        # Generate feature names
        self.feat_names = []
        for level in range(self.level, 0, -1):
            self.feat_names.extend(
                [f"A{level}", f"D{level}"]
            )  # Approximation and Detail
        self.feat_names.append("A0")  # Original approximation level

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

    def _extract_wavelet_features(self, data_array: np.ndarray, var_idx: int) -> dict:
        """
        Extract wavelet features from windowed data array.

        Args:
            data_array: 2D array where each row is a window
            var_idx: Variable index for naming

        Returns:
            Dictionary with extracted wavelet features
        """
        if data_array.size == 0:
            return {}

        # If 1D, convert to 2D (1 window)
        if data_array.ndim == 1:
            data_array = data_array.reshape(1, -1)

        # Check if window size matches the expected size
        if data_array.shape[1] != self.window_size:
            # If window is larger, truncate to window_size
            if data_array.shape[1] > self.window_size:
                data_array = data_array[:, : self.window_size]
            else:
                # If window is smaller, pad with zeros
                padding_size = self.window_size - data_array.shape[1]
                data_array = np.pad(
                    data_array, ((0, 0), (0, padding_size)), mode="constant"
                )

        # Apply wavelet transform using matrix multiplication
        # Shape: (num_windows, window_size) @ (window_size, num_features) = (num_windows, num_features)
        wavelet_coeffs = np.dot(data_array, self.wt_filter_matrix)

        # Create feature dictionary
        features_dict = {}
        for j, feat_name in enumerate(self.feat_names):
            col_name = f"var{var_idx}_{feat_name}"
            features_dict[col_name] = wavelet_coeffs[:, j]

        return features_dict

    def _apply_swt_decomposition(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply Stationary Wavelet Transform decomposition to a single signal.

        Args:
            signal: 1D array representing a single window

        Returns:
            1D array with wavelet coefficients
        """
        if len(signal) != self.window_size:
            # Pad or truncate to match expected window size
            if len(signal) > self.window_size:
                signal = signal[: self.window_size]
            else:
                signal = np.pad(
                    signal, (0, self.window_size - len(signal)), mode="constant"
                )

        # Perform SWT decomposition
        swt_coeffs = pywt.swt(signal, self.wavelet, level=self.level)

        # Flatten coefficients
        coefficients = []
        for level_coeffs in swt_coeffs:
            for coeff in level_coeffs:  # approximation and detail
                coefficients.append(np.mean(coeff))  # Use mean as representative value

        # Add original signal mean as A0
        coefficients.append(np.mean(signal))

        return np.array(coefficients)

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
        Main step logic - wavelet feature extraction.

        Args:
            data: DataFrame with windowed data

        Returns:
            DataFrame with extracted wavelet features
        """
        # Check if data is windowed
        if not self.is_windowed:
            raise ValueError(
                "Data is not windowed. Please use the Windowing class to window your data first, "
                "then set is_windowed=True in the config when initializing ExtractWaveletFeatures."
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

            # Extract wavelet features
            var_features = self._extract_wavelet_features(var_data, var_idx)

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
            print("Warning: NaN values detected in extracted wavelet features")

        if np.isinf(data.select_dtypes(include=[np.number]).values).any():
            print("Warning: Infinite values detected in extracted wavelet features")
            # Replace infinities with finite extreme values
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].replace(
                [np.inf, -np.inf], [np.finfo(np.float64).max, np.finfo(np.float64).min]
            )

        return data
