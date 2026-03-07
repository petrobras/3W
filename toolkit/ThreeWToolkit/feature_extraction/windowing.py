import numpy as np
import pandas as pd
from scipy.signal import get_window
from pydantic import Field, field_validator
from ..core.base_feature_extractor import (
    BaseFeatureExtractor,
    BaseFeatureExtractorConfig,
)


class WindowingConfig(BaseFeatureExtractorConfig):
    window: str | tuple = "hann"
    window_size: int = Field(default=100, gt=1)
    overlap: float = Field(default=0.0, ge=0.0, lt=1.0)
    normalize: bool = False
    fftbins: bool = True
    pad_last_window: bool = True
    pad_value: float = 0.0
    target: type = Field(default_factory=lambda: Windowing)

    @field_validator("window")
    def validate_window(cls, v):
        WINDOWS_WITH_REQUIRED_PARAMS = {
            "kaiser": 1,
            "kaiser_bessel_derived": 1,
            "gaussian": 1,
            "general_cosine": 1,
            "general_gaussian": 2,
            "general_hamming": 1,
            "dpss": 1,
            "chebwin": 1,
        }

        WINDOWS_WITH_OPTIONAL_OR_NO_PARAMS = {
            "boxcar",
            "triang",
            "blackman",
            "hamming",
            "hann",
            "bartlett",
            "flattop",
            "parzen",
            "bohman",
            "blackmanharris",
            "nuttall",
            "barthann",
            "cosine",
            "lanczos",
            "exponential",
            "tukey",
            "taylor",
        }

        ALL_WINDOW_NAMES = (
            set(WINDOWS_WITH_REQUIRED_PARAMS) | WINDOWS_WITH_OPTIONAL_OR_NO_PARAMS
        )

        if isinstance(v, str):
            if v not in ALL_WINDOW_NAMES:
                raise ValueError(f"Invalid window name '{v}'.")
            if v in WINDOWS_WITH_REQUIRED_PARAMS:
                raise ValueError(
                    f"Window '{v}' requires parameter(s); use a tuple like ('{v}', param)."
                )

        else:
            if len(v) == 0 or not isinstance(v[0], str):
                raise ValueError("Tuple window must start with a string window name.")

            name = v[0]
            params = v[1:]

            if name not in ALL_WINDOW_NAMES:
                raise ValueError(f"Unknown window name '{name}'.")

            if name in WINDOWS_WITH_REQUIRED_PARAMS:
                expected = WINDOWS_WITH_REQUIRED_PARAMS[name]
                if len(params) < expected:
                    raise ValueError(
                        f"Window '{name}' requires {expected} parameter(s), got {len(params)}."
                    )

        return v


class Windowing(BaseFeatureExtractor):
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

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply windowing to the input time series data.

        Args:
            df (pd.DataFrame): DataFrame with time index, signal columns, and a label column (e.g., 'class').

        Returns:
            pd.DataFrame: DataFrame containing windowed signals and corresponding labels.
        """
        self._check_window_size_vs_data(df)

        # Assume the label column is the last column
        target_col = df.columns[-1]
        signal_cols = [col for col in df.columns if col != target_col]
        n_samples = len(df)
        step = int(self.config.window_size * (1 - self.config.overlap))
        if step < 1:
            step = 1

        # Precompute window function
        win = get_window(
            self.config.window, self.config.window_size, fftbins=self.config.fftbins
        )

        signal_data = df[signal_cols].values
        label_data = np.asarray(df[target_col].values)

        windows = []
        for start in range(0, n_samples, step):
            end = start + self.config.window_size
            if end > n_samples:
                if not self.config.pad_last_window:
                    break
                # Pad the window
                pad_size = end - n_samples
                window_signals = np.pad(
                    signal_data[start:n_samples],
                    pad_width=((0, pad_size), (0, 0)),
                    mode="constant",
                    constant_values=self.config.pad_value,
                )
                window_labels = np.pad(
                    label_data[start:n_samples],
                    pad_width=(0, pad_size),
                    mode="constant",
                    constant_values=label_data[n_samples - 1] if n_samples > 0 else 0,
                )
            else:
                window_signals = signal_data[start:end]
                window_labels = label_data[start:end]

            # Apply window function
            windowed = window_signals * win.reshape(-1, 1)
            windowed_flat = windowed.T.flatten()

            # Use mode of label in window
            mode_series = pd.Series(window_labels).mode()
            label = mode_series[0] if len(mode_series) > 0 else 0

            window_row = np.append(windowed_flat, label)
            windows.append(window_row)

        # Generate column names using original variable names + label class
        col_names = []
        for col in signal_cols:
            for t in range(self.config.window_size):
                col_names.append(f"{col}_t{t}")
        col_names.append("label")

        return pd.DataFrame(windows, columns=col_names)

    def _check_window_size_vs_data(self, values: pd.DataFrame):
        """
        This method ensures that the configured window size does not exceed
        the available data length, which would make windowing impossible.

        Args:
            values (pd.DataFrame): Input data DataFrame with "signal" and "label" columns
        """
        n_samples = len(values)
        if self.config.window_size > n_samples:
            raise ValueError(
                "`window_size` must be smaller than or equal to the length of X."
            )
