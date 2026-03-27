import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from scipy.signal import get_window
from scipy.stats import mode

from typing import Literal
from pydantic import Field, field_validator
from ..core.base_feature_extractor import (
    BaseFeatureExtractor,
    BaseFeatureExtractorConfig,
)
from ..core.dataset_outputs import DatasetOutputs


class WindowingConfig(BaseFeatureExtractorConfig):
    window: str | tuple = Field(
            default="boxcar",
            description="Window type for signal data. str or tuple of (window_name, param1, param2, ...).")

    label_strategy: Literal["last", "mode"] = Field(
            default="last",
            description="Strategy for assigning labels to windows: 'last' uses the last label in the window,\
                    while 'mode' uses the most frequent label (mode) in the window.")

    window_size: int = Field(
            default=128,
            gt=1,
            description="Number of time steps in each window. Utilize power of 2 sizes for wavelet decomposition.")

    overlap: float = Field(
            default=0.0,
            ge=0.0, lt=1.0,
            description="Fraction of overlap between windows (0.0 to <1.0).")

    normalize: bool = Field(
            default=False,
            description="Whether to windows to unit scale.")

    symmetric: bool = Field(
            default=True,
            description="Whether to use symmetric windows (True) or periodic windows (False).")

    pad_last_window: bool = Field(
            default=False,
            description="Whether to pad the last window if it is smaller than `window_size`. If False, the last window will be dropped.")

    pad_value: float = Field(
            default=0.0,
            description="Value to use for padding the last window if `pad_last_window` is True.")

    target_: type = Field(default_factory=lambda: Windowing)

    @field_validator("window")
    def validate_window(cls, v: str | tuple):
        try:
            if isinstance(v, str):
                get_window(v, 128) # try to create a window with no additional parameters
            else:
                get_window(v[0], 128, *v[1:]) # try to create a window with provided parameters
        except Exception as e:
            raise ValueError(f"Invalid window parameters: {e}")

        return v

class Windowing(BaseFeatureExtractor):
    """
    A data processing step that applies windowing techniques to time series data.

    This class creates overlapping or non-overlapping windows from time series data,
    applying window functions for signal processing. It supports multiple variables
    and various window types from scipy.signal.

    The signal is always padded to the left (beginning of the time series) to ensure that the windows cover the start of
    the data. The last window can be optionally padded to ensure the entirety of the data is covered, or dropped if it
    is smaller than the specified window size.

    The signals are padded propagating the edge values.

    Attributes:
        config (WindowingConfig): Configuration object containing windowing parameters
    """

    def __init__(self, config: WindowingConfig,):
        """
        Initialize the Windowing step with the provided configuration.

        Args:
            config (WindowingConfig): Configuration object containing windowing parameters.
        """
        self.config = config

        # make tuple if window is a string for consistent processing
        if isinstance(self.config.window, str):
            window_name = (self.config.window,)
        else:
            window_name = self.config.window

        # create window function based on configuration
        self.window = get_window(
                window_name[0],
                self.config.window_size,
                *window_name[1:],
                fftbins=not self.config.symmetric)

        if self.config.normalize: # normalize window to unit L1
            self.window = self.window / np.sum(self.window)

    def transform(self, data: DatasetOutputs) -> DatasetOutputs:
        """
        Apply windowing to the input time series data.

        The output dataframe will have a multi-index with levels "window" and "variable", where "window" indicates the
        window number and "variable" indicates the original variable name. The columns will be the time steps within
        each window.

        Args:
            data: DatasetOutputs with signal and label data

        Returns:
            DatasetOutputs: Windowed signals with corresponding labels
        """

        # compute step size based on overlap
        step = int(self.config.window_size * (1 - self.config.overlap))
        step = max(step, 1)  # ensure step is at least 1 to avoid infinite loops

        col_names = data.signal.columns
        signal = data.signal.values

        n_samples, n_channels = signal.shape

        # prepare padding for the start/end of the data, if needed
        padding_start = self.config.window_size - 1

        if self.config.pad_last_window:
            # We pad to the right so that ((window_size - 1) + n_samples + padding_end - window_size) % step == 0
            padding_end = (step - ((n_samples - 1) % step)) % step
        else:
            padding_end = 0

        signal = np.pad(signal, [(padding_start, padding_end), (0, 0)], mode="edge")

        # sliding window view of the signal and labels
        signal = sliding_window_view(signal, (self.config.window_size, n_channels))[::step, 0] # (N_win, window_size,
                                                                                               #  n_channels)
        # multiply by window function and transpose window to last dimension
        signal = np.einsum("ijk,j->ikj", signal, self.window) # (N_win, n_channels, window_size)
        
        # lets assign a multi-index to the columns of the windowed signal for better interpretability
        index = pd.MultiIndex.from_product((range(signal.shape[0]), col_names), names=["window", "variable"])
        signal = pd.DataFrame(signal.reshape(-1, self.config.window_size), index=index) # (N_win * n_channels * window_size)

        if data.label is not None: # repeat for label series, if needed
            label = data.label.values
            label = np.pad(label, (padding_start, padding_end), mode="edge") # type: ignore
            label = sliding_window_view(label, self.config.window_size)[::step] # (N_win, window_size)

            # assign labels to windows based on strategy
            if self.config.label_strategy == "last":
                label = label[:, -1] # take the last label in each window
            elif self.config.label_strategy == "mode":
                # take the mode of the labels in each window
                label = mode(label, axis=1, nan_policy="omit", keepdims=False).mode
            label = pd.Series(label, name="label")
        else:
            label = None

        return DatasetOutputs(signal=signal, label=label, metadata=data.metadata.copy())
