from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..core.base_visualizer import BaseVisualizer


class PlotFFT(BaseVisualizer):
    """
    Visualizer for computing and plotting the Fast Fourier Transform (FFT)
    of a time series.
    """

    def __init__(
        self,
        series: pd.Series,
        title: str = "FFT Analysis",
        sample_rate: float | None = None,
    ) -> None:
        """
        Initialize the FFT visualizer.

        Args:
            series: Input time series used to compute the FFT.
            title: Title of the FFT plot.
            sample_rate: Optional sampling rate of the series. If provided,
                frequencies are shown in Hertz (Hz); otherwise, frequencies
                are shown in cycles per sample.

        Returns:
            None.

        Raises:
            TypeError: If series is not a pandas Series.
        """
        self.series = series
        self.title = title
        self.sample_rate = sample_rate

    def plot(self, ax: Axes | None = None) -> tuple[Figure, Axes]:
        """
        Plot the FFT amplitude spectrum of the input series.

        Args:
            ax: Matplotlib Axes to draw the FFT plot on. If None, a new
                Figure and Axes are created.

        Returns:
            A tuple containing:
                - fig: The matplotlib Figure object.
                - ax: The matplotlib Axes containing the FFT amplitude spectrum.

        Raises:
            ValueError: If the input series is empty.
            ValueError: If the input series contains only NaN values.
        """
        if self.series.empty:
            raise ValueError("Input series is empty")

        clean_series = self.series.dropna()
        if clean_series.empty:
            raise ValueError("Series contains only NaN values")

        num_samples = len(clean_series)

        if self.sample_rate is None:
            sample_period = 1.0
            freq_unit = "Cycles per Sample"
        else:
            sample_period = 1.0 / self.sample_rate
            freq_unit = "Frequency (Hz)"

        yf = np.fft.fft(clean_series.values)
        xf = np.fft.fftfreq(num_samples, sample_period)[: num_samples // 2]
        amplitude = 2.0 / num_samples * np.abs(yf[0 : num_samples // 2])

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        else:
            fig = cast(Figure, ax.get_figure())

        ax.plot(xf, amplitude, linewidth=1.5)
        ax.grid(True, alpha=0.3)
        ax.set_title(self.title)
        ax.set_xlabel(freq_unit)
        ax.set_ylabel("Amplitude")
        ax.set_xlim(0.0, float(xf.max()))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()

        return fig, ax
