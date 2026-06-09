from typing import cast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..core.base_visualizer import BaseVisualizer


class WaveletSpectrogramPlot(BaseVisualizer):
    """
    Visualizer for plotting a mock wavelet spectrogram from a time series.

    Generates a synthetic spectrogram for demonstration purposes only.
    """

    def __init__(
        self,
        series: pd.Series,
        title: str = "Wavelet Spectrogram",
    ) -> None:
        """
        Initialize the wavelet spectrogram visualizer.

        Args:
            series: Input pandas Series used to generate the spectrogram.
            title: Title of the spectrogram plot.

        Returns:
            None.

        Raises:
            TypeError: If series is not a pandas Series.
        """
        self.series = series
        self.title = title

    def plot(self, ax: Axes | None = None) -> tuple[Figure, Axes]:
        """
        Plot a wavelet spectrogram for the input series.

        Args:
            ax: Matplotlib Axes to draw the spectrogram on. If None,
                a new Figure and Axes are created.

        Returns:
            A tuple containing:
                - fig: The matplotlib Figure object.
                - ax: The matplotlib Axes with the rendered spectrogram.

        Raises:
            ValueError: If the input series is empty.
        """
        if self.series.empty:
            raise ValueError("Input series is empty")

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        else:
            fig = cast(Figure, ax.get_figure())

        time_points = len(self.series)
        frequency_scales = 50
        mock_spectrogram = np.random.rand(frequency_scales, time_points)

        for i in range(frequency_scales):
            mock_spectrogram[i, :] *= np.exp(-i / frequency_scales * 2)

        extent_tuple = (0.0, float(time_points), 1.0, float(frequency_scales))
        im = ax.imshow(
            mock_spectrogram,
            aspect="auto",
            cmap="inferno",
            origin="lower",
            extent=extent_tuple,
        )

        ax.set_title(self.title, fontsize=14)
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Frequency Scale")

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Magnitude", rotation=270, labelpad=15)

        ax.text(
            0.02,
            0.98,
            "Note: Mock implementation",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            verticalalignment="top",
        )

        fig.tight_layout()

        return fig, ax
