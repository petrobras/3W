import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Optional
from typing import cast
from statsmodels.tsa.seasonal import seasonal_decompose

from ..constants import PLOTS_DIR


class DataVisualization:
    @staticmethod
    def plot_series(
        series: pd.Series,
        title: str,
        xlabel: str,
        ylabel: str,
        overlay_events: bool = False,
        ax: Optional[Axes] = None,
        **plot_kwargs,
    ) -> Figure:
        """
        Static method to plot a time series.

        Args:
            series (pd.Series): Series with datetime index.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            overlay_events (bool): Whether to overlay vertical lines for events.
            ax (Optional[Axes]): Matplotlib Axes to plot into. Creates new if None.
            **plot_kwargs: Additional keyword arguments passed to `ax.plot`.

        Returns:
            matplotlib.figure.Figure: The resulting plot figure.
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = cast(Figure, ax.figure)

        ax.plot(series.index, series.values, label="Value", **plot_kwargs)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)

        if overlay_events:
            for date in series.index[series.isna()]:
                ax.axvline(x=date, color="red", linestyle="--", alpha=0.7)

        return fig

    @staticmethod
    def _save_plot(title: str) -> str:
        """Helper to save a plot to the 'plots' directory."""
        plot_dir = Path(PLOTS_DIR)

        # Create the directory if it doesn't exist.
        plot_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{title.replace(' ', ' ').lower()}.png"

        filepath = plot_dir / filename

        plt.savefig(filepath)
        plt.close()

        print(f"DataVisualization: Chart saved to '{filepath}'")
        return str(filepath)

    @staticmethod
    def plot_multiple_series(
        series_list: list[pd.Series],
        labels: list[str],
        title: str,
        xlabel: str,
        ylabel: str,
    ) -> str:
        for i, series in enumerate(series_list):
            if isinstance(series, np.ndarray):
                series_list[i] = pd.Series(series)
            elif not isinstance(series, pd.Series):
                raise ValueError("Input series must be pandas Series or numpy ndarray.")

        plt.figure(figsize=(10, 5))
        for series, label in zip(series_list, labels):
            plt.plot(series.index, series.values, label=label)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        return DataVisualization._save_plot(title)

    @staticmethod
    def plot_fft(series: pd.Series, title: str = "FFT Analysis") -> str:
        """Calculates and plots the Fast Fourier Transform of a series."""
        N = len(series)
        T = 1.0 / N  # Sample spacing
        yf = np.fft.fft(series.values)
        xf = np.fft.fftfreq(N, T)[: N // 2]

        plt.figure(figsize=(10, 5))
        plt.plot(xf, 2.0 / N * np.abs(yf[0 : N // 2]))
        plt.grid()
        plt.title(title)
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        return DataVisualization._save_plot(title)

    @staticmethod
    def seasonal_decompose(
        series: pd.Series, model: str = "additive", period: int = 12
    ) -> str:
        """Performs and plots seasonal decomposition."""
        result = seasonal_decompose(series, model=model, period=period)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        result.observed.plot(ax=ax1, legend=False)
        ax1.set_ylabel("Observed")
        result.trend.plot(ax=ax2, legend=False)
        ax2.set_ylabel("Trend")
        result.seasonal.plot(ax=ax3, legend=False)
        ax3.set_ylabel("Seasonal")
        result.resid.plot(ax=ax4, legend=False)
        ax4.set_ylabel("Residual")
        plt.suptitle("Seasonal Decomposition", y=0.94)
        plt.tight_layout()
        return DataVisualization._save_plot("Seasonal_Decomposition")

    @staticmethod
    def correlation_heatmap(
        df_of_series: pd.DataFrame, title: str = "Correlation Heatmap"
    ) -> str:
        """Plots a correlation heatmap for a DataFrame."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(df_of_series.corr(), annot=True, cmap="viridis", fmt=".2f")
        plt.title(title)
        plt.tight_layout()
        return DataVisualization._save_plot(title)

    @staticmethod
    def plot_wavelet_spectrogram(
        series: pd.Series, title: str = "Wavelet Spectrogram"
    ) -> str:
        """Mock plot for a wavelet spectrogram."""
        plt.figure(figsize=(10, 5))
        # In a real scenario, use libraries like pywt. For now, a mock image.
        mock_spectrogram = np.random.rand(50, len(series))
        plt.imshow(mock_spectrogram, aspect="auto", cmap="inferno")
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Frequency Scale")
        return DataVisualization._save_plot(title)
