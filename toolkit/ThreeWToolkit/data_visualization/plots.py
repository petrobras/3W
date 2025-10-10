from abc import ABC
from pathlib import Path
from typing import Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from statsmodels.tsa.seasonal import seasonal_decompose

from ThreeWToolkit.constants import PLOTS_DIR


class DataVisualization(ABC):
    """
    Abstract base class providing static methods for data visualization.

    This class offers a comprehensive suite of visualization tools specifically
    designed for time series analysis, correlation analysis, and frequency domain
    analysis. All methods are static and return matplotlib Figure objects along
    with saved plot paths.
    """

    @staticmethod
    def plot_series(
        series: pd.Series,
        title: str,
        xlabel: str,
        ylabel: str,
        overlay_events: bool = False,
        ax: Axes | None = None,
        **plot_kwargs,
    ) -> tuple[Figure, str]:
        """
        Plot a single time series with optional event overlay.

        Creates a line plot of a pandas Series with datetime index. Optionally
        overlays vertical lines at points where the series has missing values
        to indicate potential events or anomalies.

        Args:
            series (pd.Series): Time series data with datetime index to plot.
            title (str): Title for the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            overlay_events (bool, optional): If True, adds vertical dashed lines
                at positions where series has NaN values. Defaults to False.
            ax (Axes | None, optional): Matplotlib Axes object to plot on.
                If None, creates new figure and axes. Defaults to None.
            **plot_kwargs: Additional keyword arguments passed to ax.plot().
                Common options include color, linewidth, linestyle, etc.

        Returns:
            tuple[Figure, str]: A tuple containing:
                - matplotlib Figure object
                - str: Path to the saved plot file

        Example:
            >>> series = pd.Series([1, 2, 3], index=pd.date_range('2023-01-01', periods=3))
            >>> fig, path = DataVisualization.plot_series(
            ...     series, "My Series", "Date", "Value", color='blue'
            ... )
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = cast(Figure, ax.figure)

        ax.plot(series.index, series.values, label="Value", **plot_kwargs)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        if overlay_events:
            nan_dates = series.index[series.isna()]
            for date in nan_dates:
                ax.axvline(x=date, color="red", linestyle="--", alpha=0.7, linewidth=1)

            if len(nan_dates) > 0:
                ax.axvline(
                    x=nan_dates[0],
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    linewidth=1,
                    label="Missing Data",
                )
                ax.legend()

        img_path = DataVisualization._save_plot(title)

        return fig, img_path

    @staticmethod
    def plot_multiple_series(
        series_list: list[pd.Series],
        labels: list[str],
        title: str,
        xlabel: str,
        ylabel: str,
        ax: Axes | None = None,
        **plot_kwargs,
    ) -> tuple[Figure, str]:
        """
        Plot multiple time series on the same axes for comparison.

        Creates a multi-line plot allowing visual comparison of multiple time series.
        Each series is plotted with a different color and labeled according to the
        provided labels list.

        Args:
            series_list (list[pd.Series]): list of pandas Series with datetime indices.
                All series should have compatible time ranges for meaningful comparison.
            labels (list[str]): list of labels for each series. Must have the same
                length as series_list.
            title (str): Title for the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            ax (Axes | None, optional): Matplotlib Axes object to plot on.
                If None, creates new figure and axes. Defaults to None.
            **plot_kwargs: Additional keyword arguments passed to ax.plot().

        Returns:
            tuple[Figure, str]: A tuple containing:
                - matplotlib Figure object
                - str: Path to the saved plot file

        Raises:
            ValueError: If the length of series_list and labels don't match.

        Example:
            >>> series1 = pd.Series([1, 2, 3], index=pd.date_range('2023-01-01', periods=3))
            >>> series2 = pd.Series([2, 4, 6], index=pd.date_range('2023-01-01', periods=3))
            >>> fig, path = DataVisualization.plot_multiple_series(
            ...     [series1, series2], ["Series A", "Series B"],
            ...     "Comparison", "Date", "Value"
            ... )
        """
        if len(series_list) != len(labels):
            raise ValueError("series_list and labels must have the same length")

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = cast(Figure, ax.figure)

        cmap = plt.get_cmap("Set1", len(series_list))
        colors = [cmap(i) for i in range(len(series_list))]

        for i, (series, label) in enumerate(zip(series_list, labels)):
            ax.plot(
                series.index, series.values, label=label, color=colors[i], **plot_kwargs
            )

        ax.legend(loc="best")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        img_path = DataVisualization._save_plot(title)
        return fig, img_path

    @staticmethod
    def correlation_heatmap(
        df_of_series: pd.DataFrame, ax: Optional[Axes] = None, **kwargs
    ) -> tuple[Figure, str]:
        """
        Generate a correlation heatmap for multiple time series.

        Creates a heatmap visualization of the correlation matrix between different
        time series columns in a DataFrame. Useful for identifying relationships
        between multiple variables.

        Args:
            df_of_series (pd.DataFrame): DataFrame where each column represents
                a time series. The correlation will be computed between columns.
            ax (Optional[Axes], optional): Matplotlib Axes object to draw on.
                If None, creates new figure and axes. Defaults to None.
            **kwargs: Additional keyword arguments:
                - title (str): Title for the heatmap. Default: "Correlation Heatmap"
                - figsize (tuple): Figure size. Default: (10, 8)
                - Other arguments passed to sns.heatmap() such as:
                  - annot (bool): Show correlation values in cells
                  - cmap (str): Colormap name
                  - center (float): Center value for colormap

        Returns:
            tuple[Figure, str]: A tuple containing:
                - matplotlib Figure object
                - str: Path to the saved plot file

        Example:
            >>> df = pd.DataFrame({
            ...     'A': [1, 2, 3, 4],
            ...     'B': [2, 4, 6, 8],
            ...     'C': [1, 3, 2, 4]
            ... })
            >>> fig, path = DataVisualization.correlation_heatmap(
            ...     df, annot=True, cmap='coolwarm'
            ... )
        """
        title: str = kwargs.pop("title", "Correlation Heatmap")
        figsize: tuple = kwargs.pop("figsize", (10, 8))

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = cast(Figure, ax.figure)

        if df_of_series.empty:
            ax.text(
                0.5,
                0.5,
                "Empty DataFrame\nNo data to display",
                fontsize=14,
                ha="center",
                va="center",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="lightgray"),
            )
            ax.set_title(title)
            fig.tight_layout()
            return fig, DataVisualization._save_plot(title)

        # Set default heatmap parameters if not provided
        heatmap_defaults = {
            "annot": True,
            "fmt": ".2f",
            "cmap": "coolwarm",
            "center": 0,
            "square": True,
            "cbar_kws": {"shrink": 0.8},
        }

        # Update defaults with user-provided kwargs
        for key, value in heatmap_defaults.items():
            kwargs.setdefault(key, value)

        corr_matrix = df_of_series.corr()
        sns.heatmap(corr_matrix, ax=ax, **kwargs)
        ax.set_title(title)
        fig.tight_layout()

        img_path = DataVisualization._save_plot(title)
        return fig, img_path

    @staticmethod
    def plot_fft(
        series: pd.Series,
        title: str = "FFT Analysis",
        sample_rate: float | None = None,
    ) -> tuple[Figure, str]:
        """
        Compute and plot the Fast Fourier Transform (FFT) of a time series.

        Performs frequency domain analysis by computing the FFT of the input series
        and plotting the amplitude spectrum. Useful for identifying periodic patterns
        and dominant frequencies in time series data.

        Args:
            series (pd.Series): Input time series data. Should contain numeric values.
            title (str, optional): Title for the plot. Defaults to "FFT Analysis".
            sample_rate (float | None, optional): Sampling rate of the data in Hz.
                If None, assumes unit spacing. Defaults to None.

        Returns:
            tuple[Figure, str]: A tuple containing:
                - matplotlib Figure object
                - str: Path to the saved plot file

        Note:
            The plot shows the positive frequency components only (single-sided spectrum).
            The frequency axis will be in units of 1/time_unit if sample_rate is None,
            or in Hz if sample_rate is provided.

        Example:
            >>> # Create a signal with 50Hz and 120Hz components
            >>> t = np.linspace(0, 1, 500)
            >>> signal = np.sin(50 * 2 * np.pi * t) + 0.5 * np.sin(120 * 2 * np.pi * t)
            >>> series = pd.Series(signal)
            >>> fig, path = DataVisualization.plot_fft(series, sample_rate=500)
        """
        if series.empty:
            raise ValueError("Input series is empty")

        # Remove NaN values
        clean_series = series.dropna()
        if clean_series.empty:
            raise ValueError("Series contains only NaN values")

        N = len(clean_series)

        if sample_rate is None:
            # Assume unit spacing
            T = 1.0
            freq_unit = "Cycles per Sample"
        else:
            T = 1.0 / sample_rate
            freq_unit = "Frequency (Hz)"

        # Compute FFT
        yf = np.fft.fft(clean_series.values)
        xf = np.fft.fftfreq(N, T)[: N // 2]

        # Compute amplitude spectrum
        amplitude = 2.0 / N * np.abs(yf[0 : N // 2])

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(xf, amplitude, linewidth=1.5)
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel(freq_unit)
        ax.set_ylabel("Amplitude")
        ax.set_xlim(0, max(xf))

        # Add some styling
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        img_path = DataVisualization._save_plot(title)
        return fig, img_path

    @staticmethod
    def seasonal_decompose(
        series: pd.Series, model: str = "additive", period: int | None = None
    ) -> tuple[Figure, str]:
        """
        Perform and plot seasonal decomposition of a time series.

        Decomposes a time series into its trend, seasonal, and residual components
        using statistical decomposition methods. This is useful for understanding
        the underlying patterns in time series data.

        Args:
            series (pd.Series): Time series data with datetime index. Should have
                sufficient data points for meaningful decomposition (at least 2 full periods).
            model (str, optional): Type of decomposition model. Options:
                - "additive": Y(t) = Trend(t) + Seasonal(t) + Residual(t)
                - "multiplicative": Y(t) = Trend(t) * Seasonal(t) * Residual(t)
                Defaults to "additive".
            period (Optional[int], optional): Length of the seasonal period.
                If None, attempts to automatically detect the period. For monthly
                data, use 12; for daily data with weekly patterns, use 7.

        Returns:
            tuple[Figure, str]: A tuple containing:
                - matplotlib Figure object with 4 subplots (observed, trend, seasonal, residual)
                - str: Path to the saved plot file

        Raises:
            ValueError: If the series is too short for decomposition or if an
                invalid model type is specified.

        Example:
            >>> # Monthly data with seasonal pattern
            >>> dates = pd.date_range('2020-01-01', periods=36, freq='M')
            >>> values = np.sin(np.arange(36) * 2 * np.pi / 12) + np.arange(36) * 0.1
            >>> series = pd.Series(values, index=dates)
            >>> fig, path = DataVisualization.seasonal_decompose(series, period=12)
        """
        if series.empty:
            raise ValueError("Input series is empty")

        if model not in ["additive", "multiplicative"]:
            raise ValueError("Model must be either 'additive' or 'multiplicative'")

        # Remove NaN values
        clean_series = series.dropna()
        if len(clean_series) < 10:
            raise ValueError(
                "Series too short for decomposition (minimum 10 points required)"
            )

        try:
            result = seasonal_decompose(clean_series, model=model, period=period)
        except Exception as e:
            raise ValueError(f"Decomposition failed: {str(e)}")

        # Create the decomposition plot
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

        # Plot each component
        result.observed.plot(ax=axes[0], color="blue", linewidth=1.5)
        axes[0].set_ylabel("Observed")
        axes[0].set_title("Original Time Series")
        axes[0].grid(True, alpha=0.3)

        result.trend.plot(ax=axes[1], color="red", linewidth=1.5)
        axes[1].set_ylabel("Trend")
        axes[1].set_title("Trend Component")
        axes[1].grid(True, alpha=0.3)

        result.seasonal.plot(ax=axes[2], color="green", linewidth=1.5)
        axes[2].set_ylabel("Seasonal")
        axes[2].set_title("Seasonal Component")
        axes[2].grid(True, alpha=0.3)

        result.resid.plot(ax=axes[3], color="orange", linewidth=1.5)
        axes[3].set_ylabel("Residual")
        axes[3].set_title("Residual Component")
        axes[3].set_xlabel("Date")
        axes[3].grid(True, alpha=0.3)

        # Add overall title
        fig.suptitle(
            f"Seasonal Decomposition ({model.title()} Model)", fontsize=16, y=0.98
        )

        # Improve layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)

        img_path = DataVisualization._save_plot("Seasonal_Decomposition")
        return fig, img_path

    @staticmethod
    def plot_wavelet_spectrogram(
        series: pd.Series, title: str = "Wavelet Spectrogram"
    ) -> tuple[Figure, str]:
        """
        Create a mock wavelet spectrogram visualization.

        This is a placeholder implementation that generates a mock spectrogram.
        In a production environment, this should be replaced with actual wavelet
        transform implementation using libraries like PyWavelets (pywt).

        Args:
            series (pd.Series): Input time series data.
            title (str, optional): Title for the spectrogram.
                Defaults to "Wavelet Spectrogram".

        Returns:
            tuple[Figure, str]: A tuple containing:
                - matplotlib Figure object
                - str: Path to the saved plot file

        Note:
            This is a mock implementation. For real wavelet analysis, consider
            using libraries like:
            - PyWavelets (pywt): Comprehensive wavelet transforms
            - SciPy: Basic wavelet functionality
            - ssqueezepy: Synchrosqueezed wavelet transforms

        Example:
            >>> series = pd.Series(np.random.randn(100))
            >>> fig, path = DataVisualization.plot_wavelet_spectrogram(series)
        """
        if series.empty:
            raise ValueError("Input series is empty")

        fig, ax = plt.subplots(figsize=(12, 8))

        # Mock spectrogram data
        # In real implementation, this would be replaced with actual wavelet transform
        time_points = len(series)
        frequency_scales = 50
        mock_spectrogram = np.random.rand(frequency_scales, time_points)

        # Add some structure to make it look more realistic
        for i in range(frequency_scales):
            mock_spectrogram[i, :] *= np.exp(-i / frequency_scales * 2)

        # Create the spectrogram plot
        extent_tuple = (0.0, float(time_points), 1.0, float(frequency_scales))
        im = ax.imshow(
            mock_spectrogram,
            aspect="auto",
            cmap="inferno",
            origin="lower",
            extent=extent_tuple,
        )

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Frequency Scale")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Magnitude", rotation=270, labelpad=15)

        # Add note about mock implementation
        ax.text(
            0.02,
            0.98,
            "Note: Mock implementation",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            verticalalignment="top",
        )

        plt.tight_layout()

        img_path = DataVisualization._save_plot(title)
        return fig, img_path

    @staticmethod
    def _save_plot(title: str) -> str:
        """
        Save the current matplotlib plot to the plots directory.

        This internal helper method handles the creation of the plots directory
        and saves the current figure with a standardized filename based on the title.

        Args:
            title (str): Title of the plot, used to generate filename.

        Returns:
            str: Full path to the saved plot file.

        Note:
            - Creates the plots directory if it doesn't exist
            - Automatically closes the plot after saving to free memory
            - Filename is generated by replacing spaces with underscores and converting to lowercase
            - Prints confirmation message with the save path
        """
        plot_dir = Path(PLOTS_DIR)
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Clean the title for use as filename
        safe_title = title.replace(" ", "_").replace("/", "_").replace("\\", "_")
        safe_title = "".join(
            c for c in safe_title if c.isalnum() or c in ["_", "-"]
        ).lower()
        filename = f"{safe_title}.png"

        filepath = plot_dir / filename

        # Save with high DPI for better quality
        plt.savefig(
            filepath, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )

        print(f"DataVisualization: Chart saved to '{filepath}'")
        return str(filepath)

    @staticmethod
    def create_subplot_grid(
        nrows: int, ncols: int, figsize: tuple[int, int] | None = None
    ) -> tuple[Figure, np.ndarray]:
        """
        Create a grid of subplots for complex visualizations.

        Utility method to create a standardized subplot grid with consistent
        styling and layout. Useful for creating dashboard-style visualizations
        with multiple related plots.

        Args:
            nrows (int): Number of rows in the subplot grid.
            ncols (int): Number of columns in the subplot grid.
            figsize (tuple[int, int] | None, optional): Figure size (width, height).
                If None, automatically calculated based on grid size.

        Returns:
            tuple[Figure, np.ndarray]: A tuple containing:
                - matplotlib Figure object
                - numpy array of Axes objects

        Example:
            >>> fig, axes = DataVisualization.create_subplot_grid(2, 2)
            >>> # Use individual axes: axes[0, 0], axes[0, 1], etc.
        """
        if figsize is None:
            figsize = (5 * ncols, 4 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        # Ensure axes is always a 2D array for consistent indexing
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1 or ncols == 1:
            axes = axes.reshape(nrows, ncols)

        fig.tight_layout(pad=3.0)

        return fig, axes
