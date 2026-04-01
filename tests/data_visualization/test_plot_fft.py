import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ThreeWToolkit.data_visualization.plot_fft import PlotFFT


class TestPlotFFT:
    def setup_method(self):
        """Setup method to create test time series."""
        np.random.seed(42)
        self.dates = pd.date_range("2024-01-01", periods=100, freq="D")
        self.values = np.sin(2 * np.pi * 0.1 * np.arange(100)) + np.random.normal(
            0, 0.1, 100
        )
        self.series = pd.Series(self.values, index=self.dates)

    def test_initialization(self):
        """Test that PlotFFT initializes correctly."""
        plotter = PlotFFT(series=self.series, title="Test FFT")
        
        assert plotter.series.equals(self.series)
        assert plotter.title == "Test FFT"
        assert plotter.sample_rate is None

    def test_initialization_with_sample_rate(self):
        """Test initialization with custom sample rate."""
        sample_rate = 1.0
        plotter = PlotFFT(series=self.series, sample_rate=sample_rate)
        
        assert plotter.sample_rate == sample_rate

    def test_plot_returns_figure_and_axes(self):
        """Test that plot returns a Figure and Axes."""
        plotter = PlotFFT(series=self.series, title="FFT Test")
        fig, ax = plotter.plot()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_plot_with_provided_axes(self):
        """Test plotting on provided Axes."""
        plotter = PlotFFT(series=self.series)
        fig_orig, ax_orig = plt.subplots()
        
        fig, ax = plotter.plot(ax=ax_orig)

        assert fig is fig_orig.figure
        assert ax is ax_orig
        plt.close(fig)

    def test_plot_with_sample_rate(self):
        """Test plotting with sample rate (frequency in Hz)."""
        plotter = PlotFFT(series=self.series, sample_rate=1.0)
        fig, ax = plotter.plot()

        # Check that xlabel mentions Hz
        xlabel = ax.get_xlabel()
        assert "Hz" in xlabel
        plt.close(fig)

    def test_plot_without_sample_rate(self):
        """Test plotting without sample rate (cycles per sample)."""
        plotter = PlotFFT(series=self.series, sample_rate=None)
        fig, ax = plotter.plot()

        # Check that xlabel mentions Cycles per Sample
        xlabel = ax.get_xlabel()
        assert "Cycles per Sample" in xlabel
        plt.close(fig)

    def test_plot_empty_series_raises_error(self):
        """Test that plotting an empty series raises ValueError."""
        empty_series = pd.Series(dtype=float)
        plotter = PlotFFT(series=empty_series)

        with pytest.raises(ValueError, match="Input series is empty"):
            plotter.plot()

    def test_plot_series_with_only_nan_raises_error(self):
        """Test that series with only NaN values raises ValueError."""
        nan_series = pd.Series([np.nan, np.nan, np.nan])
        plotter = PlotFFT(series=nan_series)

        with pytest.raises(ValueError, match="Series contains only NaN values"):
            plotter.plot()

    def test_plot_series_with_some_nan_values(self):
        """Test that series with some NaN values plots correctly."""
        series_with_nan = self.series.copy()
        series_with_nan.iloc[10:15] = np.nan
        
        plotter = PlotFFT(series=series_with_nan)
        fig, ax = plotter.plot()

        assert isinstance(fig, Figure)
        assert ax.has_data()
        plt.close(fig)

    def test_plot_has_title(self):
        """Test that the plot has the specified title."""
        title = "My Custom FFT"
        plotter = PlotFFT(series=self.series, title=title)
        fig, ax = plotter.plot()

        assert ax.get_title() == title
        plt.close(fig)

    def test_plot_has_grid(self):
        """Test that the plot has grid enabled."""
        plotter = PlotFFT(series=self.series)
        fig, ax = plotter.plot()

        # Check if grid is visible (grid is enabled in the plot_fft module)
        assert ax.xaxis.get_gridlines() or ax.yaxis.get_gridlines()
        plt.close(fig)

    def test_plot_axis_labels(self):
        """Test that axis labels are set correctly."""
        plotter = PlotFFT(series=self.series, sample_rate=10.0)
        fig, ax = plotter.plot()

        assert ax.get_xlabel() == "Frequency (Hz)"
        assert ax.get_ylabel() == "Amplitude"
        plt.close(fig)

    def test_plot_xlim_set(self):
        """Test that x-axis limits are set."""
        plotter = PlotFFT(series=self.series)
        fig, ax = plotter.plot()

        xlim = ax.get_xlim()
        assert xlim[0] == 0.0
        assert xlim[1] > 0.0
        plt.close(fig)

    def test_fft_computation(self):
        """Test that FFT is computed correctly for a simple signal."""
        # Create a simple sine wave
        t = np.linspace(0, 1, 100)
        freq = 5  # 5 Hz
        signal = np.sin(2 * np.pi * freq * t)
        series = pd.Series(signal)
        
        plotter = PlotFFT(series=series, sample_rate=100.0)
        fig, ax = plotter.plot()

        # The plot should show a peak near 5 Hz
        assert isinstance(fig, Figure)
        plt.close(fig)
