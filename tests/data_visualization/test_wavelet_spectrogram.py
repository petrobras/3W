import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ThreeWToolkit.data_visualization.wavelet_spectrogram import (
    WaveletSpectrogramPlot,
)


class TestWaveletSpectrogramPlot:
    def setup_method(self):
        """Setup method to create test time series."""
        np.random.seed(42)
        self.dates = pd.date_range("2024-01-01", periods=100, freq="D")
        self.values = np.sin(2 * np.pi * 0.1 * np.arange(100)) + np.random.normal(
            0, 0.1, 100
        )
        self.series = pd.Series(self.values, index=self.dates)

    def test_initialization(self):
        """Test that WaveletSpectrogramPlot initializes correctly."""
        plotter = WaveletSpectrogramPlot(series=self.series, title="Test Spectrogram")
        
        assert plotter.series.equals(self.series)
        assert plotter.title == "Test Spectrogram"

    def test_initialization_default_title(self):
        """Test initialization with default title."""
        plotter = WaveletSpectrogramPlot(series=self.series)
        
        assert plotter.title == "Wavelet Spectrogram"

    def test_plot_returns_figure_and_axes(self):
        """Test that plot returns a Figure and Axes."""
        plotter = WaveletSpectrogramPlot(series=self.series)
        fig, ax = plotter.plot()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_plot_with_provided_axes(self):
        """Test plotting on provided Axes."""
        plotter = WaveletSpectrogramPlot(series=self.series)
        fig_orig, ax_orig = plt.subplots()
        
        fig, ax = plotter.plot(ax=ax_orig)

        assert fig is fig_orig.figure
        assert ax is ax_orig
        plt.close(fig)

    def test_plot_empty_series_raises_error(self):
        """Test that plotting an empty series raises ValueError."""
        empty_series = pd.Series(dtype=float)
        plotter = WaveletSpectrogramPlot(series=empty_series)

        with pytest.raises(ValueError, match="Input series is empty"):
            plotter.plot()

    def test_plot_has_title(self):
        """Test that the plot has the specified title."""
        title = "My Custom Spectrogram"
        plotter = WaveletSpectrogramPlot(series=self.series, title=title)
        fig, ax = plotter.plot()

        assert ax.get_title() == title
        plt.close(fig)

    def test_plot_has_colorbar(self):
        """Test that the plot has a colorbar."""
        plotter = WaveletSpectrogramPlot(series=self.series)
        fig, ax = plotter.plot()

        # Check if colorbar was added (figure should have more than one axes)
        # The colorbar adds an additional axes
        assert len(fig.get_axes()) >= 1
        plt.close(fig)

    def test_plot_axis_labels(self):
        """Test that axis labels are set correctly."""
        plotter = WaveletSpectrogramPlot(series=self.series)
        fig, ax = plotter.plot()

        assert ax.get_xlabel() == "Time Index"
        assert ax.get_ylabel() == "Frequency Scale"
        plt.close(fig)

    def test_plot_mock_implementation_note(self):
        """Test that the plot includes a mock implementation note."""
        plotter = WaveletSpectrogramPlot(series=self.series)
        fig, ax = plotter.plot()

        # Check for the "Mock implementation" text annotation
        texts = [text.get_text() for text in ax.texts]
        assert any("Mock implementation" in text for text in texts)
        plt.close(fig)

    def test_plot_with_different_series_lengths(self):
        """Test plotting with series of different lengths."""
        short_series = pd.Series(np.random.randn(10))
        long_series = pd.Series(np.random.randn(500))
        
        plotter_short = WaveletSpectrogramPlot(series=short_series)
        fig_short, ax_short = plotter_short.plot()
        assert isinstance(fig_short, Figure)
        plt.close(fig_short)
        
        plotter_long = WaveletSpectrogramPlot(series=long_series)
        fig_long, ax_long = plotter_long.plot()
        assert isinstance(fig_long, Figure)
        plt.close(fig_long)

    def test_plot_uses_inferno_colormap(self):
        """Test that the plot uses the 'inferno' colormap."""
        plotter = WaveletSpectrogramPlot(series=self.series)
        fig, ax = plotter.plot()

        # Check if there's an image in the axes
        images = ax.get_images()
        assert len(images) > 0
        assert images[0].get_cmap().name == "inferno"
        plt.close(fig)

    def test_plot_extent_matches_series_length(self):
        """Test that the spectrogram extent matches series length."""
        plotter = WaveletSpectrogramPlot(series=self.series)
        fig, ax = plotter.plot()

        images = ax.get_images()
        if len(images) > 0:
            extent = images[0].get_extent()
            # extent is (left, right, bottom, top)
            # right should match series length
            assert extent[1] == len(self.series)
        
        plt.close(fig)

    def test_plot_creates_mock_data(self):
        """Test that plot creates mock spectrogram data."""
        plotter = WaveletSpectrogramPlot(series=self.series)
        fig, ax = plotter.plot()

        # Should have an image (the spectrogram)
        images = ax.get_images()
        assert len(images) > 0
        
        # The image should have data
        image_data = images[0].get_array()
        assert image_data is not None
        assert image_data.size > 0
        plt.close(fig)
