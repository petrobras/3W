import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ThreeWToolkit.data_visualization.seasonal_decomposition import (
    SeasonalDecompositionPlot,
)


class TestSeasonalDecompositionPlot:
    def setup_method(self):
        """Setup method to create test time series with seasonal pattern."""
        np.random.seed(42)
        # Create a longer series for decomposition (needs at least 2 full periods)
        n_points = 100
        time = np.arange(n_points)
        trend = 0.5 * time
        seasonal = 10 * np.sin(2 * np.pi * time / 12)
        noise = np.random.normal(0, 1, n_points)

        self.dates = pd.date_range("2024-01-01", periods=n_points, freq="D")
        self.values = trend + seasonal + noise
        self.series = pd.Series(self.values, index=self.dates)

    def test_initialization(self):
        """Test that SeasonalDecompositionPlot initializes correctly."""
        plotter = SeasonalDecompositionPlot(
            series=self.series, model="additive", period=12
        )

        assert plotter.series.equals(self.series)
        assert plotter.model == "additive"
        assert plotter.period == 12

    def test_initialization_default_model(self):
        """Test initialization with default model."""
        plotter = SeasonalDecompositionPlot(series=self.series)

        assert plotter.model == "additive"

    def test_plot_returns_figure_and_axes(self):
        """Test that plot returns a Figure and Axes."""
        plotter = SeasonalDecompositionPlot(
            series=self.series, model="additive", period=12
        )
        fig, ax = plotter.plot()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_plot_creates_four_subplots(self):
        """Test that plot creates 4 subplots for decomposition components."""
        plotter = SeasonalDecompositionPlot(
            series=self.series, model="additive", period=12
        )
        fig, ax = plotter.plot()

        # Should have 4 axes: observed, trend, seasonal, residual
        assert len(fig.axes) == 4
        plt.close(fig)

    def test_plot_additive_model(self):
        """Test plotting with additive model."""
        plotter = SeasonalDecompositionPlot(
            series=self.series, model="additive", period=12
        )
        fig, ax = plotter.plot()

        assert isinstance(fig, Figure)
        assert "Additive" in fig._suptitle.get_text()
        plt.close(fig)

    def test_plot_multiplicative_model(self):
        """Test plotting with multiplicative model."""
        # Create positive values for multiplicative model
        positive_series = pd.Series(np.abs(self.values) + 10, index=self.dates)
        plotter = SeasonalDecompositionPlot(
            series=positive_series, model="multiplicative", period=12
        )
        fig, ax = plotter.plot()

        assert isinstance(fig, Figure)
        assert "Multiplicative" in fig._suptitle.get_text()
        plt.close(fig)

    def test_plot_empty_series_raises_error(self):
        """Test that plotting an empty series raises ValueError."""
        empty_series = pd.Series(dtype=float)
        plotter = SeasonalDecompositionPlot(series=empty_series)

        with pytest.raises(ValueError, match="Input series is empty"):
            plotter.plot()

    def test_plot_invalid_model_raises_error(self):
        """Test that invalid model type raises ValueError."""
        plotter = SeasonalDecompositionPlot(
            series=self.series, model="invalid_model", period=12
        )

        with pytest.raises(
            ValueError, match="Model must be either 'additive' or 'multiplicative'"
        ):
            plotter.plot()

    def test_plot_series_too_short_raises_error(self):
        """Test that series too short for decomposition raises ValueError."""
        short_series = pd.Series([1, 2, 3, 4, 5])
        plotter = SeasonalDecompositionPlot(series=short_series, period=2)

        with pytest.raises(ValueError, match="Series too short for decomposition"):
            plotter.plot()

    def test_plot_with_nan_values(self):
        """Test that series with NaN values is handled (NaNs dropped)."""
        series_with_nan = self.series.copy()
        series_with_nan.iloc[10:15] = np.nan

        plotter = SeasonalDecompositionPlot(
            series=series_with_nan, model="additive", period=12
        )
        fig, ax = plotter.plot()

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_with_auto_period(self):
        """Test plotting with automatic period detection (period=None)."""
        plotter = SeasonalDecompositionPlot(
            series=self.series, model="additive", period=None
        )
        fig, ax = plotter.plot()

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_ax_parameter_ignored(self):
        """Test that ax parameter is ignored (for API consistency)."""
        plotter = SeasonalDecompositionPlot(
            series=self.series, model="additive", period=12
        )
        fig_orig, ax_orig = plt.subplots()

        # Even with ax provided, function creates new figure with 4 subplots
        fig, ax = plotter.plot(ax=ax_orig)

        assert isinstance(fig, Figure)
        # Should still have 4 axes, not use the provided one
        assert len(fig.axes) == 4
        plt.close(fig)
        plt.close(fig_orig)

    def test_decomposition_requires_minimum_periods(self):
        """Test that decomposition requires sufficient data points for the period."""
        # Create a series with fewer points than 2*period
        short_series = pd.Series(
            np.random.randn(15), index=pd.date_range("2024-01-01", periods=15)
        )
        plotter = SeasonalDecompositionPlot(
            series=short_series, model="additive", period=12
        )

        # Should raise error because series is too short for 2 full periods
        with pytest.raises(ValueError):
            plotter.plot()

    def test_plot_component_labels(self):
        """Test that all components have proper labels."""
        plotter = SeasonalDecompositionPlot(
            series=self.series, model="additive", period=12
        )
        fig, ax = plotter.plot()

        axes = fig.axes
        assert "Observed" in axes[0].get_ylabel()
        assert "Trend" in axes[1].get_ylabel()
        assert "Seasonal" in axes[2].get_ylabel()
        assert "Residual" in axes[3].get_ylabel()
        plt.close(fig)
