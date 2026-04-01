import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import pytest

from ThreeWToolkit.data_visualization import DataVisualization


class TestPlotSeries:
    def setup_method(self):
        """
        Setup method to create a default time series.
        """
        np.random.seed(42)
        self.dates = pd.date_range("2024-01-01", periods=10, freq="D")
        self.values = np.random.normal(loc=10, scale=2, size=10)
        self.series = pd.Series(self.values, index=self.dates)

    def test_plot_series_returns_figure(self):
        """
        Test if plot_series returns a matplotlib Figure with given Axes.
        """
        fig, ax = plt.subplots()
        fig, _ = DataVisualization.plot_series(
            series=self.series,
            title="Test Plot",
            xlabel="Date",
            ylabel="Value",
            overlay_events=False,
            ax=ax,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_plot_series_without_ax(self):
        """
        Test that plot_series works when no Axes is provided (ax=None).
        """
        fig, _ = DataVisualization.plot_series(
            series=self.series,
            title="No Ax Provided",
            xlabel="Date",
            ylabel="Value",
            overlay_events=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(fig.gca(), Axes)

    def test_plot_series_with_nan_overlay(self):
        """
        Test if overlay_events handles NaN values without crashing.
        """
        series_with_nan = self.series.copy()
        series_with_nan.iloc[3] = np.nan

        fig, ax = plt.subplots()
        fig, _ = DataVisualization.plot_series(
            series=series_with_nan,
            title="Series with NaN",
            xlabel="Date",
            ylabel="Value",
            overlay_events=True,
            ax=ax,
        )

        assert isinstance(fig, Figure)
        assert ax.has_data()
        plt.close(fig)

    def test_plot_series_without_overlay(self):
        """
        Test if plot renders correctly without overlay.
        """
        fig, ax = plt.subplots()
        fig, _ = DataVisualization.plot_series(
            series=self.series,
            title="Without Overlay",
            xlabel="Date",
            ylabel="Value",
            overlay_events=False,
            ax=ax,
        )

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_plot_series_with_empty_series(self):
        """
        Test that an empty series raises a ValueError.
        """
        empty_series = pd.Series(dtype=float)

        with pytest.raises(ValueError):
            DataVisualization.plot_series(
                series=empty_series,
                title="Empty Series",
                xlabel="Date",
                ylabel="Value",
                overlay_events=False,
            )

    def test_plot_series_with_all_nan(self):
        """
        Test that a series with only NaN values raises a ValueError.
        """
        nan_series = pd.Series(
            [np.nan, np.nan, np.nan], index=pd.date_range("2024-01-01", periods=3)
        )

        with pytest.raises(ValueError):
            DataVisualization.plot_series(
                series=nan_series,
                title="Only NaN",
                xlabel="Date",
                ylabel="Value",
                overlay_events=True,
            )
