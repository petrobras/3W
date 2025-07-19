import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ThreeWToolkit.data_visualization import PlotMultipleSeries


class TestPlotMultipleSeries:
    
    def setup_method(self):
        """
        Setup method to create multiple time series.
        """
        np.random.seed(42)
        self.dates = pd.date_range("2024-01-01", periods=10, freq="D")
        self.series1 = pd.Series(np.random.normal(10, 2, 10), index=self.dates)
        self.series2 = pd.Series(np.random.normal(20, 3, 10), index=self.dates)
        self.series_list = [self.series1, self.series2]
        self.labels = ["Series 1", "Series 2"]

    def test_plot_multiple_series_returns_figure(self):
        """
        Test if plot_multiple_series returns a matplotlib Figure with given Axes.
        """
        fig, ax = plt.subplots()
        fig = PlotMultipleSeries.plot_multiple_series(
            series_list=self.series_list,
            labels=self.labels,
            title="Test Multiple Series",
            xlabel="Date",
            ylabel="Value",
            ax=ax
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_plot_multiple_series_without_ax(self):
        """
        Test that plot_multiple_series works when no Axes is provided (ax=None).
        """
        fig = PlotMultipleSeries.plot_multiple_series(
            series_list=self.series_list,
            labels=self.labels,
            title="No Ax Provided",
            xlabel="Date",
            ylabel="Value"
        )

        assert isinstance(fig, Figure)
        assert isinstance(fig.gca(), Axes)

    def test_plot_multiple_series_with_empty_list(self):
        """
        Test that empty series_list returns a figure without error.
        """
        fig, ax = plt.subplots()
        fig = PlotMultipleSeries.plot_multiple_series(
            series_list=[],
            labels=[],
            title="Empty Series List",
            xlabel="Date",
            ylabel="Value",
            ax=ax
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.lines) == 0