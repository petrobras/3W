import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import pytest

from ThreeWToolkit.data_visualization.plots import DataVisualization


class TestPlotCorrelationHeatmap:
    def setup_method(self):
        """
        Setup method to create a DataFrame of time series.
        """
        np.random.seed(42)
        self.data = pd.DataFrame(
            {
                "A": np.random.normal(0, 1, 10),
                "B": np.random.normal(1, 2, 10),
                "C": np.random.normal(-1, 1, 10),
            }
        )

    def test_correlation_heatmap_returns_figure(self):
        """
        Test if correlation_heatmap returns a matplotlib Figure with given Axes.
        """
        fig, ax = plt.subplots()
        fig, _ = DataVisualization.correlation_heatmap(
            df_of_series=self.data, ax=ax, title="Test Heatmap"
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_correlation_heatmap_without_ax(self):
        """
        Test that correlation_heatmap works when no Axes is provided (ax=None).
        """
        fig, _ = DataVisualization.correlation_heatmap(
            df_of_series=self.data, title="No Ax Provided"
        )

        assert isinstance(fig, Figure)
        assert isinstance(fig.gca(), Axes)

    def test_correlation_heatmap_with_kwargs(self):
        """
        Test that correlation_heatmap accepts and applies extra kwargs (e.g., annot=True).
        """
        fig, _ = DataVisualization.correlation_heatmap(
            df_of_series=self.data,
            title="With Annotations",
            annot=True,
            fmt=".1f",
            cmap="coolwarm",
        )

        assert isinstance(fig, Figure)

    def test_correlation_heatmap_with_empty_dataframe(self):
        """
        Test that correlation_heatmap handles an empty DataFrame gracefully.
        """
        empty_df = pd.DataFrame()
        fig, _ = DataVisualization.correlation_heatmap(
            df_of_series=empty_df, title="Empty DataFrame"
        )

        assert isinstance(fig, Figure)
        assert isinstance(fig.gca(), Axes)

    def test_correlation_heatmap_only_nan_raises_value_error(self):
        """
        Test that correlation_heatmap raises ValueError when all values are NaN.
        """
        nan_df = pd.DataFrame(
            {
                "A": [np.nan, np.nan, np.nan],
                "B": [np.nan, np.nan, np.nan],
                "C": [np.nan, np.nan, np.nan],
            }
        )

        with pytest.raises(ValueError, match="Series contains only NaN values"):
            DataVisualization.correlation_heatmap(df_of_series=nan_df)
