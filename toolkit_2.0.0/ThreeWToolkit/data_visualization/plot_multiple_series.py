import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Optional
from typing import cast


class PlotMultipleSeries:
    @staticmethod
    def plot_multiple_series(
        series_list: list[pd.Series],
        labels: list[str],
        title: str,
        xlabel: str,
        ylabel: str,
        ax: Optional[Axes] = None,
        **plot_kwargs,
    ) -> Figure:
        """
        Static method to plot multiple time series on the same plot.

        Args:
            series_list (list[pd.Series]): List of series with datetime indices.
            labels (list[str]): List of labels for each series.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            ax (Optional[Axes]): Matplotlib Axes to plot into. Creates new if None.
            **plot_kwargs: Additional keyword arguments passed to `ax.plot`.

        Returns:
            matplotlib.figure.Figure: The resulting plot figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = cast(Figure, ax.figure)

        for series, label in zip(series_list, labels):
            ax.plot(series.index, series.values, label=label, **plot_kwargs)

        if ax.get_legend_handles_labels()[0]:
            ax.legend()

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)

        return fig
