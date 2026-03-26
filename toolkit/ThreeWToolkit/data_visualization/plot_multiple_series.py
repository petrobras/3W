from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from .base_visualizer import BaseVisualizer


class PlotMultipleSeries(BaseVisualizer):
    """
    Visualizer for plotting multiple time series on the same axes.

    Receives a list of Series and corresponding labels, and renders them
    together using a consistent color palette.
    """

    def __init__(
        self,
        series_list: list[pd.Series],
        labels: list[str],
        title: str,
        xlabel: str,
        ylabel: str,
        **plot_kwargs,
    ) -> None:
        """
        Initialize the multiple-series plot visualizer.

        Args:
            series_list: List of pandas Series to be plotted.
            labels: List of labels corresponding to each series.
            title: Title of the plot.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            **plot_kwargs: Additional keyword arguments forwarded to
                matplotlib Axes.plot.

        Returns:
            None.

        Raises:
            TypeError: If series_list or labels are not valid lists.
            ValueError: If series_list is empty.
            ValueError: If series_list and labels have different lengths.
        """
        # Explicit type checks, as requested in the review
        if not isinstance(series_list, list):
            raise TypeError("series_list must be a list")
        if not isinstance(labels, list):
            raise TypeError("labels must be a list")

        if not series_list:
            raise ValueError("series_list must not be empty")
        if len(series_list) != len(labels):
            raise ValueError("series_list and labels must have the same length")

        self.series_list = series_list
        self.labels = labels
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_kwargs = plot_kwargs

    def plot(self, ax: Axes | None = None) -> tuple[Figure, Axes]:
        """
        Plot multiple time series on a single Axes.

        Args:
            ax: Matplotlib Axes to draw the time series on. If None,
                a new Figure and Axes are created.

        Returns:
            A tuple containing:
                - fig: The matplotlib Figure object.
                - ax: The matplotlib Axes with the rendered time series.

        Raises:
            None.
        """
        if ax is None:
            fig = Figure(figsize=(12, 6))
            FigureCanvas(fig)
            ax = fig.add_subplot(1, 1, 1)
        else:
            fig = cast(Figure, ax.get_figure())

        cmap = plt.get_cmap("Set1", len(self.series_list))
        colors = [cmap(i) for i in range(len(self.series_list))]

        plotted_any = False
        for i, (series, label) in enumerate(
            zip(self.series_list, self.labels, strict=True)
        ):
            # Use dropna() only to check if the series is completely empty;
            # keep the original series for plotting so NaNs still create gaps.
            clean_series = series.dropna()
            if clean_series.empty:
                continue

            ax.plot(
                series.index,
                series.values,
                label=label,
                color=colors[i],
                **self.plot_kwargs,
            )
            plotted_any = True

        if plotted_any:
            ax.legend(loc="best")

        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.grid(True, alpha=0.3)

        return fig, ax
