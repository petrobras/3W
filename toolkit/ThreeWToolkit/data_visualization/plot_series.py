from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..core.base_visualizer import BaseVisualizer


class PlotSeries(BaseVisualizer):
    """
    Visualizer for plotting a single time series.

    Supports optional event overlay (vertical lines at NaN positions)
    and accepts additional matplotlib plotting parameters.
    """

    def __init__(
        self,
        series: pd.Series,
        title: str,
        xlabel: str,
        ylabel: str,
        overlay_events: bool = False,
        **plot_kwargs,
    ) -> None:
        """
        Initialize the single-series plot visualizer.

        Args:
            series: Input pandas Series to be plotted.
            title: Title of the plot.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            overlay_events: Whether to overlay event markers at NaN positions.
            **plot_kwargs: Additional keyword arguments forwarded to
                matplotlib Axes.plot.

        Returns:
            None.

        Raises:
            TypeError: If series is not a pandas Series.
        """
        self.series = series
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.overlay_events = overlay_events
        self.plot_kwargs = plot_kwargs

    def plot(self, ax: Axes | None = None) -> tuple[Figure, Axes]:
        """
        Plot a single time series.

        Args:
            ax: Matplotlib Axes to draw the time series on. If None,
                a new Figure and Axes are created.

        Returns:
            A tuple containing:
                - fig: The matplotlib Figure object.
                - ax: The matplotlib Axes with the rendered plot.

        Raises:
            ValueError: If the series is empty.
            ValueError: If the series contains only NaN values.
        """
        if self.series.empty:
            raise ValueError("Series is empty. Cannot generate plot.")

        if self.series.dropna().empty:
            raise ValueError("Series contains only NaN values. Nothing to plot.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = cast(Figure, ax.get_figure())

        ax.plot(
            self.series.index,
            self.series.values,
            label="Value",
            **self.plot_kwargs,
        )
        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.grid(True, alpha=0.3)

        if self.overlay_events:
            nan_dates = self.series.index[self.series.isna()]
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

        return fig, ax
