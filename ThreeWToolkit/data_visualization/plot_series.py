import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Optional
from typing import cast


class DataVisualization:
    @staticmethod
    def plot_series(
        series: pd.Series,
        title: str,
        xlabel: str,
        ylabel: str,
        overlay_events: bool = False,
        ax: Optional[Axes] = None,
        **plot_kwargs,
    ) -> Figure:
        """
        Static method to plot a time series.

        Args:
            series (pd.Series): Series with datetime index.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            overlay_events (bool): Whether to overlay vertical lines for events.
            ax (Optional[Axes]): Matplotlib Axes to plot into. Creates new if None.
            **plot_kwargs: Additional keyword arguments passed to `ax.plot`.

        Returns:
            matplotlib.figure.Figure: The resulting plot figure.
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = cast(Figure, ax.figure)

        ax.plot(series.index, series.values, label="Value", **plot_kwargs)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)

        if overlay_events:
            for date in series.index[series.isna()]:
                ax.axvline(x=date, color="red", linestyle="--", alpha=0.7)

        return fig
