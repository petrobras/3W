import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import cast
from ..core.base_data_visualization import BaseDataVisualization


class PlotCorrelationHeatmap(BaseDataVisualization):
    @staticmethod
    def correlation_heatmap(
        df_of_series: pd.DataFrame, ax: Axes | None = None, **kwargs
    ) -> Figure:
        """
        Static method to plot a correlation heatmap.

        Args:
            df_of_series (pd.DataFrame): DataFrame where each column is a time series.
            ax (Optional[Axes]): Matplotlib Axes to draw the plot onto. If None, a new figure and axes are created.
            **kwargs: Additional keyword arguments passed to sns.heatmap() or used internally
                      (e.g., title=..., figsize=...).

        Returns:
            matplotlib.figure.Figure: The resulting plot figure.
        """

        title: str = kwargs.pop("title", "Correlation Heatmap")
        figsize: tuple = kwargs.pop("figsize", (10, 8))

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = cast(Figure, ax.figure)

        if df_of_series.empty:
            ax.text(
                0.5,
                0.5,
                "Empty DataFrame",
                fontsize=14,
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(title)
            fig.tight_layout()
            return fig

        corr_matrix = df_of_series.corr()
        sns.heatmap(corr_matrix, ax=ax, **kwargs)
        ax.set_title(title)
        fig.tight_layout()

        return fig
