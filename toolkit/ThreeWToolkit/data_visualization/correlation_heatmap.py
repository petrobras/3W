from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..core.base_visualizer import BaseVisualizer


class CorrelationHeatmap(BaseVisualizer):
    """
    Visualizer for computing and plotting a correlation heatmap.

    This class receives a DataFrame of series, computes the correlation
    matrix, and renders it as a heatmap using seaborn.
    """

    def __init__(
        self,
        df_of_series: pd.DataFrame,
        **kwargs,
    ) -> None:
        """
        Initialize the CorrelationHeatmap visualizer.

        Args:
            df_of_series: DataFrame containing multiple series or variables
                used to compute the correlation matrix.
            **kwargs: Optional keyword arguments forwarded to seaborn.heatmap,
                such as title, figsize, colormap settings, and annotation options.

        Returns:
            None.

        Raises:
            TypeError: If df_of_series is not a pandas DataFrame.
        """
        self.df_of_series = df_of_series
        self.kwargs = kwargs

    def plot(self, ax: Axes | None = None) -> tuple[Figure, Axes]:
        """
        Plot the correlation heatmap.

        Args:
            ax: Matplotlib Axes to draw the heatmap on. If None, a new
                Figure and Axes are created.

        Returns:
            A tuple containing:
                - fig: The matplotlib Figure object.
                - ax: The matplotlib Axes where the heatmap is rendered.

        Raises:
            ValueError: If the DataFrame contains only NaN values.
        """
        title: str = self.kwargs.pop("title", "Correlation Heatmap")
        figsize: tuple = self.kwargs.pop("figsize", (10, 8))

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = cast(Figure, ax.get_figure())

        if self.df_of_series.empty:
            ax.text(
                0.5,
                0.5,
                "Empty DataFrame\nNo data to display",
                fontsize=14,
                ha="center",
                va="center",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="lightgray"),
            )
            ax.set_title(title)
            fig.tight_layout()
            return fig, ax

        if self.df_of_series.isna().all().all():
            raise ValueError("Series contains only NaN values")

        heatmap_defaults = {
            "annot": True,
            "fmt": ".2f",
            "cmap": "coolwarm",
            "center": 0,
            "square": True,
            "cbar_kws": {"shrink": 0.8},
        }
        for key, value in heatmap_defaults.items():
            self.kwargs.setdefault(key, value)

        corr_matrix = self.df_of_series.corr()
        sns.heatmap(corr_matrix, ax=ax, **self.kwargs)
        ax.set_title(title)
        fig.tight_layout()

        return fig, ax
