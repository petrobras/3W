import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from statsmodels.tsa.seasonal import seasonal_decompose

from ..core.base_visualizer import BaseVisualizer


class SeasonalDecompositionPlot(BaseVisualizer):
    """
    Visualizer for performing seasonal decomposition on a time series
    and plotting its components (observed, trend, seasonal, residual).
    """

    def __init__(
        self,
        series: pd.Series,
        model: str = "additive",
        period: int | None = None,
    ) -> None:
        """
        Initialize the seasonal decomposition visualizer.

        Args:
            series: Input pandas Series to decompose.
            model: Type of seasonal component. Must be 'additive' or
                'multiplicative'.
            period: Period of the seasonal component.

        Returns:
            None.

        Raises:
            TypeError: If series is not a pandas Series.
        """
        self.series = series
        self.model = model
        self.period = period

    def plot(self, ax: Axes | None = None) -> tuple[Figure, Axes]:
        """
        Perform seasonal decomposition and plot its components.

        Args:
            ax: Unused. Present only for API consistency.

        Returns:
            A tuple containing:
                - fig: The matplotlib Figure object.
                - ax: The matplotlib Axes corresponding to the observed component.

        Raises:
            ValueError: If the input series is empty.
            ValueError: If the model is not 'additive' or 'multiplicative'.
            ValueError: If the series is too short for decomposition.
            ValueError: If the decomposition process fails.
        """
        if self.series.empty:
            raise ValueError("Input series is empty")

        if self.model not in ["additive", "multiplicative"]:
            raise ValueError("Model must be either 'additive' or 'multiplicative'")

        clean_series = self.series.dropna()
        if len(clean_series) < 10:
            raise ValueError(
                "Series too short for decomposition (minimum 10 points required)"
            )

        try:
            result = seasonal_decompose(
                clean_series, model=self.model, period=self.period
            )
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Decomposition failed: {str(e)}") from e

        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

        # Observed
        result.observed.plot(ax=axes[0], color="blue", linewidth=1.5)
        axes[0].set_ylabel("Observed")
        axes[0].set_title("Original Time Series")
        axes[0].grid(True, alpha=0.3)

        # Trend
        result.trend.plot(ax=axes[1], color="red", linewidth=1.5)
        axes[1].set_ylabel("Trend")
        axes[1].set_title("Trend Component")
        axes[1].grid(True, alpha=0.3)

        # Seasonal
        result.seasonal.plot(ax=axes[2], color="green", linewidth=1.5)
        axes[2].set_ylabel("Seasonal")
        axes[2].set_title("Seasonal Component")
        axes[2].grid(True, alpha=0.3)

        # Residual
        result.resid.plot(ax=axes[3], color="orange", linewidth=1.5)
        axes[3].set_ylabel("Residual")
        axes[3].set_title("Residual Component")
        axes[3].set_xlabel("Date")
        axes[3].grid(True, alpha=0.3)

        fig.suptitle(
            f"Seasonal Decomposition ({self.model.title()} Model)",
            fontsize=16,
            y=0.98,
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.94)

        return fig, axes[0]
