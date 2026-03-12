import numpy as np
import pandas as pd
from typing import Literal
from pydantic import Field
from collections import defaultdict
from ..core.base_preprocessing import BasePreprocessing, BasePreprocessingConfig


class NormalizeConfig(BasePreprocessingConfig):
    norm: Literal["l1", "l2", "max"] = "l2"
    axis: Literal[0, 1] = 0
    copy_values: bool = True
    return_norm_values: bool = False
    target_: type = Field(default_factory=lambda: Normalize)


class Normalize(BasePreprocessing):
    """
    A data processing step that normalizes signal data using z-score normalization.

    Collects statistics (mean and std) from signal columns across events during training,
    then applies normalization to standardize the signals.

    Attributes:
        config (NormalizeConfig): Configuration object containing normalization parameters
        collected (dict): Accumulated statistics for each signal column
        statistics (dict): Computed mean and std for each signal column
    """

    def __init__(
        self,
        config: NormalizeConfig,
    ):
        """
        Initialize the Normalize step with the provided configuration.

        Args:
            config (NormalizeConfig): Configuration containing norm type, axis, and other parameters
        """
        self.config = config
        self.collected = defaultdict(lambda: {"sum": 0.0, "sum_sq": 0.0, "count": 0})

    def fit(self, data: dict) -> None:
        """
        Collect statistics from a single event for aggregation.

        Accumulates sum, sum of squares, and count for each column in the 'signal' DataFrame across events.

        Args:
            data (dict): Input event data containing 'signal' DataFrame
        """
        signal_df = data.get("signal")
        if signal_df is None or not isinstance(signal_df, pd.DataFrame):
            return  # Skip if no signal data

        for col in signal_df.columns:
            values = signal_df[col].dropna()  # Handle NaN values
            self.collected[col]["sum"] += values.sum()
            self.collected[col]["sum_sq"] += (values**2).sum()
            self.collected[col]["count"] += len(values)

    def compute(self) -> None:
        """
        Compute global statistics from all collected data.

        Calculates mean and standard deviation for each column.
        """
        self.statistics = {}
        for col, stats in self.collected.items():
            if stats["count"] > 0:
                mean = stats["sum"] / stats["count"]
                var = (stats["sum_sq"] - (stats["sum"] ** 2) / stats["count"]) / stats[
                    "count"
                ]
                std = np.sqrt(max(var, 0))  # Ensure non-negative variance
                self.statistics[col] = {"mean": mean, "std": std}
            else:
                self.statistics[col] = {"mean": 0.0, "std": 1.0}
        pass

    def transform(self, data: dict) -> dict:
        """
        Apply normalization to the 'signal' data using computed statistics.

        Performs standard z-score normalization: (x - mean) / std on signal columns.

        Args:
            data (dict): Input event data containing 'signal' DataFrame

        Returns:
            dict: Event data with normalized 'signal' DataFrame
        """
        if self.statistics is None:
            raise ValueError(
                "Statistics not computed. Call compute_statistics() first."
            )

        signal_df = data.get("signal")
        if signal_df is None or not isinstance(signal_df, pd.DataFrame):
            return data  # Return unchanged if no signal data

        normalized_signal = signal_df.copy()
        for col in signal_df.columns:
            if col in self.statistics:
                mean = self.statistics[col]["mean"]
                std = self.statistics[col]["std"]
                if std > 0:
                    normalized_signal[col] = (signal_df[col] - mean) / std
                else:
                    normalized_signal[col] = (
                        signal_df[col] - mean
                    )  # If std is 0, just center

        # Return the event dict with normalized signal
        result = data.copy()
        result["signal"] = normalized_signal
        return result
