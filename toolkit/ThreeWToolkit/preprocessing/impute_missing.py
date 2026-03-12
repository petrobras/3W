import pandas as pd
from typing import Literal
from pydantic import Field, ValidationInfo, field_validator
from collections import defaultdict
from ..core.base_preprocessing import BasePreprocessing, BasePreprocessingConfig


class ImputeMissingConfig(BasePreprocessingConfig):
    strategy: Literal["mean", "median", "constant", "ffill", "bfill", "interpolate"] = (
        "interpolate"
    )
    fill_value: int | float | None = None
    columns: list[str] | None = None
    interpolate_method: Literal["linear", "nearest", "zero"] = "linear"
    target_: type = Field(default_factory=lambda: ImputeMissing)

    @field_validator("fill_value")
    def check_fill_value_for_constant(cls, v, info: ValidationInfo):
        strategy = info.data.get("strategy")
        if strategy == "constant" and v is None:
            raise ValueError("You must provide `fill_value` when strategy='constant'")
        return v

    @field_validator("interpolate_method")
    def check_interpolate_method(cls, v, info: ValidationInfo):
        strategy = info.data.get("strategy")
        if strategy == "interpolate" and v is None:
            raise ValueError(
                "You must provide `interpolate_method` when strategy='interpolate'"
            )
        return v


class ImputeMissing(BasePreprocessing):
    """
    A data processing step that handles missing values in signal columns using various imputation strategies.

    Supports global strategies (mean, median, constant) with statistics collected across events,
    and time-series strategies (ffill, bfill, interpolate) applied per-event.

    Attributes:
        config (ImputeMissingConfig): Configuration object containing imputation parameters
        collected (dict): Accumulated statistics for global imputation (only for mean/median/constant)
        impute_values (dict): Computed imputation values per column (only for mean/median/constant)
    """

    def __init__(
        self,
        config: ImputeMissingConfig,
    ):
        """
        Initialize the ImputeMissing step with the provided configuration.

        Args:
            config (ImputeMissingConfig): Configuration containing strategy, columns, and fill_value
        """
        self.config = config
        self.collected = defaultdict(lambda: {"sum": 0.0, "count": 0, "values": []})

    def fit(self, data: dict) -> None:
        """
        Collect statistics from a single event for imputation.

        Only for mean, median, constant: accumulates sum and count, or collects values.
        For ffill, bfill, interpolate: no collection needed (applied per-event).

        Args:
            data (dict): Input event data containing 'signal' DataFrame
        """
        if self.config.strategy in ["ffill", "bfill", "interpolate"]:
            return  # No global collection for time-series strategies

        signal_df = data.get("signal")
        if signal_df is None or not isinstance(signal_df, pd.DataFrame):
            return  # Skip if no signal data

        cols_to_collect = (
            self.config.columns
            if self.config.columns is not None
            else signal_df.columns.tolist()
        )

        for col in cols_to_collect:
            if col in signal_df.columns and pd.api.types.is_numeric_dtype(
                signal_df[col]
            ):
                values = signal_df[col].dropna()
                if self.config.strategy == "mean":
                    self.collected[col]["sum"] += values.sum()
                    self.collected[col]["count"] += len(values)
                elif self.config.strategy == "median":
                    self.collected[col]["values"].extend(values.tolist())

    def compute(self) -> None:
        """
        Compute imputation values from collected statistics.
        Only needed for mean, median, constant.
        """
        if self.config.strategy in ["ffill", "bfill", "interpolate"]:
            return  # No computation for time-series strategies

        self.impute_values = {}
        for col, stats in self.collected.items():
            if self.config.strategy == "mean" and stats["count"] > 0:
                self.impute_values[col] = stats["sum"] / stats["count"]
            elif self.config.strategy == "median" and stats["values"]:
                self.impute_values[col] = pd.Series(stats["values"]).median()
            elif self.config.strategy == "constant":
                self.impute_values[col] = self.config.fill_value
            else:
                self.impute_values[col] = 0.0

    def transform(self, data: dict) -> dict:
        """
        Execute the missing value imputation on the specified columns.

        For time-series strategies (ffill, bfill, interpolate): apply per-event.
        For global strategies (mean, median, constant): use pre-computed values.

        Args:
            data (dict): Input event data containing 'signal' DataFrame

        Returns:
            dict: Event data with imputed 'signal' DataFrame
        """
        signal_df = data.get("signal")
        if signal_df is None or not isinstance(signal_df, pd.DataFrame):
            return data

        # Determine which columns to impute
        cols_to_impute = (
            self.config.columns
            if self.config.columns is not None
            else signal_df.columns.tolist()
        )

        # Filter to valid columns
        valid_cols = [col for col in cols_to_impute if col in signal_df.columns]
        if not valid_cols:
            return data

        # Create a copy and impute
        data_copy = signal_df.copy()
        if self.config.strategy == "ffill":
            for col in valid_cols:
                data_copy[col] = data_copy[col].ffill()

        elif self.config.strategy == "bfill":
            for col in valid_cols:
                data_copy[col] = data_copy[col].bfill()

        elif self.config.strategy == "interpolate":
            for col in valid_cols:
                data_copy[col] = data_copy[col].interpolate(
                    method=self.config.interpolate_method
                )

        else:  # mean, median, constant
            if not hasattr(self, "impute_values"):
                raise ValueError(
                    "Impute values not computed. Call compute_statistics() first."
                )
            for col in valid_cols:
                if col in self.impute_values:
                    data_copy[col] = data_copy[col].fillna(self.impute_values[col])

        # Return the event dict with imputed signal
        result = data.copy()
        result["signal"] = data_copy

        return result
