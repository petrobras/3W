from ThreeWToolkit.core.base_dataset import BaseDataset
import pandas as pd
from typing import Literal
from pydantic import Field, ValidationInfo, field_validator

from ..dataset.transformed_dataset import TransformedDataset
from ..core.base_preprocessing import BasePreprocessing, BasePreprocessingConfig
from ..core.dataset_outputs import DatasetOutputs


class ImputeMissingConfig(BasePreprocessingConfig):
    missing_column_threshold: float = Field(
        default=0.6,
        description="Drop columns that are all-NaN in more than this fraction of events.",
    )
    strategy: Literal["constant", "mean", "ffill", "bfill", "interpolate"] = (
        "constant"
    )
    fill_value: float | None = 0.0
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

    Supports global strategies (mean, constant) with statistics collected across events,
    and time-series strategies (ffill, bfill, interpolate) applied per-event.

    Attributes:
        config (ImputeMissingConfig): Configuration object containing imputation parameters
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
        self.drop_columns = None
        self.global_average = None

    def fit(self, data: BaseDataset) -> None:
        """
        Collect event statistics needed for imputation.

        Only for mean, median, constant: accumulates sum and count, or collects values.
        For ffill, bfill, interpolate: no collection needed (applied per-event).

        Args:
            data (dict): Input event data containing 'signal' DataFrame
        """
        # Verify if dataset passes nan threshold check and determine columns to drop based on all-NaN fraction
        is_all_nan = []
        for event in data:
            is_all_nan.append(event.signal.isna().all())
        is_all_nan = pd.concat(is_all_nan, axis=1).transpose()

        # Track columns that are all-NaN in each event to determine which columns to drop based on the configured threshold
        drop_columns = is_all_nan.mean() < self.config.missing_column_threshold
        self.drop_columns = drop_columns.index[drop_columns].tolist() # type: ignore


        if self.config.strategy in ["constant", "ffill", "bfill", "interpolate"]:
            return  # No global collection needed for time-series strategies

        # Fit to find values to impute for.
        dropped_cols_dataset = TransformedDataset(data, self._drop_columns)
        self._compute_global_average(dropped_cols_dataset)

    def transform(self, data: DatasetOutputs) -> DatasetOutputs:
        """
        Execute the missing value imputation on the specified columns.

        Also drops events (rows) where the label column is NaN.

        For time-series strategies (ffill, bfill, interpolate): apply per-event.
        For global strategies (mean, median, constant): use pre-computed values.

        Args:
            data (dict): Input event data containing 'signal' DataFrame

        Returns:
            dict: Event data with imputed 'signal' DataFrame
        """

        # drop missing columns
        data = self._drop_columns(data)

        if self.config.strategy == "constant":
            data.signal = data.signal.fillna(self.config.fill_value)
            return data
        if self.config.strategy == "mean":
            if self.global_average is None:
                raise ValueError("Global average not computed. Call fit() first.")
            data.signal = data.signal.fillna(self.global_average)
            return data

        if self.config.strategy == "interpolate":
            data.signal = data.signal.interpolate(method=self.config.interpolate_method)
        if self.config.strategy == "ffill":
            data.signal = data.signal.ffill()
        if self.config.strategy == "bfill":
            data.signal = data.signal.bfill()

        # if post-imputation there are still missing values, print a warning
        if data.signal.isna().all().any(): # type: ignore
            print(
                "[ImputeMissing] Warning: After imputation, there are still missing values in the signal data."
            )

        return data


    def _drop_columns(self, data: DatasetOutputs) -> DatasetOutputs:
        if self.drop_columns is not None:
            data.signal = data.signal.drop(columns=self.drop_columns)
        return data

    def _compute_global_average(self, data: BaseDataset) -> None:
        averages = []
        counts   = []
        for event in data:
            averages.append(event.signal.mean())
            counts.append(event.signal.count())
        # compute weighted average of the averages
        averages = pd.concat(averages, axis=1).transpose()
        counts = pd.concat(counts, axis=1).transpose()

        self.global_average = (averages * counts).sum() / counts.sum()

