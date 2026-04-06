import pandas as pd
from typing import Literal
from pydantic import Field, ValidationInfo, field_validator, PrivateAttr

from ..core.base_dataset import BaseDataset
from ..core.base_preprocessing import BasePreprocessing, BasePreprocessingConfig
from ..core.dataset_outputs import DatasetOutputs


class ImputeMissingConfig(BasePreprocessingConfig):
    strategy: Literal["constant", "mean", "ffill", "bfill", "interpolate"] = Field(
        default="constant",
        description="Imputation strategy to use for filling missing values. Options include:\n"
        "- 'constant': Fill missing values with a specified constant value (requires `fill_value`).\n"
        "- 'mean': Fill missing values with the mean of the column (computed across all events during fit).\n"
        "- 'ffill': Forward-fill missing values using the last valid observation (applied per-event).\n"
        "- 'bfill': Backward-fill missing values using the next valid observation (applied per-event).\n"
        "- 'interpolate': Fill missing values using interpolation (requires `interpolate_method`, applied per-event).",
    )
    fill_value: float | None = Field(
        default=0.0,
        description="The constant value to use for filling missing values when strategy='constant'.\
                     This field is required if strategy is set to 'constant'.",
    )
    # columns: list[str] | None = None
    interpolate_method: Literal["linear", "nearest", "zero"] | None = Field(
        default=None,
        description="The interpolation method to use when strategy='interpolate'.\
                     This field is required if strategy is set to 'interpolate'. Options include:\n"
        "- 'linear': Linear interpolation (default)\n"
        "- 'nearest': Nearest-neighbor interpolation\n"
        "- 'zero': Step-wise interpolation (previous value)",
    )
    _target: type = PrivateAttr(default_factory=lambda: ImputeMissing)

    @field_validator("fill_value")
    def check_fill_value_for_constant(cls, fill_value, info: ValidationInfo):
        strategy = info.data.get("strategy")
        if strategy == "constant" and fill_value is None:
            raise ValueError("You must provide `fill_value` when strategy='constant'")
        return fill_value

    @field_validator("interpolate_method")
    def check_interpolate_method(cls, interpolate_method, info: ValidationInfo):
        strategy = info.data.get("strategy")
        if strategy == "interpolate" and interpolate_method is None:
            raise ValueError(
                "You must provide `interpolate_method` when strategy='interpolate'"
            )
        return interpolate_method


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
        self.config: ImputeMissingConfig = config
        self.global_average: pd.Series | None = None

    def fit(self, data: BaseDataset) -> None:
        """
        Collect event statistics needed for imputation.

        Only for mean, median, constant: accumulates sum and count, or collects values.
        For ffill, bfill, interpolate: no collection needed (applied per-event).

        Args:
            data (dict): Input event data containing 'signal' DataFrame
        """
        # Verify if dataset passes nan threshold check and determine columns to drop based on all-NaN fraction
        if self.config.strategy in ["constant", "ffill", "bfill", "interpolate"]:
            return  # No global collection needed for time-series strategies

        self._compute_global_average(data)

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

        signal = data.signal.copy().astype(float)
        if self.config.strategy == "constant":
            signal = signal.fillna(self.config.fill_value)
        elif self.config.strategy == "mean":
            if self.global_average is None:
                raise ValueError("Global average not computed. Call fit() first.")
            print(self.global_average)
            signal = signal.fillna(self.global_average)

        elif (
            self.config.strategy == "interpolate"
            and self.config.interpolate_method is not None
        ):
            signal = (
                signal.interpolate(method=self.config.interpolate_method)
                .bfill()
                .ffill()
            )  # interpolate
            # then fill any remaining NaNs
        elif self.config.strategy == "ffill":
            signal = (
                signal.ffill().bfill()
            )  # forward-fill then backward-fill to handle leading NaNs
        else:  # self.config.strategy == "bfill":
            signal = (
                signal.bfill().ffill()
            )  # backward-fill then forward-fill to handle trailing NaNs

        # if post-imputation there are still missing values, print a warning
        if signal.isna().all().any():  # type: ignore
            raise RuntimeError(
                "Imputation failed: some columns still contain all NaN values after imputation. Check your data and imputation strategy."
            )

        return DatasetOutputs(signal=signal, label=data.label, metadata=data.metadata)

    def _compute_global_average(self, data: BaseDataset) -> None:
        _sums = []
        _counts = []
        for event in data:
            _sums.append(event.signal.sum())
            _counts.append(event.signal.count())
        # compute weighted average of the sums
        sums = pd.concat(_sums, axis=1).transpose()
        counts = pd.concat(_counts, axis=1).transpose()

        self.global_average = sums.mean() / counts.mean()
