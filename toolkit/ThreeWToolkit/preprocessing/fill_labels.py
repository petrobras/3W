from typing import Literal
from pydantic import Field, field_validator, ValidationInfo

from ..core.base_dataset import BaseDataset
from ..core.base_preprocessing import BasePreprocessing, BasePreprocessingConfig
from ..core.dataset_outputs import DatasetOutputs


class FillLabelsConfig(BasePreprocessingConfig):
    fill_method: Literal["nearest", "ffill", "bfill", "constant"] = Field(
        default="nearest", description="Method to fill missing values in labels."
    )
    fill_value: int | None = Field(
        default=None, description="Value to use when fill_method is 'constant'."
    )
    target_: type = Field(default_factory=lambda: FillLabels)

    @field_validator("fill_value")
    @classmethod
    def validate_fill_value(cls, value, info: ValidationInfo):
        if info.data.get("fill_method") == "constant" and value is None:
            raise ValueError(
                "fill_value must be provided when fill_method is 'constant'."
            )
        return value


class FillLabels(BasePreprocessing):
    """
    Preprocessing step to fill missing values in label Series using bfill then ffill.
    Applies to each event's label data.
    """

    def __init__(self, config: FillLabelsConfig):
        self.config: FillLabelsConfig = config

    def fit(self, data: BaseDataset) -> None:
        """No need to fit anything for this preprocessing step."""
        pass

    def transform(self, data: DatasetOutputs) -> DatasetOutputs:
        """
        Fill missing values in the label Series according to the specified method.
            - If fill_method is 'constant', fill missing values with the specified fill_value.
            - If fill_method is 'ffill', fill missing values using forward fill then backward fill.
            - If fill_method is 'bfill', fill missing values using backward fill then forward fill.
            - If fill_method is 'nearest', fill missing values using nearest interpolation then bfill and ffill to handle any remaining NaNs.
        """

        if data.label is None: # Noop if there are no labels to fill
            return data

        if self.config.fill_method == "constant":
            label = data.label.fillna(self.config.fill_value)
        elif self.config.fill_method == "ffill":
            label = (
                data.label.ffill().bfill()
            )  # Fill missing values using forward fill then backward fill
        elif self.config.fill_method == "bfill":
            label = (
                data.label.bfill().ffill()
            )  # Fill missing values using backward fill then forward fill
        elif self.config.fill_method == "nearest":
            # Fill missing values using nearest interpolation then bfill and ffill to handle any remaining NaNs
            label = data.label.interpolate(method="nearest").bfill().ffill()
        else:
            raise ValueError(f"Unsupported fill method: {self.config.fill_method}")

        return DatasetOutputs(signal=data.signal, label=label, metadata=data.metadata)
