from ThreeWToolkit.core.base_dataset import BaseDataset
import pandas as pd
from typing import Literal
from pydantic import Field, field_validator
from ..core.base_preprocessing import BasePreprocessing, BasePreprocessingConfig
from ..core.dataset_outputs import DatasetOutputs


class FillLabelsConfig(BasePreprocessingConfig):
    fill_method: Literal["nearest", "ffill", "bfill", "constant"] = Field(default="nearest", description="Method to fill missing values in labels.")
    fill_value: int | None = Field(default=None, description="Value to use when fill_method is 'constant'.")
    target_: type = Field(default_factory=lambda: FillLabels)

    @field_validator("fill_value")
    def validate_fill_value(cls, value, values):
        if values.get("fill_method") == "constant" and value is None:
            raise ValueError("fill_value must be provided when fill_method is 'constant'.")
        return value


class FillLabels(BasePreprocessing):
    """
    Preprocessing step to fill missing values in label Series using bfill then ffill.
    Applies to each event's label data.
    """

    def __init__(self, config: FillLabelsConfig):
        self.config = config

    def fit(self, data: BaseDataset) -> None:
        """ No need to fit anything for this preprocessing step. """
        pass

    def transform(self, data: DatasetOutputs) -> DatasetOutputs:
        """
        Fill missing values in the label Series according to the specified method.
            - If fill_method is 'constant', fill missing values with the specified fill_value.
            - If fill_method is 'ffill', fill missing values using forward fill then backward fill.
            - If fill_method is 'bfill', fill missing values using backward fill then forward fill.
            - If fill_method is 'nearest', fill missing values using nearest interpolation then bfill and ffill to handle any remaining NaNs.
         """ 
        
        if data.label is None:
            return data

        if self.config.fill_method == "constant":
            data.label = data.label.fillna(self.config.fill_value)
        elif self.config.fill_method == "ffill":
            data.label = data.label.ffill().bfill() # Fill missing values using forward fill then backward fill
        elif self.config.fill_method == "bfill":
            data.label = data.label.bfill().ffill() # Fill missing values using backward fill then forward fill
        elif self.config.fill_method == "nearest":
            # Fill missing values using nearest interpolation then bfill and ffill to handle any remaining NaNs
            data.label = data.label.interpolate(method='nearest').bfill().ffill()

        return data
