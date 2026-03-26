import pandas as pd
from pydantic import Field
from ..core.base_preprocessing import BasePreprocessing, BasePreprocessingConfig


class FillLabelsConfig(BasePreprocessingConfig):
    target_: type = Field(default_factory=lambda: FillLabels)


class FillLabels(BasePreprocessing):
    """
    Preprocessing step to fill missing values in label columns using bfill then ffill.
    Applies to each event's label DataFrame.
    """

    def __init__(self, config: FillLabelsConfig):
        self.config = config

    def fit(self, data: dict) -> None:
        pass

    def compute(self) -> None:
        pass

    def transform(self, data: dict) -> dict:
        label_df = data.get("label")
        if label_df is None or not isinstance(label_df, pd.DataFrame):
            return data

        data_copy = label_df.copy()
        if "class" not in data_copy.columns:
            return data

        # Fill missing values in the 'class' column using bfill then ffill
        data_copy["class"] = data_copy["class"].bfill().ffill()

        result = data.copy()
        result["label"] = data_copy
        return result
