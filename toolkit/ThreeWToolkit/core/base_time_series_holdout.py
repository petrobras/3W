from abc import ABC
from pydantic import BaseModel, ConfigDict
from typing import Optional, Union
import pandas as pd


class TimeSeriesHoldoutConfig(BaseModel):
    """
    Configuration model for time series holdout splitting.

    Attributes:
        test_size (Optional[float]): Proportion or count of test samples. Must be <= 1 if float.
        train_size (Optional[float]): Proportion or count of train samples. Must be <= 1 if float.
        random_state (Optional[int]): Seed for reproducibility.
        shuffle (bool): Whether to shuffle data before splitting. Defaults to False.
        stratify (Optional[Union[pd.Series, pd.DataFrame]]): Labels for stratification. Must be set only if shuffle is True.
    """

    test_size: Optional[float] = None
    train_size: Optional[float] = None
    random_state: Optional[int] = None
    shuffle: bool = False
    stratify: Optional[Union[pd.Series, pd.DataFrame]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseTimeSeriesHoldout(ABC):
    def __init__(self, config: TimeSeriesHoldoutConfig):
        """
        Abstract base class for time series holdout logic.

        Args:
            config (TimeSeriesHoldoutConfig): Configuration for the holdout split.
        """
        self.config = config
