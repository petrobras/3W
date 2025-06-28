from typing import Tuple, Optional, Union
import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split as sklearn_train_test_split

from ..core.base_time_series_holdout import BaseTimeSeriesHoldout, TimeSeriesHoldoutConfig


class TimeSeriesHoldout(BaseTimeSeriesHoldout):
    def __init__(self, data: pd.DataFrame, pip_config: dict):
        config = TimeSeriesHoldoutConfig(**pip_config)
        super().__init__(config=config)
        self.data = data
        self.pip_config = pip_config

        """
        Initializes the TimeSeriesHoldout object with dataset and configuration.

        This class is designed to handle holdout-based train/test splits for time series data,
        with optional support for shuffling and stratification when appropriate. It wraps around
        a `BaseTimeSeriesHoldout` and passes a structured configuration using the provided `pip_config`.

        Args:
            data (pd.DataFrame): The input dataset to be used for splitting.
            pip_config (dict): A dictionary containing configuration parameters, such as:
                - 'test_size' (float or int): Proportion or count of test samples.
                - 'train_size' (float or int): Proportion or count of train samples.
                - 'random_state' (int, optional): Seed for reproducibility.
                - 'shuffle' (bool): Whether to shuffle the data before splitting.
                - 'stratify' (Series or DataFrame, optional): Class labels for stratification.
        """

    def train_test_split(
        self,
        *arrays: Union[Series, DataFrame],
        test_size: Optional[float] = None,
        train_size: Optional[float] = None,
        random_state: Optional[int] = None,
        shuffle: Optional[bool] = None,
        stratify: Optional[Union[Series, DataFrame]] = None
    ) -> Tuple:
        
        """
        Splits input arrays or matrices into train and test subsets.
        This method supports time series splitting with optional shuffling and stratification.

        Args:
            *arrays (Series or DataFrame): Input data to split. If none provided, uses `self.data`.
            test_size (float, optional): Proportion or absolute number of test samples. Must be <= 1 if float.
            train_size (float, optional): Proportion or absolute number of train samples. Must be <= 1 if float.
            random_state (int, optional): Controls the shuffling applied before the split.
            shuffle (bool): Whether to shuffle the data before splitting. Defaults to False.
            stratify (Series or DataFrame, optional): If not None, data is split in a stratified fashion, using this as class labels.

        Returns:
            Tuple: Splitted arrays in the same format as `sklearn.model_selection.train_test_split`.
        """

        if len(arrays) == 0:
            arrays = (self.data,)

        if not all(isinstance(arr, (Series, DataFrame)) for arr in arrays):
            raise TypeError("All inputs must be pandas Series or DataFrame.")

        test_size = test_size if test_size is not None else self.config.test_size
        train_size = train_size if train_size is not None else self.config.train_size
        random_state = random_state if random_state is not None else self.config.random_state
        shuffle = shuffle if shuffle is not None else self.config.shuffle
        stratify = stratify if stratify is not None else self.config.stratify

        if test_size is not None and train_size is None:
            if test_size > 1:
                raise ValueError("The test_size must be <= 1 for time series split.")

        if train_size is not None and test_size is None:
            if train_size > 1:
                raise ValueError("The train_size must be <= 1 for time series split.")

        if test_size is not None and train_size is not None:
            if test_size + train_size > 1:
                raise ValueError("The sum of train_size and test_size must be <= 1 for time series split.")
            
        if stratify is not None and not shuffle:
            raise ValueError("Stratified splitting requires shuffle=True.")

        try:
            return sklearn_train_test_split(
                *arrays,
                test_size=test_size,
                train_size=train_size,
                shuffle=shuffle,
                random_state=random_state,
                stratify=stratify
            )
        except Exception as e:
            raise RuntimeError(f"Failed to split time series data: {e}")
