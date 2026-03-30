""" Preprocessing class for cleaning possibly frozen or out-of-range signals. """

from pydantic import Field

import numpy as np
import pandas as pd

from ..core.base_dataset import BaseDataset
from ..core.dataset_outputs import DatasetOutputs
from ..core.base_feature_extractor import  BaseFeatureExtractor, BaseFeatureExtractorConfig

from ..dataset.transformed_dataset import TransformedDataset

_3W_CATEGORICAL_FEATURES = [ # List of categorical features to exclude from cleaning by default
    "ESTADO-DHSV",
    "ESTADO-M1",
    "ESTADO-M2",
    "ESTADO-PXO",
    "ESTADO-SDV-GL",
    "ESTADO-SDV-P",
    "ESTADO-W1",
    "ESTADO-W2",
    "ESTADO-XO",
    "state"
]


class CleanSignalsConfig(BaseFeatureExtractorConfig):
    """
    Configuration for the CleanSignals feature extractor.
    
    This configuration includes thresholds for identifying frozen / out-of-range signals based on the interquartile
    range (IQR) of the average and standard deviation of the signals.

    Events whose statitics (standard deviation or average) fall outside the specified IQR thresholds may be considered faulty, and
    will be replaced with NaN values.

    Larger IQR thresholds will be more lenient, while smaller IQR thresholds will be more strict in identifying frozen /
    out-of-range signals.

    Categorical signals should not be processed by this feature extractor, as the IQR-based thresholds are designed for
    continuous signals.

    """
    average_iqr_threshold: float = Field(
        default=1.5,
        gt=0.0,
        description="Threshold for identifying frozen signals based on the interquartile range (IQR) of the average.\
                     Signals with average IQR below this threshold may be considered frozen.")

    std_iqr_threshold: float = Field(
        default=1.5,
        gt=0.0,
        description="Threshold for identifying frozen signals based on the interquartile range (IQR) of the standard\
        deviation. Signals with std IQR below this threshold may be considered frozen.")

    absolute_std_threshold: float | None = Field(
        default=1e-6,
        gt=0.0,
        description="Absolute threshold for identifying frozen signals based on standard deviation. Signals with std below\
                     this threshold may be considered frozen, regardless of the IQR-based thresholds. Setting this to\
                     None will disable the absolute std threshold.")

    missing_column_threshold: float = Field(
        default=0.6,
        description="Drop columns that are all-NaN in more than this fraction of events.\
                     This filtering occurs after the IQR-based thresholding.")


    exclude_features: list[str] = Field(
        default = _3W_CATEGORICAL_FEATURES,
        description="List of feature names to exclude from cleaning. These features will not be processed by the\
                     CleanSignals preprocessor, and will be left unchanged. By default, this includes known categorical\
                     features that should not be processed by the IQR-based thresholds.")

    target_: type = Field(default_factory=lambda: CleanSignals)

class CleanSignals(BaseFeatureExtractor):
    """
    Feature extractor for cleaning possibly frozen or out-of-range signals.
    """
    def __init__(self, config: CleanSignalsConfig):
        self.config = config

        self.average_bounds = None
        self.std_bounds = None

        self.drop_list = None

    def fit(self, data: BaseDataset) -> None:
        """
        Fit the feature extractor to the data.
        This method computes the necessary statistics from the input dataset to determine the safe ranges for the 
        signals.

        Args:
            data (DatasetOutputs): The input dataset outputs to fit on.
        """
        # Compute distribution of means and std along the dataset
        self._fit_iqr_thresholds(data)

        # apply cleaning and fit columns thresholding
        cleaned_data = TransformedDataset(data, self._filter_iqr_bounds)
        self._fit_missing_thresholds(cleaned_data)

    
    def _fit_iqr_thresholds(self, data: BaseDataset) -> None:
        averages  = []
        stds = []
        for event in data:
            averages.append(event.signal.mean())
            stds.append(event.signal.std())
        averages = pd.concat(averages, axis=1).transpose()
        stds = pd.concat(stds, axis=1).transpose()

        # compute quantiles
        average_quantiles = (averages.quantile(0.25), averages.quantile(0.75))
        average_iqr = average_quantiles[1] - average_quantiles[0]
        self.average_bounds = (average_quantiles[0] - self.config.average_iqr_threshold * average_iqr,
                                   average_quantiles[1] + self.config.average_iqr_threshold * average_iqr)

        std_quantiles = (stds.quantile(0.25), stds.quantile(0.75))
        std_iqr = std_quantiles[1] - std_quantiles[0]

        # take into account absolute std threshold when computing std bounds
        lower_std_bound = std_quantiles[0] - self.config.std_iqr_threshold * std_iqr
        if self.config.absolute_std_threshold is not None:
             lower_std_bound = lower_std_bound.clip(lower=self.config.absolute_std_threshold)
        self.std_bounds = (lower_std_bound, std_quantiles[1] + self.config.std_iqr_threshold * std_iqr)

    def _fit_missing_thresholds(self, data: BaseDataset) -> None:
        all_nans = []
        for event in data:
            all_nans.append(event.signal.isna().all())
        all_nans = pd.concat(all_nans, axis=1).transpose()
        
        all_nan_percentage = all_nans.mean()
        drop_cols = all_nan_percentage < self.config.missing_column_threshold
        self.drop_list = drop_cols[drop_cols].index.tolist()

    
    def _filter_iqr_bounds(self, data: DatasetOutputs) -> DatasetOutputs:
        if self.average_bounds is None or self.std_bounds is None:
            raise ValueError("The CleanSignals feature extractor must be fitted before calling transform.")

        signal = data.signal

        signal_average = signal.mean()
        signal_std = signal.std()

        # identify signals that are outside the IQR-based thresholds
        drop_average = (signal_average < self.average_bounds[0]) | (signal_average > self.average_bounds[1])
        drop_std = (signal_std < self.std_bounds[0]) | (signal_std > self.std_bounds[1])

        drop = drop_average | drop_std
        removed_columns = drop[drop].index.tolist()

        # filter out removed columns based on exclude_features list in config
        removed_columns = [col for col in removed_columns if col not in self.config.exclude_features]

        # replace out-of-range signals with NaN values
        signal = signal.assign(**{col: np.nan for col in removed_columns})

        return DatasetOutputs(signal=signal, label=data.label, metadata=data.metadata)

    def _filter_missing_cols(self, data: DatasetOutputs) -> DatasetOutputs:
        if self.drop_list is None:
            raise ValueError("The CleanSignals feature extractor must be fitted before calling transform.")

        removed_columns = [col for col in self.drop_list if col not in self.config.exclude_features]
        signal = data.signal.drop(columns=removed_columns)

        return DatasetOutputs(signal=signal, label=data.label, metadata=data.metadata)

    def transform(self, data: DatasetOutputs) -> DatasetOutputs:
        data = self._filter_iqr_bounds(data)
        data = self._filter_missing_cols(data)
        return data
