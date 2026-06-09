from pydantic import Field, PrivateAttr

import numpy as np
import pandas as pd

from ..core.base_dataset import BaseDataset
from ..core.dataset_outputs import DatasetOutputs
from ..core.base_preprocessing import (
    BasePreprocessing,
    BasePreprocessingConfig,
)

from ..dataset.transformed_dataset import TransformedDataset

_3W_CATEGORICAL_FEATURES = (
    [  # List of categorical features to exclude from cleaning by default
        "ESTADO-DHSV",
        "ESTADO-M1",
        "ESTADO-M2",
        "ESTADO-PXO",
        "ESTADO-SDV-GL",
        "ESTADO-SDV-P",
        "ESTADO-W1",
        "ESTADO-W2",
        "ESTADO-XO",
        "state",
    ]
)


class CleanSignalsConfig(BasePreprocessingConfig):
    """Configuration for identifying and cleaning frozen or out-of-range signals using IQR thresholds."""

    average_iqr_threshold: float = Field(
        default=3.0,
        gt=0.0,
        description="IQR threshold for average values. Signals below this may be considered frozen.",
    )

    std_iqr_threshold: float = Field(
        default=3.0,
        gt=0.0,
        description="IQR threshold for standard deviation. Signals below this may be considered frozen.",
    )

    absolute_std_threshold: float | None = Field(
        default=1e-6,
        gt=0.0,
        description="Absolute standard deviation threshold for frozen detection. Set to None to disable.",
    )

    missing_column_threshold: float = Field(
        default=0.6,
        description="Drop columns that are all-NaN in more than this fraction of events.",
    )

    exclude_features: list[str] = Field(
        default=_3W_CATEGORICAL_FEATURES,
        description="Feature names to exclude from cleaning. Categorical features left unchanged.",
    )

    _target: type = PrivateAttr(default_factory=lambda: CleanSignals)


class CleanSignals(BasePreprocessing):
    """
    Feature extractor for cleaning possibly frozen or out-of-range signals.
    """

    def __init__(self, config: CleanSignalsConfig):
        """
        Initializes the CleanSignals feature extractor with the given configuration.

        Args:
            config: CleanSignalsConfig object containing the IQR thresholds and other parameters for cleaning.
        """
        self.config: CleanSignalsConfig = config

        self.average_bounds: tuple[pd.Series, pd.Series] | None = None
        self.std_bounds: tuple[pd.Series, pd.Series] | None = None

        self.drop_list: list[str] = []

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
        """Compute the IQR-based thresholds for average and std of the signals based on the training data.
        This method computes the average and standard deviation for each signal across all events in the dataset,
        and then determines the IQR-based thresholds for identifying out-of-range signals.

        Args:
            data (BaseDataset): The input dataset to compute the thresholds from.
        """
        _averages = []
        _stds = []
        for event in data:
            _averages.append(event.signal.mean())
            _stds.append(event.signal.std())
        averages = pd.concat(_averages, axis=1).transpose()
        stds = pd.concat(_stds, axis=1).transpose()

        # compute quantiles
        average_quantiles = (averages.quantile(0.25), averages.quantile(0.75))
        average_iqr = average_quantiles[1] - average_quantiles[0]
        self.average_bounds = (
            average_quantiles[0] - self.config.average_iqr_threshold * average_iqr,
            average_quantiles[1] + self.config.average_iqr_threshold * average_iqr,
        )

        std_quantiles = (stds.quantile(0.25), stds.quantile(0.75))
        std_iqr = std_quantiles[1] - std_quantiles[0]

        # take into account absolute std threshold when computing std bounds
        lower_std_bound = std_quantiles[0] - self.config.std_iqr_threshold * std_iqr
        if self.config.absolute_std_threshold is not None:
            lower_std_bound = lower_std_bound.clip(
                lower=self.config.absolute_std_threshold
            )
        self.std_bounds = (
            lower_std_bound,
            std_quantiles[1] + self.config.std_iqr_threshold * std_iqr,
        )

    def _fit_missing_thresholds(self, data: BaseDataset) -> None:
        """
        Compute the list of columns to drop based on the fraction of all-NaN values across events in the dataset.

        Args:
            data (BaseDataset): The input dataset to compute the missing column thresholds from.
        """
        _all_nans = []
        for event in data:
            _all_nans.append(event.signal.isna().all())
        all_nans = pd.concat(_all_nans, axis=1).transpose()

        all_nan_percentage = all_nans.mean()
        drop_cols = all_nan_percentage >= self.config.missing_column_threshold
        self.drop_list = drop_cols[drop_cols].index.tolist()

    def _filter_iqr_bounds(self, data: DatasetOutputs) -> DatasetOutputs:
        """
        Filter out signals that are outside the IQR-based thresholds by replacing them with NaN values.

        Args:
            data (DatasetOutputs): The input dataset outputs to filter.
        Returns:
            DatasetOutputs with out-of-range signals replaced by NaN values."""
        if self.average_bounds is None or self.std_bounds is None:
            raise ValueError(
                "The CleanSignals feature extractor must be fitted before calling transform."
            )

        signal = data.signal

        signal_average = signal.mean()
        signal_std = signal.std()

        # identify signals that are outside the IQR-based thresholds
        drop_average = (signal_average < self.average_bounds[0]) | (
            signal_average > self.average_bounds[1]
        )
        drop_std = (signal_std < self.std_bounds[0]) | (signal_std > self.std_bounds[1])

        drop = drop_average | drop_std
        removed_columns = drop[drop].index.tolist()

        # filter out removed columns based on exclude_features list in config
        removed_columns = [
            col for col in removed_columns if col not in self.config.exclude_features
        ]

        # replace out-of-range signals with NaN values
        signal = signal.assign(**{col: np.nan for col in removed_columns})

        return DatasetOutputs(signal=signal, label=data.label, metadata=data.metadata)

    def _filter_missing_cols(self, data: DatasetOutputs) -> DatasetOutputs:
        """Filter out columns that are all-NaN in more than the specified fraction of events by dropping them from the signal.

        Args:
            data (DatasetOutputs): The input dataset outputs to filter.
        Returns: DatasetOutputs with columns dropped according to the missing column threshold.
        """
        if self.drop_list is None:
            raise RuntimeError(
                "The CleanSignals feature extractor must be fitted before calling transform."
            )

        removed_columns = [
            col for col in self.drop_list if col not in self.config.exclude_features
        ]
        signal = data.signal.drop(columns=removed_columns)

        return DatasetOutputs(signal=signal, label=data.label, metadata=data.metadata)

    def transform(self, data: DatasetOutputs) -> DatasetOutputs:
        """Apply the cleaning transformations to the input dataset outputs.
        Args:
            data (DatasetOutputs): The input dataset outputs to transform.
        Returns: DatasetOutputs with cleaned signals according to the fitted thresholds.
        """
        data = self._filter_iqr_bounds(data)
        data = self._filter_missing_cols(data)
        return data
