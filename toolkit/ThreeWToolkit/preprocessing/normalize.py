import numpy as np
import pandas as pd
from typing import Literal
from pydantic import Field, field_validator
from ..core.base_dataset import BaseDataset
from ..core.base_preprocessing import BasePreprocessing, BasePreprocessingConfig
from ..core.dataset_outputs import DatasetOutputs


class NormalizeConfig(BasePreprocessingConfig):
    norm: Literal["l1", "l2", "max"] | float = Field(default="l2", description="Normalization method to apply. Can be 'l1', 'l2', 'max' for standard normalization, or a generic norm")
    target_: type = Field(default_factory=lambda: Normalize)

    @field_validator("norm")
    def validate_norm(cls, value):
        if isinstance(value, str) and value not in {"l1", "l2", "max"}:
            raise ValueError("norm must be 'l1', 'l2', 'max' or a float value.")
        if isinstance(value, (int, float)) and value <= 0:
            raise ValueError("If norm is a numeric value, it must be greater than 0.")
        return value


class Normalize(BasePreprocessing):
    """
    A data processing step that normalizes signal data using z-score normalization.

    Collects statistics (mean and std) from signal columns across events during training,
    then applies normalization to standardize the signals.

    Attributes:
        config (NormalizeConfig): Configuration object containing normalization parameters
        collected (dict): Accumulated statistics for each signal column
        statistics (dict): Computed mean and std for each signal column
    """

    def __init__(self, config: NormalizeConfig,):
        """
        Initialize the Normalize step with the provided configuration.

        Args:
            config (NormalizeConfig): Configuration containing norm type, axis, and other parameters
        """
        self.config = config

        if self.config.norm == "l1":
            self.norm = 1
        elif self.config.norm == "l2":
            self.norm = 2
        elif self.config.norm == "max":
            self.norm = np.inf
        else:
            self.norm = self.config.norm

        self.global_average = None
        self.global_moment  = None

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

    def _compute_global_moments(self, data: BaseDataset) -> None:
        moments = []
        counts  = []
        for event in data:
            moments.append((event.signal - self.global_average).abs().pow(self.norm).mean())
            counts.append(event.signal.count())
        # compute weighted average of the moments
        moments = pd.concat(moments, axis=1).transpose()
        counts = pd.concat(counts, axis=1).transpose()

        self.global_moment = (moments * counts).sum() / counts.sum()
        self.global_moment = self.global_moment.pow(1 / self.norm)

    def _compute_global_max(self, data: BaseDataset) -> None:
        maxes = []
        for event in data:
            maxes.append((event.signal - self.global_average).abs().max())
        # compute global max across all events
        maxes = pd.concat(maxes, axis=1).transpose()
        self.global_moment = maxes.max()

    def fit(self, data: BaseDataset) -> None:
        """
        Collect statistics from a single event for aggregation.

        Args:
            data: DatasetOutputs object containing signal DataFrame
        """

        self._compute_global_average(data)
        if self.config.norm == "max":
            self._compute_global_max(data)
        else:
            self._compute_global_moments(data)

    def transform(self, data: DatasetOutputs) -> DatasetOutputs:
        """
        Apply normalization to the 'signal' data using computed statistics.

        Performs Lp normalization: (x - mean) / k on signal columns.

        Args:
            data: DatasetOutputs object containing signal DataFrame

        Returns:
            DatasetOutputs: Transformed data with normalized signal DataFrame
        """
        if self.global_average is None or self.global_moment is None:
            raise ValueError("Normalize: fit must be called before transform to compute statistics.")

        data.signal = (data.signal - self.global_average) / self.global_moment

        return data
