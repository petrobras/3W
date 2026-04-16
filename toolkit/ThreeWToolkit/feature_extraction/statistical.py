from typing import cast

import numpy as np
import pandas as pd
import warnings

from scipy import stats
from ..core.base_feature_extractor import (
    BaseFeatureExtractor,
    BaseFeatureExtractorConfig,
)
from pydantic import Field, field_validator, PrivateAttr

from ..core.dataset_outputs import DatasetOutputs

_STATISTICAL_FEATURES = {
    "mean": lambda x: np.mean(x, axis=1),
    # moments
    "std": lambda x: np.std(x, axis=1, ddof=0),
    "var": lambda x: np.var(x, axis=1, ddof=0),
    "skew": lambda x: stats.moment(
        x, axis=1, order=3
    ),  # dont use stats.skew to avoid NaNs.
    "kurt": lambda x: stats.moment(
        x, axis=1, order=4
    ),  # use moment instead which returns 0 for constant values
    # quantiles
    "min": lambda x: np.min(x, axis=1),
    "q25": lambda x: np.percentile(x, 25, axis=1),
    "median": lambda x: np.median(x, axis=1),
    "q75": lambda x: np.percentile(x, 75, axis=1),
    "max": lambda x: np.max(x, axis=1),
}


class StatisticalConfig(
    BaseFeatureExtractorConfig,
):
    """Configuration for Statistical feature extractor."""

    features: list[str] = Field(
        default_factory=lambda: list(_STATISTICAL_FEATURES.keys()),
        description="List of statistical\
            features to compute. Available features: "
        + ", ".join(_STATISTICAL_FEATURES.keys()),
    )

    _target: type = PrivateAttr(default_factory=lambda: StatisticalFeatures)

    @field_validator("features")
    @classmethod
    def validate_features(cls, features: list[str]) -> list[str]:
        """Validates that selected features are available."""
        invalid_features = set(features) - set(_STATISTICAL_FEATURES)
        if invalid_features:
            raise ValueError(f"Invalid features: {invalid_features}.")
        return features


class StatisticalFeatures(BaseFeatureExtractor):
    """
    Extracts statistical features from windowed time series data.
    Input: DataFrame (windowed, each row is a window)
    Output: DataFrame (features per window)
    """

    def __init__(self, config: StatisticalConfig):
        """Initializes the StatisticalFeatures extractor with the given configuration.

        Args:
            config: StatisticalConfig object containing the list of features to compute.
        """
        self.config: StatisticalConfig = config

    def transform(self, data: DatasetOutputs) -> DatasetOutputs:
        """Compute selected statistical features for each window in the input data.

        Args:
            data: Windowed dataset outputs with multi-index (window, variable).

        Returns:
            DatasetOutputs with extracted statistical features.
        """
        if data.signal.index.names != ["window", "variable"]:
            raise ValueError("StatisticalFeatures must operate on windowed data.")

        values = data.signal.values

        # compute statistics over the rows (windows) for each column independently
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", category=RuntimeWarning
            )  # ignore warnings from constant values
            features = {
                s: _STATISTICAL_FEATURES[s](values) for s in self.config.features
            }

        # assemble multiindex DataFrame with features as cols
        signal_df = pd.DataFrame(features, index=data.signal.index)
        # unstack variable to get per-variable features in columns
        signal_df = cast(pd.DataFrame, signal_df.unstack("variable"))  # safe cast
        # flatten multiindex columns
        signal_df.columns = ["_".join(col).strip() for col in signal_df.columns]

        return DatasetOutputs(
            signal=signal_df,  # type: ignore
            label=data.label,
            metadata=data.metadata.copy(),
        )
