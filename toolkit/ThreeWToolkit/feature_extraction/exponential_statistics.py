from typing import cast
import numpy as np
import pandas as pd
from ..core.dataset_outputs import DatasetOutputs
from ..core.base_feature_extractor import (
    BaseFeatureExtractor,
    BaseFeatureExtractorConfig,
)
from pydantic import Field, field_validator, PrivateAttr

_AVAILABLE_FEATURES = {
    "ew_mean",
    "ew_std",
    "ew_skew",
    "ew_kurt",
    "ew_min",
    "ew_1qrt",
    "ew_med",
    "ew_3qrt",
    "ew_max",
}


class EWStatisticalConfig(
    BaseFeatureExtractorConfig,
):
    """Configuration for the Exponentially Weighted Statistical feature extractor."""

    decay: float = Field(
        default=0.95,
        gt=0,
        le=1,
        description="Decay factor for exponential weighting. Must be in the range (0, 1].\
                    Values closer to 0 give more weight to recent observations, while values closer to 1 give more\
                    weight to older observations.",
    )
    window_size: int = Field(
        default=128,
        gt=0,
        description="Size of the window for calculating exponentially weighted statistics. Must be a positive integer.",
    )

    features: list[str] = Field(
        default_factory=lambda: list(_AVAILABLE_FEATURES),
        description="List of exponentially weighted statistical features to compute. Available features: "
        + ", ".join(_AVAILABLE_FEATURES),
    )

    _target: type = PrivateAttr(default_factory=lambda: EWStatisticalFeatures)

    @field_validator("decay")
    @classmethod
    def check_decay_range(cls, decay):
        """Validates that decay is in the (0, 1] range."""
        if not 0 < decay <= 1:
            raise ValueError("Decay must be in the range (0, 1]")
        return decay

    @field_validator("features")
    @classmethod
    def validate_features(cls, features):
        """Validates that selected features are available."""
        if features is not None:
            invalid_features = set(features) - _AVAILABLE_FEATURES
            if invalid_features:
                raise ValueError(
                    f"Invalid features: {invalid_features}. "
                    f"Available features: {_AVAILABLE_FEATURES}"
                )
        return features


class EWStatisticalFeatures(BaseFeatureExtractor):
    """
    Extracts exponentially weighted statistical features from windowed time series data.

    Applies exponential decay weights to calculate weighted statistics, giving more
    importance to recent observations within each window.

    Exponentially weighted statistics can be thought as taking the expectation of the data under an exponentially decaying
    weighting scheme.

    For the moments:
    ew_mean = E_w[x] = sum_i w_i * x_i
    ew_std = sqrt(E_w[(x - ew_mean)^2])
    ew_skew = E_w[(x - ew_mean)^3]
    ew_kurt = E_w[(x - ew_mean)^4]

    For the quantiles, we just compute them on standardized data (x - ew_mean) / ew_std for better interpretability.
    """

    def __init__(self, config: EWStatisticalConfig):
        """
        Initialize the exponentially weighted statistical feature extractor.

        Args:
            config: Configuration object with exponential weighting parameters
        """
        self.config: EWStatisticalConfig = config

        # Create exponential decay weights (recent values have higher weights)
        h = self.config.decay ** np.arange(
            self.config.window_size, 0, step=-1, dtype=np.float64
        )
        # Normalize weights so they sum to 1
        self.weights = h / np.abs(h).sum()

    def _ew_expectation(self, x):
        """Calculate the exponentially weighted expectation (mean) of x."""
        return np.einsum(
            "ij,j->i", x, self.weights
        )  # dot product of each row of x with weights

    def transform(self, data: DatasetOutputs) -> DatasetOutputs:
        """
        Apply exponentially weighted statistical feature extraction to the input data.

        Args:
            data: DatasetOutputs with signal and label data

        Returns:
            DatasetOutputs with extracted features and labels
        """
        if data.signal.index.names != ["window", "variable"]:
            raise ValueError("EWStatisticalFeatures must operate on windowed data.")

        if data.signal.shape[1] != self.config.window_size:
            raise ValueError(
                f"Input data window size {data.signal.shape[1]} does not match configured window size {self.config.window_size}."
            )

        values = data.signal.values
        features = {}

        # Calculate exponentially weighted moments
        mean = self._ew_expectation(values)  # we will need this first
        if "ew_mean" in self.config.features:
            features["ew_mean"] = mean

        centered = (
            values - mean[:, None]
        )  # center the data for std, skew, kurt calculations
        std = np.sqrt(self._ew_expectation(centered**2))

        if "ew_std" in self.config.features:
            features["ew_std"] = std

        if "ew_skew" in self.config.features:
            features["ew_skew"] = self._ew_expectation(centered**3)

        if "ew_kurt" in self.config.features:
            features["ew_kurt"] = self._ew_expectation(centered**4)

        # Calculate exponentially weighted quantiles
        if any(
            f in self.config.features
            for f in ["ew_min", "ew_1qrt", "ew_med", "ew_3qrt", "ew_max"]
        ):
            # quantiles requested, so we compute the quantiles on the ew-shifted data.
            standardized = centered / (std[:, None] + 1e-8)

            quantiles = np.quantile(
                standardized, [0, 0.25, 0.5, 0.75, 1], axis=1
            )  # calculate all at once
            if "ew_min" in self.config.features:
                features["ew_min"] = quantiles[0]
            if "ew_1qrt" in self.config.features:
                features["ew_1qrt"] = quantiles[1]
            if "ew_med" in self.config.features:
                features["ew_med"] = quantiles[2]
            if "ew_3qrt" in self.config.features:
                features["ew_3qrt"] = quantiles[3]
            if "ew_max" in self.config.features:
                features["ew_max"] = quantiles[4]

        # assemble multiindex DataFrame with features as cols
        signal_df = pd.DataFrame(features, index=data.signal.index)
        # unstack variable to get per-variable features in columns
        signal_df = cast(pd.DataFrame, signal_df.unstack("variable"))  # safe cast
        # flatten multiindex columns
        signal_df.columns = ["_".join(col).strip() for col in signal_df.columns] 

        return DatasetOutputs(signal=signal_df, label=data.label, metadata=data.metadata)
