import numpy as np
import pandas as pd

from scipy import stats
from ..core.base_feature_extractor import (
    OverlapOffsetMixin,
    EpsMixin,
    WindowSizeMixin,
    FeatureSelectionMixin,
    BaseFeatureExtractor,
    BaseFeatureExtractorConfig,
)
from pydantic import Field, field_validator
from typing import ClassVar


class StatisticalConfig(
    OverlapOffsetMixin,
    EpsMixin,
    WindowSizeMixin,
    FeatureSelectionMixin,
    BaseFeatureExtractorConfig,
):
    """Configuration for Statistical feature extractor."""

    target_: type = Field(default_factory=lambda: StatisticalFeatures)

    AVAILABLE_FEATURES: ClassVar[list[str]] = [
        "mean",
        "std",
        "var",
        "min",
        "max",
        "median",
        "skew",
        "kurt",
        "q25",
        "q75",
        "range",
        "iqr",
    ]

    @field_validator("selected_features")
    @classmethod
    def validate_selected_features(cls, v):
        """Validates that selected features are available."""
        if v is not None:
            invalid_features = set(v) - set(cls.AVAILABLE_FEATURES)
            if invalid_features:
                raise ValueError(
                    f"Invalid features: {invalid_features}. "
                    f"Available features: {cls.AVAILABLE_FEATURES}"
                )
        return v


class StatisticalFeatures(BaseFeatureExtractor):
    """
    Extracts statistical features from windowed time series data.
    Input: DataFrame (windowed, each row is a window)
    Output: DataFrame (features per window)
    """

    FEATURES = [
        "mean",
        "std",
        "var",
        "min",
        "max",
        "median",
        "skew",
        "kurt",
        "q25",
        "q75",
        "range",
        "iqr",
    ]

    def __init__(self, config: StatisticalConfig):
        self.config = config
        self.selected_features = config.selected_features or self.FEATURES
        self.label_column = getattr(config, "label_column", None)
        self.offset = getattr(config, "offset", 0)
        self.eps = getattr(config, "eps", 1e-8)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract statistical features from windowed DataFrame.
        Args:
            data: DataFrame with windowed data (columns: varX_tY, label optional)
        Returns:
            DataFrame with extracted features (one row per window)
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        if data.empty:
            raise ValueError("Input data is empty")

        # Apply offset if needed
        if self.offset > 0:
            if self.offset >= len(data):
                raise ValueError(
                    f"Offset ({self.offset}) is larger than data length ({len(data)})"
                )
            data = data.iloc[self.offset :].copy()

        # Identify signal columns (exclude label)
        columns = data.columns.tolist()
        label_col = self.label_column or (
            columns[-1] if columns[-1].lower() == "class" else None
        )
        signal_cols = [col for col in columns if col != label_col]
        print(signal_cols)

        # Compute features for each signal column independently
        features = {}
        for col in signal_cols:
            arr = data[[col]].to_numpy()
            arr = arr.reshape(-1, 1) if arr.ndim == 1 else arr
            if "mean" in self.selected_features:
                features[f"{col}_mean"] = np.mean(arr, axis=1)
            if "std" in self.selected_features:
                features[f"{col}_std"] = np.std(arr, axis=1, ddof=0)
            if "var" in self.selected_features:
                features[f"{col}_var"] = np.var(arr, axis=1, ddof=0)
            if "min" in self.selected_features:
                features[f"{col}_min"] = np.min(arr, axis=1)
            if "max" in self.selected_features:
                features[f"{col}_max"] = np.max(arr, axis=1)
            if "median" in self.selected_features:
                features[f"{col}_median"] = np.median(arr, axis=1)
            if "skew" in self.selected_features:
                stds = np.std(arr, axis=1, ddof=0)
                mask = stds > self.eps
                skew = np.zeros(arr.shape[0])
                if np.any(mask):
                    skew[mask] = stats.skew(arr[mask], axis=1)
                features[f"{col}_skew"] = skew
            if "kurt" in self.selected_features:
                stds = np.std(arr, axis=1, ddof=0)
                mask = stds > self.eps
                kurt = np.zeros(arr.shape[0])
                if np.any(mask):
                    kurt[mask] = stats.kurtosis(arr[mask], axis=1)
                features[f"{col}_kurt"] = kurt
            if "q25" in self.selected_features:
                features[f"{col}_q25"] = np.percentile(arr, 25, axis=1)
            if "q75" in self.selected_features:
                features[f"{col}_q75"] = np.percentile(arr, 75, axis=1)
            if "range" in self.selected_features:
                features[f"{col}_range"] = np.ptp(arr, axis=1)
            if "iqr" in self.selected_features:
                q75 = np.percentile(arr, 75, axis=1)
                q25 = np.percentile(arr, 25, axis=1)
                features[f"{col}_iqr"] = q75 - q25

        # Build output DataFrame
        out_df = pd.DataFrame(features)
        # Add label if present
        if label_col and label_col in data.columns:
            out_df[label_col] = data[label_col].values

        print(f"Extracted features: {out_df.columns.tolist()}")

        return out_df
