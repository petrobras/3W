import pandas as pd
import torch
import numpy as np
from pydantic import field_validator

from ..core.base_feature_extractor import BaseFeatureExtractor, FeatureExtractorConfig
from ..preprocessing._data_processing import windowing


class EWStatisticalConfig(FeatureExtractorConfig):
    """
    Configuration for the Exponentially Weighted
    Statistical feature extractor.
    """

    window_size: int = 100
    decay: float
    overlap: float = 0.0
    offset: int = 0
    eps: float = 1e-6

    @field_validator("overlap")
    def check_overlap_range(cls, v):
        """Validates that overlap is in the [0, 1) range."""
        if not 0 <= v < 1:
            raise ValueError("Overlap must be in the range [0, 1)")
        return v

    @field_validator("offset")
    def check_offset_value(cls, v):
        """Validates that offset is not negative."""
        if v < 0:
            raise ValueError("Offset must be a non-negative integer.")
        return v

    @field_validator("eps")
    def check_eps_value(cls, v):
        """Validates that epsilon is a small, positive number."""
        if v <= 0:
            raise ValueError("Epsilon (eps) must be positive.")
        return v


class ExtractEWStatisticalFeatures(BaseFeatureExtractor):
    """
    Configuration for the Exponetially Weighted
    Statistical feature extractor.
    """

    FEATURES = [
        "ew_mean",
        "ew_std",
        "ew_skew",
        "ew_kurt",
        "ew_min",
        "ew_1qrt",
        "ew_med",
        "ew_3qrt",
        "ew_max",
    ]

    def __init__(self, config: EWStatisticalConfig):
        super().__init__(config)
        self.window_size = config.window_size
        self.overlap = config.overlap
        self.offset = config.offset
        self.eps = config.eps
        h = config.decay ** torch.arange(
            self.window_size, 0, step=-1, dtype=torch.double
        )
        self.weights = h / (h.abs().sum() + self.eps)

    def _apply_weights(self, X, dim=-1):
        """Take the exponentially weighted average through a dot product."""
        return torch.tensordot(X, self.weights, dims=[[dim], [0]])

    def __call__(self, tags: pd.DataFrame, y: pd.Series | None = None, event_type=None):
        # Raise an error if y is not provided.
        if y is None:
            raise ValueError(
                "The 'y' series (labels) must be provided for feature extraction."
            )

        original_index_name = tags.index.name

        if self.offset > 0:
            tags = tags.iloc[self.offset :]
            y = y.iloc[self.offset :]

        # This list will track columns that actually produced features.
        processed_columns = []
        all_column_features = []

        for col_name in tags.columns:
            series = tags[col_name]

            # Skipping column with insufficient data
            if len(series) < self.window_size:
                continue

            windows_df = windowing(
                X=series,
                window="boxcar",
                window_size=self.window_size,
                overlap=self.overlap,
            )

            if windows_df.empty:
                continue

            # If we get here, the column was successfully processed.
            processed_columns.append(col_name)

            # Handle the extra column returned by the windowing function.
            if windows_df.shape[1] > self.window_size:
                windows_df = windows_df.iloc[:, : self.window_size]

            windows_tensor = torch.tensor(windows_df.values).double()

            mean = self._apply_weights(windows_tensor, dim=-1)
            std = self._apply_weights(
                torch.pow(windows_tensor - mean.unsqueeze(-1), 2)
            ).sqrt()
            cstags = (windows_tensor - mean.unsqueeze(-1)) / (
                std.unsqueeze(-1) + self.eps
            )
            skew = self._apply_weights(cstags.pow(3), dim=-1)
            kurt = self._apply_weights(cstags.pow(4), dim=-1)
            quantiles = torch.tensor([0.00, 0.25, 0.50, 0.75, 1.00]).double()
            q = cstags.quantile(quantiles, dim=-1)

            records = {
                f"{col_name}_{feat}": val
                for feat, val in zip(
                    self.FEATURES, [mean, std, skew, kurt, q[0], q[1], q[2], q[3], q[4]]
                )
            }
            all_column_features.append(pd.DataFrame(records))

        if not all_column_features:
            out_columns = [f"{t}_{f}" for f in self.FEATURES for t in tags.columns]
            empty_index = pd.Index([], name=original_index_name)
            X = pd.DataFrame(columns=out_columns, dtype=np.float64, index=empty_index)
            y_out = pd.Series([], dtype=y.dtype, index=empty_index)
            return X, y_out

        # Build the final column list ONLY from the processed columns.
        out_columns_final = [
            f"{t}_{f}" for f in self.FEATURES for t in processed_columns
        ]

        stride = int(self.window_size * (1 - self.overlap))
        if stride == 0:
            stride = 1

        output_index = tags.index[self.window_size - 1 :: stride]
        num_windows = len(all_column_features[0])
        output_index = output_index[:num_windows]

        X = pd.concat(all_column_features, axis=1)
        X.index = output_index
        X.index.name = original_index_name
        X = X.reindex(columns=out_columns_final)

        # Align y with the new windowed index
        y_out = y.loc[output_index]

        return X, y_out
