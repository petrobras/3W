import pandas as pd
import torch
import numpy as np
from pydantic import field_validator

from ..core.base_feature_extractor import BaseFeatureExtractor, FeatureExtractorConfig
from ..preprocessing._data_processing import windowing


class EWStatisticalConfig(FeatureExtractorConfig):
    """Configuration for the EWMA Statistical feature extractor, using overlap."""

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


class ExtractEWStatisticalFeatures(BaseFeatureExtractor):
    """
    PyTorch implementation of an exponentially weighted statistical feature mapper,
    refactored to use the toolkit's `windowing` function.
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
        self.h = h / (h.abs().sum() + self.eps)

    def _E(self, X, dim=-1):
        """Take the exponentially weighted average through a dot product in the specified dimension."""
        return torch.tensordot(X, self.h, dims=[[dim], [0]])

    def __call__(self, tags: pd.DataFrame, event_type=None) -> pd.DataFrame:
        original_index_name = tags.index.name

        if self.offset > 0:
            tags = tags.iloc[self.offset :]

        # This list will track columns that actually produced features.
        processed_columns = []
        all_column_features = []

        for col_name in tags.columns:
            series = tags[col_name]

            # Skipping column with insufficient data
            if len(series) < self.window_size:
                continue

            # Using "boxcar" because the exponential weighting is handled separately by `_E`.
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

            mean = self._E(windows_tensor, dim=-1)
            std = self._E(torch.pow(windows_tensor - mean.unsqueeze(-1), 2)).sqrt()
            cstags = (windows_tensor - mean.unsqueeze(-1)) / (
                std.unsqueeze(-1) + self.eps
            )
            skew = self._E(cstags.pow(3), dim=-1)
            kurt = self._E(cstags.pow(4), dim=-1)
            quantiles = torch.tensor([0.00, 0.25, 0.50, 0.75, 1.00]).double()
            q = cstags.quantile(quantiles, dim=-1)

            records = {
                f"{col_name}_ew_mean": mean,
                f"{col_name}_ew_std": std,
                f"{col_name}_ew_skew": skew,
                f"{col_name}_ew_kurt": kurt,
                f"{col_name}_ew_min": q[0],
                f"{col_name}_ew_1qrt": q[1],
                f"{col_name}_ew_med": q[2],
                f"{col_name}_ew_3qrt": q[3],
                f"{col_name}_ew_max": q[4],
            }
            all_column_features.append(pd.DataFrame(records))

        if not all_column_features:
            out_columns = [f"{t}_{f}" for f in self.FEATURES for t in tags.columns]
            empty_index = pd.Index([], name=original_index_name)
            return pd.DataFrame(
                columns=out_columns, dtype=np.float64, index=empty_index
            )

        # Build the final column list ONLY from the processed columns.
        out_columns_final = [
            f"{t}_{f}" for f in self.FEATURES for t in processed_columns
        ]

        stride = int(self.window_size * (1 - self.overlap))

        output_index = tags.index[self.window_size - 1 :: stride]
        num_windows = len(all_column_features[0])
        output_index = output_index[:num_windows]

        final_df = pd.concat(all_column_features, axis=1)
        final_df.index = output_index
        final_df.index.name = original_index_name

        # Now, reindex will only organize the columns that should actually exist.
        return final_df.reindex(columns=out_columns_final)
