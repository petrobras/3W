import numpy as np
import pandas as pd
import torch
from pydantic import field_validator

from ..core.base_feature_extractor import BaseFeatureExtractor, FeatureExtractorConfig
from ..preprocessing._data_processing import windowing


class StatisticalConfig(FeatureExtractorConfig):
    """Configuration now uses overlap to match the windowing function."""

    window_size: int = 100
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


class ExtractStatisticalFeatures(BaseFeatureExtractor):
    """
    Final, definitively corrected version. This implementation correctly handles
    scenarios where some columns produce features and others do not.
    """

    FEATURES = ["mean", "std", "skew", "kurt", "min", "1qrt", "med", "3qrt", "max"]

    def __init__(self, config: StatisticalConfig):
        super().__init__(config)
        self.window_size = config.window_size
        self.overlap = config.overlap
        self.offset = config.offset
        self.eps = config.eps

    def __call__(self, tags: pd.DataFrame, event_type=None) -> pd.DataFrame:
        original_index_name = tags.index.name

        if self.offset > 0:
            tags = tags.iloc[self.offset :]

        # This list will track columns that actually produce features.
        processed_columns = []
        all_column_features = []

        for col_name in tags.columns:
            # Skipping column with insufficient data
            if len(tags[col_name]) < self.window_size:
                continue

            windows_df = windowing(
                X=tags[col_name],
                window="boxcar",
                window_size=self.window_size,
                overlap=self.overlap,
            )

            if windows_df.empty:
                continue

            # This column was successfully processed, so it is added to the list.
            processed_columns.append(col_name)

            if windows_df.shape[1] > self.window_size:
                windows_df = windows_df.iloc[:, : self.window_size]

            windows_tensor = torch.Tensor(windows_df.values).double()

            std, mean = torch.std_mean(windows_tensor, dim=-1, unbiased=False)
            cstags = (windows_tensor - mean.unsqueeze(-1)) / (
                std.unsqueeze(-1) + self.eps
            )
            skew = cstags.pow(3).mean(dim=-1)
            kurt = cstags.pow(4).mean(dim=-1)
            quantiles_tensor = torch.tensor([0.00, 0.25, 0.50, 0.75, 1.00]).double()
            q = windows_tensor.quantile(quantiles_tensor, dim=-1)

            records = {
                f"{col_name}_mean": mean,
                f"{col_name}_std": std,
                f"{col_name}_skew": skew,
                f"{col_name}_kurt": kurt,
                f"{col_name}_min": q[0],
                f"{col_name}_1qrt": q[1],
                f"{col_name}_med": q[2],
                f"{col_name}_3qrt": q[3],
                f"{col_name}_max": q[4],
            }
            all_column_features.append(pd.DataFrame(records))

        # If no columns were successfully processed, return an empty DataFrame.
        if not all_column_features:
            # Defining out_columns here for the empty case, using the original columns
            out_columns = [f"{t}_{f}" for f in self.FEATURES for t in tags.columns]
            empty_index = pd.Index([], name=original_index_name)
            return pd.DataFrame(
                columns=out_columns, dtype=np.float64, index=empty_index
            )

        # Build the final column list ONLY from processed columns.
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

        # Now, reindex will only use columns that should actually exist.
        return final_df.reindex(columns=out_columns_final)
