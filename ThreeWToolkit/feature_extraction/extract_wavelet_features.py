import pandas as pd
import torch
import numpy as np
import pywt
from typing import Optional
from pydantic import field_validator

from ..core.base_feature_extractor import BaseFeatureExtractor, FeatureExtractorConfig
from ..preprocessing._data_processing import windowing


class WaveletConfig(FeatureExtractorConfig):
    """Configuration for the Wavelet feature extractor, using overlap."""

    level: int = 1
    overlap: float = 0.0
    offset: int = 0

    @field_validator("level")
    def check_level_is_positive(cls, v):
        """Validates that the wavelet level is a positive integer."""
        if v < 1:
            raise ValueError("Wavelet level must be a positive integer (>= 1).")
        return v

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


class ExtractWaveletFeatures(BaseFeatureExtractor):
    """
    PyTorch implementation of the wavelet feature mapper, refactored
    to use the toolkit's `windowing` function.
    """

    def __init__(self, config: WaveletConfig):
        super().__init__(config)
        self.level = config.level
        self.window_size = 2**config.level
        self.overlap = config.overlap
        self.offset = config.offset

        impulse = np.zeros(self.window_size)
        impulse[-1] = 1

        # SWT: Stationary Wavelet Transform
        swt_coefficients = pywt.swt(impulse, "haar", level=self.level)
        wt_filter_matrix = np.stack(
            [coeff[i] for coeff in swt_coefficients for i in range(2)] + [impulse],
            axis=-1,
        )

        self.feat_names = [
            f"{type_}{level}"
            for level in range(self.level, 0, -1)
            for type_ in [
                "A",
                "D",
            ]  # A -> approximation coefficients; D -> detail coefficients
        ] + ["A0"]  # A0 -> approx coeff on first level of wavelet filtering
        self.wt_filter_matrix = torch.tensor(wt_filter_matrix).double()

    def __call__(
        self, tags: pd.DataFrame, event_type: Optional[str] = None
    ) -> pd.DataFrame:
        # preserve names and index
        original_index_name = tags.index.name

        if self.offset > 0:
            tags = tags.iloc[self.offset :]

        # This list will track columns that actually produce features.
        processed_columns = []
        all_column_features = []

        for col_name in tags.columns:
            series = tags[col_name]
            # Skipping column with insufficient data
            if len(series) < self.window_size:
                continue

            # Using the windowing function with a "boxcar" window
            windows_df = windowing(
                X=series,
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

            windows_tensor = torch.tensor(windows_df.values).double()

            # Apply the wavelet filter matrix to the windowed data
            coeffs = torch.tensordot(
                windows_tensor, self.wt_filter_matrix, dims=([-1], [0])
            )

            records = {
                f"{col_name}_{f}": coeffs[:, j] for j, f in enumerate(self.feat_names)
            }
            all_column_features.append(pd.DataFrame(records))

        # If no columns were successfully processed, return an empty DataFrame.
        if not all_column_features:
            # Defining out_columns here for the empty case, using the original columns
            out_columns = [f"{t}_{f}" for f in self.feat_names for t in tags.columns]
            return pd.DataFrame(
                columns=out_columns,
                dtype=np.float64,
                index=pd.Index([], name=original_index_name),
            )

        # Build the final column list ONLY from processed columns.
        out_columns_final = [
            f"{t}_{f}" for f in self.feat_names for t in processed_columns
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
