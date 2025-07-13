import numpy as np
import pandas as pd
import torch
from ..core.base_feature_extractor import BaseFeatureExtractor, FeatureExtractorConfig


class StatisticalConfig(FeatureExtractorConfig):
    window_size: int = 10
    stride: int = 10
    offset: int = 0
    eps: float = 1e-6


class ExtractStatisticalFeatures(BaseFeatureExtractor):
    """
    PyTorch implementation of the statistical feature mapper
    """

    FEATURES = ["mean", "std", "skew", "kurt", "min", "1qrt", "med", "3qrt", "max"]

    def __init__(self, config: StatisticalConfig):
        super().__init__(config)
        self.window_size = config.window_size
        self.stride = config.stride
        self.offset = config.offset
        self.eps = config.eps

    def __call__(self, tags, event_type=None):
        # preserve names and index
        columns = tags.columns
        index = tags.index

        # output names
        out_columns = [f"{t}_{f}" for f in self.FEATURES for t in columns]

        if len(tags) < self.offset + self.window_size:
            # not enough for a single window
            out = pd.DataFrame(
                columns=out_columns, dtype=np.float64
            )  # return empty dataframe
            out.index.name = index.name
            return out

        tags = torch.Tensor(tags.values).double()

        # apply windowing operation
        tags = tags[self.offset :].unfold(0, self.window_size, self.stride)
        index = index[self.offset :][self.window_size - 1 :: self.stride]

        # central moment
        std, mean = torch.std_mean(tags, dim=-1, unbiased=False)
        # centralized - standardized values
        cstags = (tags - mean.unsqueeze(-1)) / (std.unsqueeze(-1) + self.eps)
        # normalized moments
        skew = cstags.pow(3).mean(dim=-1)
        kurt = cstags.pow(4).mean(dim=-1)

        # quantiles
        quantiles = torch.tensor([0.00, 0.25, 0.50, 0.75, 1.00]).double()
        q0, q1, q2, q3, q4 = tags.quantile(quantiles, dim=-1)

        records = {}
        for i, t in enumerate(columns):
            records[f"{t}_mean"] = mean[:, i]
            records[f"{t}_std"] = std[:, i]
            records[f"{t}_skew"] = skew[:, i]
            records[f"{t}_kurt"] = kurt[:, i]
            records[f"{t}_min"] = q0[:, i]
            records[f"{t}_1qrt"] = q1[:, i]
            records[f"{t}_med"] = q2[:, i]
            records[f"{t}_3qrt"] = q3[:, i]
            records[f"{t}_max"] = q4[:, i]

        # fill dataframe in correct order
        out = pd.DataFrame.from_records(records, index=index, columns=out_columns)
        out.index.name = index.name  # also preserve index
        return out
