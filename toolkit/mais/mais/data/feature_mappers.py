""" Feature mapping strategies """

import numpy as np
import pandas as pd
import scipy.stats as sp
import pywt

import torch


class StatisticalFeatureMapper:
    """generates statistical descriptor for a window of our data"""

    FEATURES = {
        "mean": lambda x: x.mean(),
        "std": lambda x: x.std(),
        "skew": lambda x: x.apply(sp.skew, raw=True),  # pandas skew is bad
        "kurt": lambda x: x.apply(sp.kurtosis, raw=True),  # kurtosis too
        "max": lambda x: x.max(),
        "min": lambda x: x.min(),
        "1qrt": lambda x: x.quantile(0.25),
        "med": lambda x: x.median(),
        "3qrt": lambda x: x.quantile(0.75),
    }

    def __init__(self, window_size, stride=1, offset=0):
        self.window_size = window_size
        self.stride = stride
        self.offset = offset

    def __call__(self, tags, event_type=None):

        # apply rolling window
        tag_names = tags.columns
        tags = tags.rolling(window=self.window_size)

        feats = []
        for f in StatisticalFeatureMapper.FEATURES:

            # apply feature mapping
            feat = StatisticalFeatureMapper.FEATURES[f](tags)
            # append feature_name to column
            feat = feat.rename(columns={t: f"{t}_{f}" for t in tag_names})
            feats.append(feat)

        feats = pd.concat(feats, axis="columns")
        return feats[self.offset :: self.stride]


class TorchStatisticalFeatureMapper:
    """PyTorch implementation of the statistical feature mapper"""

    FEATURES = ["mean", "std", "skew", "kurt", "min", "1qrt", "med", "3qrt", "max"]

    def __init__(self, window_size, stride=10, offset=0, eps=1e-6):
        self.window_size = window_size
        self.stride = stride
        self.offset = offset
        self.eps = eps

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


class TorchEWStatisticalFeatureMapper:
    """PyTorch implementation of a truncated exponentially weighted
    statistical feature mapper"""

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

    def __init__(self, window_size, decay, stride=10, offset=0, eps=1e-6):
        self.window_size = window_size
        self.stride = stride
        self.offset = offset
        self.eps = eps

        h = decay ** torch.arange(window_size, 0, step=-1)
        self.h = h / (h.abs().sum() + self.eps)  # L1 normalization

    def _E(self, X, dim=-1):
        """Take the exponentially weighted average trhough a dot in the specified dimension"""
        return torch.tensordot(X, self.h, dims=[[dim], [0]])

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

        tags = torch.tensor(tags.values).double()

        # apply windowing operation
        tags = tags[self.offset :].unfold(0, self.window_size, self.stride)
        index = index[self.offset :][self.window_size - 1 :: self.stride]

        # central moment
        mean = self._E(tags, dim=-1)
        std = self._E(torch.pow(tags - mean.unsqueeze(-1), 2)).sqrt()

        # centralized - standardized values
        cstags = (tags - mean.unsqueeze(-1)) / (std.unsqueeze(-1) + self.eps)
        # normalized moments
        skew = self._E(cstags.pow(3), dim=-1)
        kurt = self._E(cstags.pow(4), dim=-1)

        # quantiles
        quantiles = torch.tensor([0.00, 0.25, 0.50, 0.75, 1.00]).double()
        q0, q1, q2, q3, q4 = cstags.quantile(quantiles, dim=-1)

        records = {}
        for i, t in enumerate(columns):
            records[f"{t}_ew_mean"] = mean[:, i]
            records[f"{t}_ew_std"] = std[:, i]
            records[f"{t}_ew_skew"] = skew[:, i]
            records[f"{t}_ew_kurt"] = kurt[:, i]
            records[f"{t}_ew_min"] = q0[:, i]
            records[f"{t}_ew_1qrt"] = q1[:, i]
            records[f"{t}_ew_med"] = q2[:, i]
            records[f"{t}_ew_3qrt"] = q3[:, i]
            records[f"{t}_ew_max"] = q4[:, i]

        # fill dataframe in correct order
        out = pd.DataFrame.from_records(records, index=index, columns=out_columns)
        out.index.name = index.name  # also preserve index
        return out


class TorchWaveletFeatureMapper:
    """PyTorch implementation of the wavelet feature mapper"""

    def __init__(self, level, stride, offset=0):
        self.level = level
        self.window_size = 2**level
        self.stride = stride
        self.offset = offset

        impulse = np.zeros(self.window_size)
        impulse[-1] = 1
        hs = pywt.swt(impulse, "haar", level=self.level)
        H = np.stack([h[i] for h in hs for i in range(2)] + [impulse], axis=-1)

        self.feat_names = [
            f"{type_}{level}"
            for level in range(self.level, 0, -1)
            for type_ in ["A", "D"]
        ] + ["A0"]
        self.H = torch.tensor(H).double()

    def __call__(self, tags, event_type=None):
        # preserve names and index
        columns = tags.columns
        index = tags.index

        # output names
        out_columns = [f"{t}_{f}" for f in self.feat_names for t in columns]

        if len(tags) < self.offset + self.window_size:
            # not enough for a single window
            out = pd.DataFrame(
                columns=out_columns, dtype=np.float64
            )  # return empty dataframe
            out.index.name = index.name
            return out

        tags = torch.tensor(tags.values).double()

        # apply windowing operation
        tags = tags[self.offset :].unfold(0, self.window_size, self.stride)
        index = index[self.offset :][self.window_size - 1 :: self.stride]

        coeffs = torch.tensordot(tags, self.H, dims=([-1], [0]))

        records = {}
        for i, t in enumerate(columns):
            for j, f in enumerate(self.feat_names):
                records[f"{t}_{f}"] = coeffs[:, i, j]

        # fill dataframe in correct order
        out = pd.DataFrame.from_records(records, index=index, columns=out_columns)
        out.index.name = index.name  # also preserve index
        return out


class MixedMapper:
    """join features of multiple mappers. Feature sizes must be consistent"""

    def __init__(self, *args):
        self.mappers = args

    def __call__(self, tags, event_type=None):
        return pd.concat([m(tags, event_type) for m in self.mappers], axis="columns")
