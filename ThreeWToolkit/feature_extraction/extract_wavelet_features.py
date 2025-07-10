import pandas as pd
import torch
import numpy as np
import pywt
from typing import Optional
from ..core.base_feature_extractor import BaseFeatureExtractor, FeatureExtractorConfig


class WaveletConfig(FeatureExtractorConfig):
    level: int = 1
    stride: int = 1
    offset: int = 0


class ExtractWaveletFeatures(BaseFeatureExtractor):
  """PyTorch implementation of the wavelet feature mapper"""

  def __init__(self, config: WaveletConfig):
    super().__init__(config)
    self.level = config.level
    self.window_size = 2**config.level
    self.stride = config.stride
    self.offset = config.offset

    impulse = np.zeros(self.window_size)
    impulse[-1] = 1
    hs = pywt.swt(impulse, "haar", level=self.level)
    H = np.stack([h[i] for h in hs for i in range(2)] + [impulse], axis=-1)

    self.feat_names = [
        f"{type_}{level}"
        for level in range(self.level, 0, -1)
        for type_ in ["A", "D"] # A -> approximation coefficients; D -> detail coefficients
    ] + ["A0"] # A0 -> approx coeff on first level of wavelet filtering
    self.H = torch.tensor(H).double()

  def __call__(self, tags: pd.DataFrame, event_type: Optional[str] = None):
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
