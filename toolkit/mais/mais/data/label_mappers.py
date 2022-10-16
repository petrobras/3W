""" Strategies for deciding on label for a given region of data, using pandas or torch backends """

import pandas as pd
import numpy as np
import scipy.stats as sp
import torch

from mais.data.dataset import MAEDataset


class RollingLabelStrategy:
    """Base class that just wraps applications of apply,
    leverages pandas' Rolling function"""

    def __init__(self, window_size, stride=1, offset=0):
        self.window_size = window_size
        self.stride = stride
        self.offset = offset

    def apply(self, y, event_type):
        raise NotImplementedError

    def __call__(self, labels, event_type):
        def f(y):
            return self.apply(y, event_type)

        labels = labels.rolling(window=self.window_size).apply(f, raw=True)
        return labels[self.offset :: self.stride]


class BinaryMCLStrategy(RollingLabelStrategy):
    """Window label gets assigned to most common value,
    mapping transients and faults of ALL classes to true"""

    def apply(self, y, event_type=None):
        """map all fault types to True and apply mode over window"""
        return sp.mode(y > 0)[0]


class MulticlassMCLStrategy(RollingLabelStrategy):
    """Window label gets assigned to most common value,
    mapping transients and faults to the CORRESPONDING CLASS CODE"""

    def apply(self, y, event_type=None):
        """map transient codes to fault codes and apply mode over window"""
        return sp.mode(y % 100)[0]


class OVAMCLStrategy(RollingLabelStrategy):
    """Window label gets assigned to most common value,
    mapping transients and faults of SPECIFIC CLASS to true"""

    def __init__(self, fault_code, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fault_code = fault_code

    def apply(self, y, event_type=None):
        return sp.mode(y % 100 == self.fault_code)[0]


class TorchLabelStrategy:
    """Base class that just wraps applications of apply,
    leverages pytorch unfold function"""

    def __init__(self, window_size, stride=1, offset=0):
        self.window_size = window_size
        self.stride = stride
        self.offset = offset

    def apply(self, y, event_type):
        raise NotImplementedError

    def __call__(self, labels, event_type):

        # store index
        index = labels.index

        # not enough samples for windowing, return empty
        if len(labels) < self.offset + self.window_size:
            out = pd.Series(name=MAEDataset.LABEL_NAME, dtype=np.float64)
            out.index.name = index.name
            return out

        # pass to pytorch as float (propagate nan)
        labels = torch.tensor(labels.values, dtype=torch.float32).squeeze()

        # apply windowing
        y = labels[self.offset :].unfold(0, self.window_size, self.stride)
        index = index[self.offset :][self.window_size - 1 :: self.stride]

        out = pd.Series(
            name=MAEDataset.LABEL_NAME,
            data=self.apply(y, event_type),
            index=index,
            dtype=np.float64,
        )
        out.index.name = index.name
        return out


class TorchBinaryMCLStrategy(TorchLabelStrategy):
    """any fault indicator, most common label"""

    def apply(self, y, event_type=None):
        return torch.mode(y, dim=-1)[0] > 0


class TorchBinaryMRLStrategy(TorchLabelStrategy):
    """any fault indicator, most recent label"""

    def apply(self, y, event_type=None):
        return y[:, -1] > 0


class TorchOVAMCLStrategy(TorchLabelStrategy):
    """specific class indicator, most common label"""

    def __init__(self, fault_code, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fault_code = fault_code

    def apply(self, y, event_type=None):
        return (torch.mode(y % 100)[0] == self.fault_code).float()


class TorchOVATransientMCLStrategy(TorchLabelStrategy):
    """transients of specific class, most common label"""

    def __init__(self, fault_code, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fault_code = 100 + fault_code

    def apply(self, y, event_type=None):
        return 1.0 * torch.mode(y % 100 == self.fault_code)[0]


class TorchOVAMRLStrategy(TorchLabelStrategy):
    """transients and faults of specific class, most recent label, propagates nans"""

    _NAN = torch.tensor([np.nan]).float()

    def __init__(self, fault_code, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fault_code = fault_code

    def apply(self, y, event_type=None):
        last = y[:, -1]
        return torch.where(
            last.isnan(), self._NAN, (last % 100 == self.fault_code).float()
        )


class TorchMulticlassMCLStrategy(TorchLabelStrategy):
    """detect transients and faults, most common value"""

    def apply(self, y, event_type=None):
        return torch.mode(y % 100, dim=-1)[0]


class TorchMulticlassMRLStrategy(TorchLabelStrategy):
    """detect transients and faults, most common value"""

    _NAN = torch.tensor([np.nan]).float()

    def apply(self, y, event_type=None):
        last = y[:, -1]
        return torch.where(last.isnan(), self._NAN, (last % 100).float())
