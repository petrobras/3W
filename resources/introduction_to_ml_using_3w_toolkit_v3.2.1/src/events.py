"""
Small helpers around the 3W Toolkit objects used in notebook 1.

The toolkit loads, splits, scales and windows the events; these functions only
cover what it does not offer: the course-specific cleaning of an event, a view
of in-memory events that toolkit steps can be fitted on, and the repacking of
windows into one dataframe each (the format saved for notebooks 2 to 5).
"""

from typing import List, Optional

import pandas as pd

from ThreeWToolkit.core.base_dataset import BaseDataset
from ThreeWToolkit.core.dataset_outputs import DatasetOutputs

from . import config


def clean_event(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Clean one event: drop rows without label, downsample, fill gaps.

    The toolkit has no equivalent for the downsampling by `SAMPLING_RATE`.
    Returns None for events without data or with a sensor that is entirely
    missing (nothing to fill it with).
    """
    df = df.dropna(subset=[config.CLASS_COLUMN]).reset_index(drop=True)
    if config.ENABLE_DATA_SAMPLING and config.SAMPLING_RATE > 1:
        if config.SAMPLING_METHOD == "random":
            n_rows = len(df) // config.SAMPLING_RATE
            df = (
                df.sample(n=n_rows, random_state=config.RANDOM_SEED)
                .sort_index()
                .reset_index(drop=True)
            )
        else:  # uniform: one row in every SAMPLING_RATE
            df = df.iloc[:: config.SAMPLING_RATE].reset_index(drop=True)
    numeric = df.select_dtypes(include="number").columns
    df[numeric] = df[numeric].ffill().bfill()
    if len(df) == 0 or df.dropna().empty:
        return None
    return df


class InMemoryEvents(BaseDataset):
    """Events already in memory, as a toolkit dataset (so `Normalize.fit` accepts them)."""

    def __init__(self, signals: List[pd.DataFrame]):
        self.signals = signals

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> DatasetOutputs:
        return DatasetOutputs(signal=self.signals[idx], label=None, metadata={})


def windows_to_frames(
    event: pd.DataFrame, windowed: DatasetOutputs, window_size: int, step: int
) -> List[pd.DataFrame]:
    """One dataframe per window emitted by the toolkit `Windowing` for `event`.

    `Windowing` returns a frame indexed by (window, variable) with the time steps
    as columns; here each window becomes a (window_size, columns) dataframe that
    keeps the label column of the event, as the notebooks 2 to 5 expect.
    """
    sensors = [c for c in event.columns if c != config.CLASS_COLUMN]
    values = windowed.signal.to_numpy().reshape(-1, len(sensors), window_size)
    frames = []
    for k, window in enumerate(values.transpose(0, 2, 1)):
        frame = event.iloc[k * step : k * step + window_size].reset_index(drop=True)
        frame[sensors] = window
        frames.append(frame)
    return frames
