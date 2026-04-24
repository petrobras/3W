"""Utility functions for the clustering sub-package."""

from __future__ import annotations

from typing import List

import numpy as np


def compute_dba_centroid(
    series: List[np.ndarray],
    indices: List[int] | None = None,
) -> np.ndarray:
    """Compute the DTW Barycenter Averaging (DBA) centroid of a set of series.

    Args:
        series: List of time series arrays.
        indices: Optional subset of indices into ``series`` to use. If ``None``,
            all series are used.

    Returns:
        1-D array representing the DBA centroid.

    Raises:
        ImportError: If ``dtaidistance`` is not installed.
        ValueError: If the resulting series subset is empty.
    """
    try:
        from dtaidistance import dtw_barycenter
    except ImportError as exc:
        raise ImportError(
            "dtaidistance is required for DBA centroid computation. "
            "Install it with: pip install ThreeWToolkit[clustering]"
        ) from exc

    subset = (
        [series[i].astype(np.float64) for i in indices]
        if indices is not None
        else [s.astype(np.float64) for s in series]
    )

    if not subset:
        raise ValueError("No series provided for centroid computation.")

    return dtw_barycenter.dba_loop(subset, c=None, use_c=True)
