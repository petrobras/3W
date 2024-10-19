"""This is the 3W Toolkit's miscellaneous sub-module.

All resources that do not fit in the other sub-modules are define here.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import os

from matplotlib.patches import Patch
from pathlib import Path
from multiprocessing.dummy import Pool as ThreadPool
from collections import defaultdict
from natsort import natsorted

import warnings

warnings.simplefilter("ignore", FutureWarning)
import plotly.offline as py
import plotly.graph_objs as go
import glob
import pandas as pd
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from .base import (
    COLUMNS_DATA_FILES,
    LABELS_DESCRIPTIONS,
    PATH_DATASET,
    VARS,
    EVENT_NAMES,
    PARQUET_EXTENSION,
    PARQUET_ENGINE,
    load_3w_dataset,  # To work with 3W v2.0
)


# Methods
#


def create_table_of_instances(df):
    """Creates a table of instances (pandas.DataFrame) that shows the
    amount of instances that compose the 3W Dataset, by knowledge source
    (real, simulated and imputed instances) and by instance label.

    Args:
        df (pandas.DataFrame): DataFrame with the 3W Dataset 2.0 data.

    Returns:
        pandas.DataFrame: The created table that shows the amount of
            instances that compose the 3W Dataset, by knowledge source
            (real, simulated and imputed instances) and by instance
            label.
    """

    # Create a new column with the instance label and description
    df['INSTANCE LABEL'] = df['label'].astype(str) + " - " + df['label'].map(LABELS_DESCRIPTIONS)

    # Create the table of instances
    toi = (
        df.groupby(['INSTANCE LABEL', 'state'])
        .size()
        .reset_index()
        .pivot('state', 'INSTANCE LABEL', 0)
        .fillna(0)
        .astype(int)
        .T
    )

    # Rename the columns to represent the data sources
    toi = toi.rename(columns={0: 'REAL', 1: 'SIMULATED', 2: 'IMPUTED'})

    # Add a 'TOTAL' column and row
    toi['TOTAL'] = toi.sum(axis=1)
    toi.loc['TOTAL'] = toi.sum(axis=0)

    return toi


def filter_rare_undesirable_events(toi, threshold, simulated=False, imputed=False):
    """Generates a table of instances (pandas.DataFrame) that shows the
    amount of filtered instances, by knowledge source (real, `simulated`
    and `imputed` instances) and by instance label. This filter keeps
    only real instances, as well as `simulated` and `imputed` if
    indicated, of rare event types. An event type is considered rare if
    the amount of instances labeled as this event relative to the total
    number of instances is less than the indicated `threshold`. In both
    totalizations, `simulated` and `imputed` instances are only
    considered if indicated, but real instances are always taken into
    account.

    Args:
        toi (pandas.DataFrame): Table that shows the amount of instances
            that compose the 3W Dataset, by knowledge source (real,
            `simulated` and `imputed` instances) and by instance
            label. This object is not modified in this function.
        threshold (float): Relative limit that establishes rare event
            types.
        simulated (bool, optional): Indicates whether `simulated`
            instances should be considered. Defaults to False.
        imputed (bool, optional): Indicates whether `imputed` instances
            should be considered. Defaults to False.

    Returns:
        pandas.DataFrame: The table of instances (pandas.DataFrame) that
            shows the amount of filtered instances, by knowledge source
            (real, simulated and imputed instances) and by instance
            label.
    """
    # Simulated and imputed instances are optional, but real
    # instances are always considered
    totals = 0
    if simulated:
        totals += toi["SIMULATED"]
    if imputed:
        totals += toi["IMPUTED"]
    totals += toi["REAL"]

    # Absolute limit
    limit = threshold * totals[-1]

    # Applies the filter in a new pandas.DataFrame
    rue = toi.loc[totals < limit].copy()
    rue.loc["TOTAL"] = rue.sum(axis=0)

    return rue


def load_instance(label, fp):
    """Loads all data and metadata from a specific `instance`.

    Args:
        label (int): Label of the instance.
        fp (Path): Full path to the instance file.

    Raises:
        Exception: Error if the Parquet file passed as arg cannot be
        read.

    Returns:
        pandas.DataFrame: Its index contains the timestamps loaded from
            the Parquet file. Its columns contain data loaded from the
            other columns of the Parquet file and metadata loaded from
            the argument `instance` (label, well, and id).
    """

    try:
        # Loads well and id metadata from the argument `instance`
        well, id = fp.stem.split("_")

        # Loads data from the Parquet file
        df = pd.read_parquet(fp, engine=PARQUET_ENGINE)
        assert (
            df.columns == COLUMNS_DATA_FILES[1:]
        ).all(), f"invalid columns in the file {fp}: {df.columns.tolist()}"
    except Exception as e:
        raise Exception(f"error reading file {fp}: {e}")

    # Incorporates the loaded metadata
    df["label"] = label
    df["well"] = well
    df["id"] = id

    # Incorporates the loaded data and ordenates the df's columns
    df = df[["label", "well", "id"] + COLUMNS_DATA_FILES[1:]]

    return df


def load_instances(df):  # Changed function signature
    """Loads all data and metadata from the DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame with the 3W Dataset 2.0 data.

    Returns:
        pandas.DataFrame: DataFrame with loaded instances.
    """

    # Prepares for multiple parallel loadings
    pool = ThreadPool()
    dfs = []

    try:
        # Calls multiple loadings in parallel
        for label, fp in df[['label', 'filepath']].values:  # Assuming 'filepath' column exists
            dfs.append(load_instance(label, Path(fp)))
    finally:
        # If the instance cannot be loaded
        pool.terminate()

    # Concatenates dfs and return the result
    return pd.concat(dfs)


def create_and_plot_scatter_map(df):
    """Creates and plots scatter map with all the real instances listed
    in the `df` argument.

    Args:
        df (pandas.DataFrame): DataFrame with the 3W Dataset 2.0 data.

    Returns:
        tuple: Tuple containing the first and the last year of
            occurrence among all instances, respectively.
    """

    # Finds the first and the last year of occurrence among all instances
    df_time = (
        df.reset_index()
        .groupby(["well", "id", "label"])["timestamp"]
        .agg(["min", "max"])
    )

    well_times = defaultdict(list)
    well_classes = defaultdict(list)
    for (well, id, label), (tmin, tmax) in df_time.iterrows():
        well_times[well].append((tmin, (tmax - tmin)))
        well_classes[well].append(label)

    wells = df["well"].unique()
    well_code = {w: i for i, w in enumerate(sorted(wells))}

    # Configures and plots the scatter map
    cmap = plt.get_cmap("Paired")
    my_colors = [cmap(i) for i in [3, 0, 5, 8, 11, 2, 1, 4, 9, 7, 6, 10]]
    my_cmap = mcolors.ListedColormap(my_colors, name="my_cmap")
    plt.register_cmap(name="my_cmap", cmap=my_cmap)
    cmap = plt.get_cmap("my_cmap")
    height = 5
    border = 2
    first_year = np.min(df_time["min"]).year
    last_year = np.max(df_time["max"]).year
    plt.rcParams["axes.labelsize"] = 9
    plt.rcParams["font.size"] = 9
    plt.rcParams["legend.fontsize"]
