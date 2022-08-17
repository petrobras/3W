"""This is the 3W toolkit's miscellaneous sub-module.

All resources that do not fit in the other sub-modules are define here.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors

from matplotlib.patches import Patch
from pathlib import Path
from multiprocessing.dummy import Pool as ThreadPool
from collections import defaultdict
from natsort import natsorted

from .base import COLUMNS_DATA_FILES, LABELS_DESCRIPTIONS, PATH_DATASET

# Methods
#
def label_and_file_generator(real=True, simulated=False, drawn=False):
    """This is a generating function that returns tuples for all
    indicated instance sources (`real`, `simulated` and/or
    `hand-drawn`). Each tuple refers to a specific instance and contains
    its label (int) and its full path (Path). All 3W dataset's instances
    are considered.

    Args:
        real (bool, optional): Indicates whether `real` instances should
            be considered. Defaults to True.
        simulated (bool, optional): Indicates whether `simulated`
            instances should be considered. Defaults to False.
        drawn (bool, optional): Indicates whether `hand-drawn` instances
            should be considered. Defaults to False.

    Yields:
        generator: Tuples for all indicated instance sources. Each tuple
            refers to a specific instance and contains its label (int)
            and its full path (Path).
    """
    for i in Path(PATH_DATASET).iterdir():
        try:
            # Considers only directories
            if i.is_dir():
                label = int(i.stem)
                for fp in i.iterdir():
                    # Considers only csv files
                    if fp.suffix == ".csv":
                        # Considers only instances from the requested
                        # source
                        if (
                            (simulated and fp.stem.startswith("SIMULATED"))
                            or (drawn and fp.stem.startswith("DRAWN"))
                            or (
                                real
                                and (not fp.stem.startswith("SIMULATED"))
                                and (not fp.stem.startswith("DRAWN"))
                            )
                        ):
                            yield label, fp
        except:
            # Otherwise (e.g. files or directory without instances), do
            # nothing
            pass


def get_all_labels_and_files():
    """Gets lists with tuples related to all real, simulated, or
    hand-drawn instances contained in the 3w dataset. Each list
    considers instances from a single source. Each tuple refers to a
    specific instance and contains its label (int) and its full path
    (Path).

    Returns:
        tuple: Tuple containing three lists with tuples related to real,
            simulated, and hand-drawn instances, respectively.
    """
    real_instances = list(
        label_and_file_generator(real=True, simulated=False, drawn=False)
    )
    simulated_instances = list(
        label_and_file_generator(real=False, simulated=True, drawn=False)
    )
    drawn_instances = list(
        label_and_file_generator(real=False, simulated=False, drawn=True)
    )

    return real_instances, simulated_instances, drawn_instances


def create_table_of_instances(real_instances, simulated_instances, drawn_instances):
    """Creates a table of instances (pandas.DataFrame) that shows the
    amount of instances that compose the 3W dataset, by knowledge source
    (real, simulated and hand-drawn instances) and by instance label.

    Args:
        real_instances (list): List with tuples related to all
            real instances contained in the 3w dataset. Each tuple
            must refer to a specific instance and must contain its label
            (int) and its full path (Path).
        simulated_instances (list): List with tuples related to all
            simulated instances contained in the 3w dataset. Each tuple
            must refer to a specific instance and must contain its label
            (int) and its full path (Path).
        drawn_instances (list): List with tuples related to all
            hand-drawn instances contained in the 3w dataset. Each tuple
            must refer to a specific instance and must contain its label
            (int) and its full path (Path).

    Returns:
        pandas.DataFrame: The created table that shows the amount of
            instances that compose the 3W dataset, by knowledge source
            (real, simulated and hand-drawn instances) and by instance
            label.
    """
    # Gets the label's description of all instances as a list of dicts
    list_instances = (
        [
            {
                "INSTANCE LABEL": str(label) + " - " + LABELS_DESCRIPTIONS[label],
                "SOURCE": "REAL",
            }
            for label, fp in real_instances
        ]
        + [
            {
                "INSTANCE LABEL": str(label) + " - " + LABELS_DESCRIPTIONS[label],
                "SOURCE": "SIMULATED",
            }
            for label, fp in simulated_instances
        ]
        + [
            {
                "INSTANCE LABEL": str(label) + " - " + LABELS_DESCRIPTIONS[label],
                "SOURCE": "HAND-DRAWN",
            }
            for label, fp in drawn_instances
        ]
    )

    # Transforms the list of dicts into a pandas.DataFrame
    df_instances = pd.DataFrame(list_instances)

    # Creates the table of instances with relevant information and
    # desired format
    toi = (
        df_instances.groupby(["INSTANCE LABEL", "SOURCE"])
        .size()
        .reset_index()
        .pivot("SOURCE", "INSTANCE LABEL", 0)
        .fillna(0)
        .astype(int)
        .T
    )
    toi = toi.loc[natsorted(toi.index.values)]
    toi = toi[["REAL", "SIMULATED", "HAND-DRAWN"]]
    toi["TOTAL"] = toi.sum(axis=1)
    toi.loc["TOTAL"] = toi.sum(axis=0)

    return toi


def filter_rare_undesirable_events(toi, threshold, simulated=False, drawn=False):
    """Generates a table of instances (pandas.DataFrame) that shows the
    amount of filtered instances, by knowledge source (real, `simulated`
    and `hand-drawn` instances) and by instance label. This filter keeps
    only real instances, as well as `simulated` and `hand-drawn` if
    indicated, of rare event types. An event type is considered rare if
    the amount of instances labeled as this event relative to the total
    number of instances is less than the indicated `threshold`. In both
    totalizations, `simulated` and `hand-drawn` instances are only
    considered if indicated, but real instances are always taken into
    account.

    Args:
        toi (pandas.DataFrame): Table that shows the amount of instances
            that compose the 3W dataset, by knowledge source (real,
            `simulated` and `hand-drawn` instances) and by instance
            label. This object is not modified in this function.
        threshold (float): Relative limit that establishes rare event
            types.
        simulated (bool, optional): Indicates whether `simulated`
            instances should be considered. Defaults to False.
        drawn (bool, optional): Indicates whether `hand-drawn` instances
            should be considered. Defaults to False.

    Returns:
        pandas.DataFrame: The table of instances (pandas.DataFrame) that
            shows the amount of filtered instances, by knowledge source
            (real, simulated and hand-drawn instances) and by instance
            label.
    """
    # Simulated and hand-drawn instances are optional, but real
    # instances are always considered
    totals = 0
    if simulated:
        totals += toi["SIMULATED"]
    if drawn:
        totals += toi["HAND-DRAWN"]
    totals += toi["REAL"]

    # Absolute limit
    limit = threshold * totals[-1]

    # Applies the filter in a new pandas.DataFrame
    rue = toi.loc[totals < limit].copy()
    rue.loc["TOTAL"] = rue.sum(axis=0)

    return rue


def load_instance(instance):
    """Loads all data and metadata from a specific `instance`.

    Args:
        instance (tuple): This tuple must refer to a specific `instance`
            and contain its label (int) and its full path (Path).

    Raises:
        Exception: Error if the CSV file passed as arg cannot be read.

    Returns:
        pandas.DataFrame: Its index contains the timestamps loaded from
            the CSV file. Its columns contain data loaded from the other
            columns of the CSV file and metadata loaded from the
            argument `instance` (label, well, and id).
    """
    # Loads label metadata from the argument `instance`
    label, fp = instance

    try:
        # Loads well and id metadata from the argument `instance`
        well, id = fp.stem.split("_")

        # Loads data from the CSV file
        df = pd.read_csv(fp, index_col="timestamp", parse_dates=["timestamp"])
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


def load_instances(instances):
    """Loads all data and metadata from multiple `instances` in
    parallel.

    Args:
        instances (list): List with tuples related to real, simulated,
            or hand-drawn `instances`. Each tuple must refer to a
            specific instance and must contain its label (int) and its
            full path (Path).

    Returns:
        pandas.DataFrame: Its index contains the timestamps loaded from
            the CSV files. Its columns contain data loaded from the
            other columns of the CSV files and the metadata label, well,
            and id).
    """
    # Prepares for multiple parallel loadings
    pool = ThreadPool()
    dfs = []

    try:
        # Calls multiple loadings in parallel
        for df in pool.imap_unordered(load_instance, instances):
            dfs.append(df)
    finally:
        # If the instance cannot be loaded
        pool.terminate()

    # Concatenates dfs and return the result
    return pd.concat(dfs)


def create_and_plot_scatter_map(real_instances):
    """Creates and plots scatter map with all the real instances listed
    in the `real_instances` argument.

    Args:
        real_instances (list): List with tuples related to all
            real instances contained in the 3w dataset. Each tuple
            must refer to a specific instance and must contain its label
            (int) and its full path (Path).

    Returns:
        tuple: Tuple containing the first and the last year of
            occurrence among all instances, respectively.
    """
    # Loads all instances
    df = load_instances(real_instances)

    # Finds the first and the last year of occurrence among all instances
    df_time = (
        df.reset_index()
        .groupby(["well", "id", "label"])["timestamp"]
        .agg(["min", "max"])
    )

    well_times = defaultdict(list)
    well_classes = defaultdict(list)
    for (well, id, label), (tmin, tmax) in df_time.iterrows():
        well_times[well].append(
            (tmin.toordinal(), (tmax.toordinal() - tmin.toordinal()))
        )
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
    plt.rcParams["legend.fontsize"] = 9
    fig, ax = plt.subplots(figsize=(9, 4))
    yticks = []
    yticks_labels = []
    for well in well_times.keys():
        times = well_times[well]
        class_names = well_classes[well]
        class_colors = list(map(cmap, class_names))
        well_id = well_code[well]
        yticks.append(well_id * height + height / 2 - border / 2)
        yticks_labels.append(well)
        ax.broken_barh(
            times,
            (well_id * height, height - border),
            facecolors=class_colors,
            edgecolors=class_colors,
        )
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks_labels)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    legend_colors = [
        Patch(facecolor=cmap(l), label=str(l) + " - " + d)
        for l, d in LABELS_DESCRIPTIONS.items()
    ]
    ax.legend(
        frameon=False,
        handles=legend_colors,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=3,
    )

    return first_year, last_year


def count_properties_instance(instance):
    """Counts properties from a specific `instance`.

    Args:
        instance (tuple): This tuple must refer to a specific `instance`
            and contain its label (int) and its full path (Path).

    Raises:
        Exception: Error if the CSV file passed as arg cannot be read.

    Returns:
        dict: Dict containing the counted properties with the following
            keys: n_vars (number of variables), n_vars_missing (number
            of missing variables), n_vars_frozen (number of frozen
            variables), n_obs (number of observations), and
            n_obs_unlabeled (number of unlabeled observations).
    """
    # Preparation for counting
    _, fp = instance
    p = {"n_vars_missing": 0, "n_vars_frozen": 0}

    try:
        # Read the CSV file
        df = pd.read_csv(fp, index_col="timestamp", parse_dates=["timestamp"])
    except Exception as e:
        raise Exception(f"error reading file {fp}: {e}")

    # Counts properties
    vars = df.columns[:-1]  # Last column with class is not considered
    p["n_vars"] = len(vars)
    for var in vars:
        if df[var].isnull().all():
            p["n_vars_missing"] += 1
        u_values = df[var].unique()
        if len(u_values) == 1 and not np.isnan(u_values):
            p["n_vars_frozen"] += 1
    p["n_obs"] = len(df)
    p["n_obs_unlabeled"] = df["class"].isnull().sum()

    return p


def count_properties_instances(instances):
    """Counts properties from multiple `instances` in parallel.

    Args:
        instances (list): List with tuples related to real, simulated,
            or hand-drawn `instances`. Each tuple must refer to a
            specific instance and must contain its label (int) and its
            full path (Path).

    Returns:
        dict: Dict containing the counted properties with the following
            keys: n_vars (number of variables), n_vars_missing (number
            of missing variables), n_vars_frozen (number of frozen
            variables), n_obs (number of observations), and
            n_obs_unlabeled (number of unlabeled observations).
    """
    # Prepares for multiple parallel counts
    pool = ThreadPool()
    ps = []

    try:
        # Calls multiple counts in parallel
        for p in pool.imap_unordered(count_properties_instance, instances):
            ps.append(p)
    finally:
        # If the instance cannot be loaded
        pool.terminate()

    # Sum ps and return the result
    return dict(pd.DataFrame(ps).sum())


def calc_stats_instances(real_instances, simulated_instances, drawn_instances):
    """Calculates the 3W dataset's fundamental aspects related to
    inherent difficulties of actual data. Three statistics are
    calculated: Missing Variables, Frozen Variables, and Unlabeled
    Observations. All instances, regardless of their source, influence
    these statistics.

    Args:
        real_instances (list): List with tuples related to all
            real instances contained in the 3w dataset. Each tuple
            must refer to a specific instance and must contain its label
            (int) and its full path (Path).
        simulated_instances (list): List with tuples related to all
            simulated instances contained in the 3w dataset. Each tuple
            must refer to a specific instance and must contain its label
            (int) and its full path (Path).
        drawn_instances (list): List with tuples related to all
            hand-drawn instances contained in the 3w dataset. Each tuple
            must refer to a specific instance and must contain its label
            (int) and its full path (Path).

    Returns:
        pandas.DataFrame: Its index contains the statistic's names. Its
            columns contain statistics themselves (Amount and
            Percentage)
    """
    # Counts properties from all indicated instances
    p = count_properties_instances(
        real_instances + simulated_instances + drawn_instances
    )

    # Extract properties
    n_vars = p["n_vars"]
    n_vars_missing = p["n_vars_missing"]
    n_vars_frozen = p["n_vars_frozen"]
    n_obs = p["n_obs"]
    n_obs_unlabeled = p["n_obs_unlabeled"]

    # Calculates the statistics
    stats = {
        "Missing Variables": [
            n_vars_missing,
            f"{100*n_vars_missing/n_vars:.2f}% of {n_vars}",
        ],
        "Frozen Variables": [
            n_vars_frozen,
            f"{100*n_vars_frozen/n_vars:.2f}% of {n_vars}",
        ],
        "Unlabeled Observations": [
            n_obs_unlabeled,
            f"{100*n_obs_unlabeled/n_obs:.2f}% of {n_obs}",
        ],
    }

    return pd.DataFrame.from_dict(
        stats, orient="index", columns=["Amount", "Percentage"]
    )
