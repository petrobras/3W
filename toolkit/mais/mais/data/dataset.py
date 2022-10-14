# coding: utf-8
""" Basic dataset definitions """

from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas

from joblib import Parallel, delayed


class MAEDataset:
    """Load all files and return transformed glob

    Most of this class deals with selecting which event '.csv's should be loaded,
        and feeding each 'raw' csv to some feature mapper.

    * Constructor arguments:
        - **root_dir: STRING** -- base location of events separated by event type

        - **tgt_class: LIST** -- codes of which event types should be used

        - **selected_tags: STRING LIST** -- names of columns that should be used on
            feature extraction, i.e. ["T-TPT", "QGL"]

        - **instance_types: STRING LIST** -- types of instances that should be used list containing
            one or more of {"real", "simulated", "drawn"}

        - **feature_mapper: CALLABLE**
        (raw_tags: DataFrame,  raw_labels: DataFrame) ->
        (feature: DataFrame x label: DataFrame) --
        deals with transforming the raw data from a single event,
        to be used during training/evaluation. Outputs from each event concatenated
        afterwards.

        - **n_jobs: INT** -- number of processes to use

        -- **events**

    * Important fields:

        - **self.X: [N_SAMPLE x N_FEATURES]** ndarray created by concatenating every feature output from feature mapper

        - **self.Y: [N_SAMPLE]** ndarray created by concatenating every label output from feature mapper

        - **self.feature_names: [N_FEATURES]** ndarray as output by the feature mapper

        - **self.g_len: [N_EVENTS]** number of samples output by the feature mapper for each event

        - **self.g_class: [N_EVENTS]** list denoting the origin of the event


    * Methods:

        - **_is_instance_type(type_)**

        - **_make_set()**

        -  **process(fname, event_type)**

    """

    # tag corresponding to instance label
    LABEL_NAME = "class"

    # tag corresponding to index
    INDEX_NAME = "timestamp"

    # fault description
    CLASS_NAMES = {
        0: "NORMAL",
        1: "ABRUPT_INCREASE_OF_BSW",
        2: "SPURIOUS_CLOSURE_OF_DHSV",
        3: "SEVERE_SLUGGING",
        4: "FLOW_INSTABILITY",
        5: "RAPID_PRODUCTIVITY_LOSS",
        6: "QUICK_RESTRICTION_IN_PCK",
        7: "SCALING_IN_PCK",
        8: "HYDRATE_IN_PRODUCTION_LINE",
    }

    # list of used classes
    KNOWN_CLASSES = list(CLASS_NAMES.keys())

    # transient properties of events
    TRANSIENT_CLASS = {
        0: False,
        1: True,
        2: True,
        3: False,
        4: False,
        5: True,
        6: True,
        7: True,
        8: True,
    }

    # List of known data tags
    TAG_NAMES = [
        "P-PDG",
        "P-TPT",
        "T-TPT",
        "P-MON-CKP",
        "T-JUS-CKP",
        "P-JUS-CKGL",
        "T-JUS-CKGL",
        "QGL",
    ]

    def __init__(
        self,
        transformed_events=None,
        events=None,  # either pass in preloaded events or the root directory
        root_dir=None,
        tgt_events=[],  # which event_types to load
        instance_types=[],  # simulated and or real and or drawn
        feature_mapper=tuple,  # transformer from event to features
        n_jobs=-1,
    ):
        """Load and process dataset using supplied strategies"""

        # save parameters
        self.root_dir = root_dir
        self.tgt_events = tgt_events
        self.instance_types = instance_types
        self.n_jobs = n_jobs
        self.feature_mapper = feature_mapper

        # call the heavy load _make_set passing the (maybe Null) events
        self._make_set(events)

    def _instance_type(fname):
        """Detects if instance type is selected

        * Parameters:
            - **fname**: STRING - name of the instance file
        * Returns:
            - **STRING** - string representing the instance type of the input file name

        """
        if fname.startswith("OLGA"):
            return "simulated"
        elif fname.startswith("DESENHADA"):
            return "drawn"
        else:
            return "real"

    def load_events(data_root, n_jobs=-1):
        """scan data_root for raw files and return dict. useful for preloads"""

        def _read(tgt, fname):
            df = pandas.read_csv(
                fname,
                index_col=MAEDataset.INDEX_NAME,
                header=0,
                parse_dates=True,
                memory_map=True,
            )
            tags = df[MAEDataset.TAG_NAMES]
            labels = df[MAEDataset.LABEL_NAME]

            return {
                "file_name": str(fname.relative_to(tgt)),
                "tags": tags,
                "labels": labels,
                "event_type": int(str(tgt.relative_to(data_root))),
            }

        data_root = Path(data_root)
        target_dirs = [
            d for d in data_root.iterdir() if d.match("[0-8]")
        ]  # filter directories with classes
        tasks = [(tgt, fname) for tgt in target_dirs for fname in tgt.glob("*.csv")]
        with Parallel(n_jobs) as p:
            events = p(delayed(_read)(*t) for t in tasks)
        return events

    def transform_events(
        events, raw_mapper, tgt_events=None, instance_types=None, n_jobs=-1
    ):
        """apply raw_mapper to list of events, filtering by target events and instance types"""
        if tgt_events is not None:
            events = [e for e in events if (e["event_type"] in tgt_events)]
        if instance_types is not None:
            events = [
                e
                for e in events
                if (MAEDataset._instance_type(e["file_name"]) in instance_types)
            ]

        with Parallel(n_jobs) as p:
            return p(delayed(raw_mapper)(e) for e in events)

    def gather(transformed_events):
        Dataset = namedtuple("Dataset", ["X", "y", "g", "g_class"])

        X = pandas.concat([e[0] for e in transformed_events], axis="index").values
        y = pandas.concat([e[1] for e in transformed_events], axis="index").values
        sizes = [e[1].size for e in transformed_events]

        g = np.repeat(np.arange(len(sizes)), sizes)
        g_class = np.array([e[2] for e in transformed_events])

        return Dataset(X=X, y=y, g=g, g_class=g_class)

    def _make_set(self, events=None):
        """Loads all instances of target classes from the desired types,
           transforming the raw data to obtain its features by calling the
           *feature_mapper()* method for each instance.

        * Parameters:
            - **events**: [LIST] - Optional list of preloaded events
        """

        # load if not preloaded
        if events is None:
            events = MAEDataset.load_events(self.root_dir)

        # filter events
        def _should_keep(e):
            return (
                MAEDataset._instance_type(e["file_name"]) in self.instance_types
            ) and (e["event_type"] in self.tgt_events)

        events = [e for e in events if _should_keep(e)]

        feature_names = []
        X = []
        y = []
        g_len = []
        g_class = []

        with Parallel(self.n_jobs) as p:
            for x_, y_, et_ in p(
                delayed(self.feature_mapper)(event) for event in events
            ):
                # save feature names to check for consistency
                feature_names.append(x_.columns)

                # save features and labels
                X.append(np.double(x_.values))
                y.append(np.uint8(y_.values.ravel()))

                # save length of each event output
                g_len.append(y_.size)

                # save origin of each event
                g_class.append(et_)

        # feature name consistency check
        assert all(np.all(feature_names[0] == c) for c in feature_names)

        # glob features and labels
        self.X = np.concatenate(X, axis=0)
        self.y = np.concatenate(y, axis=0).ravel()

        # convert to np
        self.g_len = np.array(g_len)
        self.g_class = np.array(g_class)
        self.g = np.repeat(np.arange(self.g_len.size), self.g_len)

        self.feature_names = np.array(feature_names[0].to_list())
