# coding: utf-8
""" Basic dataset definitions """

from collections import namedtuple

import numpy as np
import pandas

from joblib import Parallel, delayed

from .base import (
    CLASS,
    COLUMNS_DATA_FILES,
    EVENT_NAMES,
    LABELS_DESCRIPTIONS,
    NORMAL_LABEL,
    PATH_DATASET,
    TRANSIENT_OFFSET,
    VARS,
    load_3w_dataset,
)


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
        (raw_tags: DataFrame,  raw_labels: DataFrame) ->
        (feature: DataFrame x label: DataFrame) --
        deals with transforming the raw data from a single event,
        to be used during training/evaluation. Outputs from each event concatenated
        afterwards.

        - **n_jobs: INT** -- number of processes to use

        - **events**

    * Important fields:

        - **self.X: [N_SAMPLE x N_FEATURES]** ndarray created by concatenating every feature output from feature mapper

        - **self.Y: [N_SAMPLE]** ndarray created by concatenating every label output from feature mapper

        - **self.feature_names: [N_FEATURES]** ndarray as output by the feature mapper

        - **self.g_len: [N_EVENTS]** number of samples output by the feature mapper for each event

        - **self.g_class: [N_EVENTS]** list denoting the origin of the event


    * Methods:

        - **_is_instance_type(type_)**

        - **_make_set()**

        - **process(fname, event_type)**

    """

    # Tag corresponding to instance label
    LABEL_NAME = "class"

    # Tag corresponding to index
    INDEX_NAME = "timestamp"

    # Fault description
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

    # List of used classes
    KNOWN_CLASSES = list(CLASS_NAMES.keys())

    # Transient properties of events
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
    TAG_NAMES = VARS

    def __init__(
        self,
        transformed_events=None,
        events=None,  # either pass in preloaded events or the root directory
        data_type="real",  # Type of data to load: 'real', 'simulated', or 'imputed'
        tgt_events=[],  # which event_types to load
        feature_mapper=tuple,  # transformer from event to features
        n_jobs=-1,
    ):
        """
        Load and process dataset using supplied strategies.
        """

        # save parameters
        self.data_type = data_type
        self.tgt_events = tgt_events
        self.n_jobs = n_jobs
        self.feature_mapper = feature_mapper

        # Call the heavy load _make_set passing the (maybe Null) events
        self._make_set(events)


    def load_events(self, data_type="real", n_jobs=-1):
        """
        Load events from the 3W Dataset 2.0.

        * Parameters:
            - **data_type: STRING** - Type of data to load ('real', 'simulated', or 'imputed')

        * Returns:
            - **events**: [LIST] - List of loaded events
        """

        def _read(df):
            """
            Return a dict with the summary of a target.

            * Parameters:
                - **df: pandas.DataFrame** - DataFrame with the 3W Dataset 2.0 data.

            * Returns:
                - **dict**: Dict with the summary of a target.
            """

            tags = df[MAEDataset.TAG_NAMES]
            labels = df[MAEDataset.LABEL_NAME]

            return {
                "tags": tags,
                "labels": labels,
                "event_type": df["label"].iloc[0],
            }

        # Load the 3W Dataset 2.0
        df = load_3w_dataset(data_type=data_type)

        # Split the DataFrame by event type
        events = [_read(df[df["label"] == event_type]) for event_type in df["label"].unique()]

        return events

    def transform_events(
        self, events, raw_mapper, tgt_events=None, n_jobs=-1
    ):
        """
        Apply raw_mapper to list of events, filtering by target events
        """
        if tgt_events is not None:
            events = [e for e in events if (e["event_type"] in tgt_events)]

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
        """
        Loads all instances of target classes, transforming the raw data to obtain its
        features by calling the *feature_mapper()* method for each instance.

        * Parameters:
            - **events**: [LIST] - Optional list of preloaded events
        """

        # load if not preloaded
        if events is None:
            events = MAEDataset.load_events(self, self.data_type)

        # filter events
        events = [e for e in events if (e["event_type"] in self.tgt_events)]

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

        self.feature
        