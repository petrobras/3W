""" definitions for experiment 3 module
    Multiclass classification
    Statistical features
    Most-Recent Label strategy
    Drop windows with NaN label
"""
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from mais.data.feature_mappers import TorchStatisticalFeatureMapper
from mais.data.label_mappers import TorchMulticlassMRLStrategy

from .base_experiment import BaseExperiment
from mais.data.dataset import MAEDataset


MAX_WINDOW = 1000


def sample(trial, *args, **kwargs):
    return Experiment(
        window_size=trial.suggest_int("window_size", 100, 1000, step=100),
        stride=trial.suggest_int("stride", 10, 10),
        n_components=trial.suggest_float("n_components", 0.9, 1.0),
        normal_balance=trial.suggest_int("normal_balance", 1, 10, step=1),
    )


class Experiment(BaseExperiment):
    """the docstring"""

    def __init__(
        self,
        window_size,
        stride,
        n_components,
        normal_balance,
        *args,
        **kwargs,
    ):
        super().__init__()

        # save params
        self.window_size = window_size
        self.n_components = n_components
        self.stride = stride
        self.normal_balance = normal_balance

        self._init_raw_mappers()
        self._init_preprocessor()

    def raw_transform(self, event, transient_only=True, no_nans=True):
        # filter tags and set zeros to nans
        tags = event["tags"][self.selected_tags].replace(0, np.nan)
        labels = event["labels"]
        event_type = event["event_type"]

        if transient_only and MAEDataset.TRANSIENT_CLASS[event_type]:
            transients = labels.values != event_type
            tags = tags[transients]
            labels = labels[transients]

        features = self._feature_mapper(tags, event_type)
        labels = self._label_mapper(labels, event_type)

        # drop windows with NaN label
        if no_nans:
            notnan = labels.notna()
            features = features[notnan]
            labels = labels[notnan]

        return features, labels, event_type

    def metric_name(self):
        return "balanced_accuracy"

    def fit(self, X, y):
        self._label_encoder.fit(y)
        X = self._scaler.fit_transform(X)
        X = self._imputer.fit_transform(X)
        self._pca.fit(X)

    def fit_transform(self, X, y):
        self._label_encoder.fit(y)
        X = self._scaler.fit_transform(X)
        X = self._imputer.fit_transform(X)
        return self._pca.fit_transform(X), y

    def transform(self, X, y=None):
        y = self._label_encoder.transform(y)
        X = self._scaler.transform(X)
        X = self._imputer.transform(X)
        X = self._pca.transform(X)
        return X, y

    def _init_raw_mappers(self):
        offset = MAX_WINDOW - self.window_size
        self._feature_mapper = TorchStatisticalFeatureMapper(
            window_size=self.window_size, stride=self.stride, offset=offset
        )
        self._label_mapper = TorchMulticlassMRLStrategy(
            window_size=self.window_size,
            stride=self.stride,
            offset=offset,
        )

    def _init_preprocessor(self):
        self._label_encoder = LabelEncoder()
        # z-score
        self._scaler = StandardScaler()
        # remove nans
        self._imputer = SimpleImputer(strategy="mean")
        # pca
        self._pca = PCA(n_components=self.n_components, whiten=True)
