""" definitions for experiment 3 module
    Multiclass classification
    Statistical features
    Most-Recent Label strategy
    Drop windows with NaN label
"""
import numpy as np

from sklearn.metrics import accuracy_score, get_scorer
from sklearn.preprocessing import StandardScaler
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
        # init base class
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

        # trim estabilished fault if has transient
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
        return "accuracy"

    def metric_rf(self):
        return get_scorer("accuracy")

    def metric_lgbm(self):
        def acc(preds, train_data):
            preds_ = np.argmax(np.reshape(preds, (self.num_classes, -1)), axis=0)
            return "accuracy", accuracy_score(train_data.get_label(), preds_), True

        return acc

    def fit(self, X, y=None):
        X = self._scaler.fit_transform(X)
        X = self._imputer.fit_transform(X)
        self._pca.fit(X)

    def transform(self, X, y=None):
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
        # z-score
        self._scaler = StandardScaler()
        # remove nans
        self._imputer = SimpleImputer(strategy="mean")
        # pca
        self._pca = PCA(n_components=self.n_components, whiten=True)
