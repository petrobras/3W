""" definitions for experiment 3 module
    Multiclass classification
    Statistical+Wavelet features
    Random forest feature selection
    Most-Recent Label strategy
    Drop windows with NaN label
"""
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# from sklearn.feature_selection import f_classif, SelectPercentile

from mais.data.feature_mappers import (
    TorchWaveletFeatureMapper,
    TorchStatisticalFeatureMapper,
    MixedMapper,
)
from mais.data.label_mappers import TorchMulticlassMRLStrategy

from .base_experiment import BaseExperiment
from mais.data.dataset import MAEDataset


MAX_LEVEL = 10


def sample(trial, *args, **kwargs):
    # set default stride to 10
    stride = trial.user_attrs.get("stride") or 10

    return Experiment(
        level=trial.suggest_int("level", 4, MAX_LEVEL, step=1),
        importance_percentile=trial.suggest_float("importance_percentile", 0.1, 1.0),
        normal_balance=trial.suggest_int("normal_balance", 1, 10, step=1),
        stride=stride,
    )


class Experiment(BaseExperiment):
    """the docstring"""

    def __init__(
        self,
        level,
        stride,
        importance_percentile,
        normal_balance,
        *args,
        **kwargs,
    ):
        super().__init__()

        # save params
        self.window_size = 2**level
        self.level = level
        self.importance_percentile = importance_percentile
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
        y = self._label_encoder.fit_transform(y)
        X = self._scaler.fit_transform(X)
        X = self._imputer.fit_transform(X)
        self._fit_selector(X, y)
        return self

    def fit_transform(self, X, y):
        y = self._label_encoder.fit_transform(y)
        X = self._scaler.fit_transform(X)
        X = self._imputer.fit_transform(X)
        self._fit_selector(X, y)
        return self._select_features(X), y

    def transform(self, X, y):
        y = self._label_encoder.transform(y)
        X = self._scaler.transform(X)
        X = self._imputer.transform(X)
        return self._select_features(X), y

    def _init_raw_mappers(self):
        offset = 2**MAX_LEVEL - self.window_size
        wavelet_mapper = TorchWaveletFeatureMapper(
            level=self.level, stride=self.stride, offset=offset
        )

        stats_mapper = TorchStatisticalFeatureMapper(
            window_size=2**self.level, stride=self.stride, offset=offset
        )

        self._feature_mapper = MixedMapper(stats_mapper, wavelet_mapper)

        self._label_mapper = TorchMulticlassMRLStrategy(
            window_size=self.window_size,
            stride=self.stride,
            offset=offset,
        )

    def _fit_selector(self, X, y):
        self._forest.fit(X, y)

        importances = self._forest.feature_importances_
        feature_order = np.argsort(-importances)
        cumulative_importance = np.cumsum(importances[feature_order])
        num_feats = np.searchsorted(cumulative_importance, self.importance_percentile)
        self.selected_features = feature_order[:num_feats]

    def _select_features(self, X):
        return X[:, self.selected_features]

    def _init_preprocessor(self):
        # preprocessing pipeline
        self._scaler = StandardScaler()  # recenter
        self._imputer = SimpleImputer(strategy="mean")  # mean should be 0 anyway
        self._forest = RandomForestClassifier(n_estimators=100, n_jobs=-1)

        # label encoder for multiclass
        self._label_encoder = LabelEncoder()
