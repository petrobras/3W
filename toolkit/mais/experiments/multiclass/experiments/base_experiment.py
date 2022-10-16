import numpy as np
import sklearn.metrics


class BaseExperiment:
    """Basic Experiment interface, with sane defaults"""

    def __init__(self):
        # use all tags
        self.selected_tags = MAEDataset.TAG_NAMES

        # drop hand-drawn instances
        self.instance_types = ["real", "simulated"]

        # use all classes
        self.tgt_events = MAEDataset.KNOWN_CLASSES
        self.num_classes = len(self.tgt_events)

    def raw_transform(self, event):
        """primary expert feature extraction from raw to processed data"""
        raise NotImplementedError()

    def balance(self, X, y, g, g_class):
        """by default, sample self.normal_balance * pseudo_normal_count true normals from X"""
        if not hasattr(self, "normal_balance"):
            return X, y, g
        if self.normal_balance == 0:
            return X, y, g

        # indices of samples from target events
        (tgt_idx,) = np.nonzero(g_class[g] != 0)

        # pseudo normal count
        pseudo_normal_count = np.count_nonzero(y[tgt_idx] == 0)

        # indices of samples from normal events
        (normal_idx,) = np.nonzero(g_class[g] == 0)
        normal_count = normal_idx.size

        balanced_count = min(self.normal_balance * pseudo_normal_count, normal_count)
        normal_idx = np.random.choice(normal_idx, balanced_count, replace=False)

        # compose final mask
        selected_idx = np.concatenate([normal_idx, tgt_idx])
        selected_idx.sort()

        # filter samples
        X = X[selected_idx]
        y = y[selected_idx]
        g = g[selected_idx]

        return X, y, g

    def fit(self, X, y):
        """fit preprocess pipeline to train data"""
        raise NotImplementedError()

    def transform(self, X, y):
        """apply preprocessing pipeline to data"""
        return X, y

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)

    def metric_name(self):
        """returns metric name as a string. if sklearn has that scorer as a name can be used"""
        raise NotImplementedError()

    def metric_scorer(self):
        """returns a metric evaluation callback for RandomForest, usually sklearn `make_scorer'"""
        return sklearn.metrics.get_scorer(self.metric_name())

    def metric_lgbm(self):
        """by default, wrap an sklearn metric to the lgbm feval format"""
        name = self.metric_name()
        scorer = sklearn.metrics.get_scorer(name)
        _func = scorer._score_func
        _sign = scorer._sign == 1

        def _f(y_true, y_score):
            y_pred = y_score.reshape(-1, y_true.shape[0]).argmax(0)
            return name, _func(y_true, y_pred), _sign

        return _f
