"""Tests for sklearn_models module."""

import tempfile
from typing import Type, Any

from ThreeWToolkit.models.sklearn_models import SklearnModelsConfig, SklearnModels
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

model_configs: dict[Type, dict[str, Any]] = {
    LinearRegression: {
        "fit_intercept": True,
    },
    LogisticRegression: {"C": 1.0, "solver": "lbfgs"},
    Ridge: {"alpha": 1.0, "solver": "auto"},
    Lasso: {"alpha": 1.0, "max_iter": 1000},
    SVC: {"C": 1.0, "kernel": "rbf"},
    DecisionTreeClassifier: {"criterion": "gini", "max_depth": None},
    KNeighborsClassifier: {"n_neighbors": 5, "algorithm": "auto"},
    RandomForestClassifier: {"n_estimators": 100, "max_depth": None},
    GradientBoostingClassifier: {"n_estimators": 100, "learning_rate": 0.1},
    GaussianNB: {"var_smoothing": 1e-9},
}


class TestSklearnModelsConfig:
    """Tests for SklearnModelsConfig validation and behavior."""

    def test_invalid_model_type(self):
        """Invalid model types should raise a validation error."""

        class NotAModel:
            pass

        with pytest.raises(ValueError):
            SklearnModelsConfig(model_type=NotAModel).build()  # type: ignore

    @pytest.mark.parametrize("model_type", model_configs.keys())
    def test_all_model_types(self, model_type: Type):
        """All supported model types should work."""

        try:  # try to build the config with default parameters
            _ = SklearnModelsConfig(model_type=model_type).build()
        except Exception as e:
            pytest.fail(f"Model type {model_type} raised an exception: {e}")

        try:  # try to build the config with specified parameters
            model = SklearnModelsConfig(
                model_type=model_type, model_params=model_configs[model_type]
            ).build()
        except Exception as e:
            pytest.fail(f"Model type {model_type} raised an exception: {e}")

        # Test saving and loading the model
        with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp:
            try:
                model.save(tmp.name)
            except Exception as e:
                pytest.fail(f"Saving model type {model_type} raised an exception: {e}")
            try:
                loaded_model = SklearnModels.load(tmp.name)
            except Exception as e:
                pytest.fail(
                    f"Saving/loading model type {model_type} raised an exception: {e}"
                )
            assert (
                loaded_model.get_params() == model.get_params()
            ), f"Loaded model parameters do not match original for {model_type}"
