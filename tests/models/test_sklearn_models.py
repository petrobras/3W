"""Tests for sklearn_models module."""

import pytest
from ThreeWToolkit.constants import CHECKPOINT_DIR
from ThreeWToolkit.models.sklearn_models import (
    SklearnModels,
    SklearnModelsConfig,
    SKLEARN_MODELS,
)
from ThreeWToolkit.core.enums import ModelTypeEnum
from sklearn.base import BaseEstimator
import numpy as np


class TestSklearnModelsConfig:
    """Tests for SklearnModelsConfig validation and behavior."""

    def test_valid_config_logistic_regression(self):
        """Config for logistic regression should work."""
        config = SklearnModelsConfig(model_type=ModelTypeEnum.LOGISTIC_REGRESSION)
        assert config.model_type == ModelTypeEnum.LOGISTIC_REGRESSION
        assert config.model_params == {}

    def test_valid_config_with_params(self):
        """Config with model parameters should work."""
        config = SklearnModelsConfig(
            model_type=ModelTypeEnum.RANDOM_FOREST,
            model_params={"n_estimators": 100, "max_depth": 10},
        )
        assert config.model_params["n_estimators"] == 100
        assert config.model_params["max_depth"] == 10

    @pytest.mark.parametrize(
        "model_type",
        [
            ModelTypeEnum.LOGISTIC_REGRESSION,
            ModelTypeEnum.DECISION_TREE,
            ModelTypeEnum.RANDOM_FOREST,
            ModelTypeEnum.SVM,
            ModelTypeEnum.KNN,
            ModelTypeEnum.NAIVE_BAYES,
            ModelTypeEnum.GRADIENT_BOOSTING,
        ],
    )
    def test_all_model_types(self, model_type):
        """All supported model types should work."""
        config = SklearnModelsConfig(model_type=model_type)
        assert config.model_type == model_type

    def test_target_returns_sklearn_models_class(self):
        """_target should return SklearnModels class."""
        config = SklearnModelsConfig(model_type=ModelTypeEnum.LOGISTIC_REGRESSION)
        assert config._target == SklearnModels

    def test_random_seed_default(self):
        """Random seed should have default value."""
        config = SklearnModelsConfig(model_type=ModelTypeEnum.LOGISTIC_REGRESSION)
        assert config.random_seed is not None

    def test_random_seed_custom(self):
        """Custom random seed should be set."""
        config = SklearnModelsConfig(
            model_type=ModelTypeEnum.LOGISTIC_REGRESSION, random_seed=123
        )
        assert config.random_seed == 123


class TestSklearnModels:
    """Tests for SklearnModels wrapper."""

    @pytest.fixture
    def logistic_config(self):
        """Provide a logistic regression config."""
        return SklearnModelsConfig(model_type=ModelTypeEnum.LOGISTIC_REGRESSION)

    @pytest.fixture
    def random_forest_config(self):
        """Provide a random forest config with params."""
        return SklearnModelsConfig(
            model_type=ModelTypeEnum.RANDOM_FOREST,
            model_params={"n_estimators": 10, "max_depth": 3},
            random_seed=42,
        )

    def test_model_creation_logistic_regression(self, logistic_config):
        """Logistic regression model should be created."""
        model = SklearnModels(logistic_config)
        assert model.model_name == "LogisticRegression"
        assert model.model_class is not None

    def test_model_creation_random_forest(self, random_forest_config):
        """Random forest model should be created with params."""
        model = SklearnModels(random_forest_config)
        assert model.model_name == "RandomForestClassifier"
        params = model.get_params()
        assert params["n_estimators"] == 10
        assert params["max_depth"] == 3

    @pytest.mark.parametrize(
        "model_type,expected_name",
        [
            (ModelTypeEnum.LOGISTIC_REGRESSION, "LogisticRegression"),
            (ModelTypeEnum.DECISION_TREE, "DecisionTreeClassifier"),
            (ModelTypeEnum.RANDOM_FOREST, "RandomForestClassifier"),
            (ModelTypeEnum.SVM, "SVC"),
            (ModelTypeEnum.KNN, "KNeighborsClassifier"),
            (ModelTypeEnum.NAIVE_BAYES, "ComplementNB"),
            (ModelTypeEnum.GRADIENT_BOOSTING, "GradientBoostingClassifier"),
        ],
    )
    def test_all_model_names(self, model_type, expected_name):
        """All models should have correct model_name."""
        config = SklearnModelsConfig(model_type=model_type)
        model = SklearnModels(config)
        assert model.model_name == expected_name

    def test_random_state_set(self, random_forest_config):
        """Random state should be set from config."""
        model = SklearnModels(random_forest_config)
        params = model.get_params()
        assert params["random_state"] == 42

    def test_get_params(self, logistic_config):
        """get_params should return sklearn model parameters."""
        model = SklearnModels(logistic_config)
        params = model.get_params()
        assert isinstance(params, dict)
        assert "C" in params  # Logistic regression has C parameter

    def test_set_params(self, logistic_config):
        """set_params should update model parameters."""
        model = SklearnModels(logistic_config)
        model.set_params(C=0.5)
        params = model.get_params()
        assert params["C"] == 0.5

    def test_save_and_load(self, logistic_config):
        """Model should save and load correctly."""

        model = SklearnModels(logistic_config)
        model.set_params(C=0.5)

        filename = "test_sklearn_save_load.pkl"
        model.save(filename)

        new_model = SklearnModels(logistic_config)
        new_model.load(filename)

        assert new_model.get_params()["C"] == 0.5

        # Cleanup
        (CHECKPOINT_DIR / filename).unlink(missing_ok=True)

    def test_save_creates_file_in_checkpoint_dir(self, logistic_config):
        """Save should create file in CHECKPOINT_DIR."""

        model = SklearnModels(logistic_config)
        filename = "test_sklearn_checkpoint.pkl"
        model.save(filename)
        assert (CHECKPOINT_DIR / filename).exists()

        # Cleanup
        (CHECKPOINT_DIR / filename).unlink(missing_ok=True)


class TestSklearnModelsMapping:
    """Tests for SKLEARN_MODELS mapping."""

    def test_all_model_types_mapped(self):
        """All expected model types should be mapped."""
        expected_types = [
            ModelTypeEnum.LOGISTIC_REGRESSION,
            ModelTypeEnum.DECISION_TREE,
            ModelTypeEnum.RANDOM_FOREST,
            ModelTypeEnum.SVM,
            ModelTypeEnum.KNN,
            ModelTypeEnum.NAIVE_BAYES,
            ModelTypeEnum.GRADIENT_BOOSTING,
        ]
        for model_type in expected_types:
            assert model_type in SKLEARN_MODELS

    def test_mapping_returns_sklearn_classes(self):
        """All mapped values should be sklearn classes."""

        for model_type, model_class in SKLEARN_MODELS.items():
            instance = model_class()
            assert isinstance(instance, BaseEstimator)


class TestSklearnModelsIntegration:
    """Integration tests for sklearn models with data."""

    @pytest.fixture
    def sample_data(self):
        """Provide sample training data."""

        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        return X, y

    @pytest.mark.parametrize(
        "model_type",
        [
            ModelTypeEnum.LOGISTIC_REGRESSION,
            ModelTypeEnum.DECISION_TREE,
            ModelTypeEnum.RANDOM_FOREST,
            ModelTypeEnum.KNN,
        ],
    )
    def test_fit_and_predict(self, model_type, sample_data):
        """Models should fit and predict."""
        X, y = sample_data
        config = SklearnModelsConfig(model_type=model_type)
        model = SklearnModels(config)

        model.model_class.fit(X, y)
        predictions = model.model_class.predict(X)

        assert len(predictions) == len(y)
        assert all(p in [0, 1] for p in predictions)

    def test_probability_prediction(self, sample_data):
        """Models with probability support should predict probabilities."""
        X, y = sample_data
        config = SklearnModelsConfig(
            model_type=ModelTypeEnum.RANDOM_FOREST,
            model_params={"n_estimators": 10},
        )
        model = SklearnModels(config)
        model.model_class.fit(X, y)

        probs = model.model_class.predict_proba(X)
        assert probs.shape == (100, 2)
        assert all(abs(p.sum() - 1.0) < 1e-6 for p in probs)

    def test_multiclass_classification(self, sample_data):
        """Models should handle multi-class classification."""

        X, _ = sample_data
        y = np.random.randint(0, 5, 100)  # 5 classes

        config = SklearnModelsConfig(model_type=ModelTypeEnum.RANDOM_FOREST)
        model = SklearnModels(config)
        model.model_class.fit(X, y)

        predictions = model.model_class.predict(X)
        assert all(p in range(5) for p in predictions)

    def test_save_load_fitted_model(self, sample_data):
        """Fitted model should save and load with state preserved."""

        X, y = sample_data
        config = SklearnModelsConfig(model_type=ModelTypeEnum.LOGISTIC_REGRESSION)
        model = SklearnModels(config)
        model.model_class.fit(X, y)

        original_predictions = model.model_class.predict(X)

        filename = "test_sklearn_fitted.pkl"
        model.save(filename)

        new_config = SklearnModelsConfig(model_type=ModelTypeEnum.LOGISTIC_REGRESSION)
        new_model = SklearnModels(new_config)
        new_model.load(filename)

        loaded_predictions = new_model.model_class.predict(X)
        assert all(original_predictions == loaded_predictions)

        # Cleanup
        (CHECKPOINT_DIR / filename).unlink(missing_ok=True)
