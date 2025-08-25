import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification

from ThreeWToolkit.core.enums import ModelTypeEnum
from ThreeWToolkit.models.sklearn_models import SklearnModels, SklearnModelsConfig
from ThreeWToolkit.metrics import _classification


class TestSklearnModels:
    """
    Unit tests for the SklearnModels class.
    """

    @pytest.fixture
    def binary_data(self):
        """Provides simple binary classification data."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 0, 1, 1])
        return X, y

    @pytest.fixture
    def multiclass_data(self):
        """Provides multiclass classification data."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42,
        )
        return X, y

    def test_initialization_with_random_seed(self):
        """Tests that a model is correctly instantiated with a random_seed."""
        config = SklearnModelsConfig(
            model_type=ModelTypeEnum.DECISION_TREE,
            random_seed=123,
            model_params={"max_depth": 2},
        )
        model = SklearnModels(config)
        assert isinstance(model.model, DecisionTreeClassifier)
        assert model.model.get_params()["random_state"] == 123
        assert model.model.get_params()["max_depth"] == 2

    def test_initialization_without_random_state(self):
        """Tests the __init__ path for a model without a 'random_state' parameter."""
        config = SklearnModelsConfig(model_type=ModelTypeEnum.KNN, random_seed=42)
        model = SklearnModels(config)
        assert isinstance(model.model, KNeighborsClassifier)

    def test_train_and_predict(self, binary_data):
        """Tests the basic .fit() and .predict() flow."""
        X, y = binary_data
        config = SklearnModelsConfig(
            model_type=ModelTypeEnum.LOGISTIC_REGRESSION, random_seed=42
        )
        model = SklearnModels(config)
        model.fit(X, y)
        predictions = model.predict(X)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(y),)

    def test_evaluate_with_binary_classification(self, binary_data):
        """
        Tests that evaluate() correctly processes a list of metric functions
        with binary classification data as input.
        """
        X, y = binary_data
        config = SklearnModelsConfig(
            model_type=ModelTypeEnum.LOGISTIC_REGRESSION, random_seed=42
        )
        model = SklearnModels(config)
        model.fit(x=X, y=y)
        metrics_to_run = [
            _classification.accuracy_score,
            _classification.roc_auc_score,
            _classification.average_precision_score,
        ]
        results = model.evaluate(x=X, y=y, metrics=metrics_to_run)
        assert isinstance(results, dict)
        assert "accuracy_score" in results
        assert "roc_auc_score" in results
        assert "average_precision_score" in results
        assert (
            results["accuracy_score"] is not None
            and 0.0 <= results["accuracy_score"] <= 1.0
        )

    def test_evaluate_with_multiclass_classification(self, multiclass_data):
        """
        Tests that evaluate() correctly handles multiclass classification.
        """
        X, y = multiclass_data
        config = SklearnModelsConfig(
            model_type=ModelTypeEnum.RANDOM_FOREST, random_seed=42
        )
        model = SklearnModels(config)
        model.fit(x=X, y=y)
        metrics_to_run = [_classification.accuracy_score, _classification.roc_auc_score]
        results = model.evaluate(x=X, y=y, metrics=metrics_to_run)
        assert "roc_auc_score" in results
        assert (
            results["roc_auc_score"] is not None
            and 0.0 <= results["roc_auc_score"] <= 1.0
        )

    def test_evaluate_skips_metrics_that_need_proba(self, binary_data):
        """
        Tests that evaluate() correctly returns None for metrics like roc_auc_score
        when the model does not support predict_proba.
        """
        X, y = binary_data
        config = SklearnModelsConfig(model_type=ModelTypeEnum.SVM, random_seed=42)
        model = SklearnModels(config)
        model.fit(x=X, y=y)
        metrics_to_run = [_classification.accuracy_score, _classification.roc_auc_score]
        results = model.evaluate(x=X, y=y, metrics=metrics_to_run)
        assert "roc_auc_score" in results
        assert results["roc_auc_score"] is None
        assert "accuracy_score" in results
        assert results["accuracy_score"] is not None

    def test_get_and_set_params(self):
        """Tests the .get_params() and .set_params() methods."""
        config = SklearnModelsConfig(
            model_type=ModelTypeEnum.DECISION_TREE,
            random_seed=42,
            model_params={"max_depth": 3},
        )
        model = SklearnModels(config)
        # Use the underlying sklearn model's get_params for assertions
        params = model.model.get_params()
        assert params["max_depth"] == 3
        model.set_params(max_depth=10)
        params2 = model.model.get_params()
        assert params2["max_depth"] == 10

    def test_save_and_load(self, binary_data, tmp_path):
        """Tests that a model can be saved and loaded correctly."""
        X, y = binary_data
        config = SklearnModelsConfig(
            model_type=ModelTypeEnum.RANDOM_FOREST, random_seed=42
        )
        original_model = SklearnModels(config)
        original_model.fit(x=X, y=y)
        model_path = tmp_path / "model.pkl"
        original_model.save(str(model_path))
        loaded_model = SklearnModels.load(str(model_path), config)
        original_preds = original_model.predict(x=X)
        loaded_preds = loaded_model.predict(x=X)
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_predict_proba_and_unsupported_model_error(self, binary_data):
        """Tests .predict_proba() success and failure paths."""
        X, y = binary_data
        # Success path
        config_lr = SklearnModelsConfig(
            model_type=ModelTypeEnum.LOGISTIC_REGRESSION, random_seed=42
        )
        model_lr = SklearnModels(config_lr)
        model_lr.fit(X, y)
        probabilities = model_lr.predict_proba(X)
        assert probabilities.shape == (len(y), 2)
        # Failure path
        config_svc = SklearnModelsConfig(model_type=ModelTypeEnum.SVM, random_seed=42)
        model_svc = SklearnModels(config_svc)
        model_svc.fit(X, y)
        with pytest.raises(NotImplementedError):
            model_svc.predict_proba(X)

    def test_invalid_metric_raises(self, binary_data):
        """Test that passing an unsupported metric raises ValueError."""
        X, y = binary_data
        config = SklearnModelsConfig(
            model_type=ModelTypeEnum.LOGISTIC_REGRESSION, random_seed=42
        )
        model = SklearnModels(config)
        model.fit(X, y)

        def dummy_metric(y_true, y_pred):
            return 0.0

        with pytest.raises(ValueError):
            model.evaluate(X, y, [dummy_metric])

    def test_save_invalid_extension_raises(self, binary_data, tmp_path):
        """Test that saving with invalid extension raises ValueError."""
        X, y = binary_data
        config = SklearnModelsConfig(
            model_type=ModelTypeEnum.RANDOM_FOREST, random_seed=42
        )
        model = SklearnModels(config)
        model.fit(X, y)
        bad_path = tmp_path / "model.txt"
        with pytest.raises(ValueError):
            model.save(str(bad_path))
