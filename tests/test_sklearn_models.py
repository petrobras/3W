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
            n_samples=100, n_features=10, n_informative=5,
            n_classes=3, n_clusters_per_class=1, random_state=42
        )
        return X, y

    def test_initialization_with_random_seed(self):
        """Tests that a model is correctly instantiated with a random_seed."""
        config = SklearnModelsConfig(
            model_type=ModelTypeEnum.DECISION_TREE,
            random_seed=123,
            model_params={"max_depth": 2}
        )
        model = SklearnModels(config)
        
        assert isinstance(model.model, DecisionTreeClassifier)
        assert model.model.get_params()["random_state"] == 123
        assert model.model.get_params()["max_depth"] == 2
        
    def test_initialization_without_random_state(self):
        """Tests the __init__ path for a model without a 'random_state' parameter."""
        config = SklearnModelsConfig(model_type=ModelTypeEnum.KNN)
        model = SklearnModels(config)
        assert isinstance(model.model, KNeighborsClassifier)

    def test_train_and_predict(self, binary_data):
        """Tests the basic .train() and .predict() flow."""
        X, y = binary_data
        config = SklearnModelsConfig(model_type=ModelTypeEnum.LOGISTIC_REGRESSION)
        model = SklearnModels(config)
        
        model.train(X, y)
        predictions = model.predict(X)
        
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(y),)

    def test_evaluate_with_binary_classification(self, binary_data):
        """
        Tests that evaluate() correctly processes a list of metric functions
        with binary classification data as input.
        """
        X, y = binary_data
        config = SklearnModelsConfig(model_type=ModelTypeEnum.LOGISTIC_REGRESSION)
        model = SklearnModels(config)
        model.train(x=X, y=y)
        
        # Define the list of metrics to calculate
        metrics_to_run = [
            _classification.accuracy_score,
            _classification.roc_auc_score,
            _classification.average_precision_score
        ]
        
        results = model.evaluate(x=X, y=y, metrics=metrics_to_run)
        
        # Assert that the output dictionary has the correct keys and values
        assert isinstance(results, dict)
        assert "accuracy_score" in results
        assert "roc_auc_score" in results
        assert "average_precision_score" in results
        assert 0.0 <= results["accuracy_score"] <= 1.0
        
    def test_evaluate_with_multiclass_classification(self, multiclass_data):
        """
        Tests that evaluate() correctly handles multiclass classification.
        """
        X, y = multiclass_data
        # Use a model that supports predict_proba for multiclass evaluation
        config = SklearnModelsConfig(model_type=ModelTypeEnum.RANDOM_FOREST)
        model = SklearnModels(config)
        model.train(x=X, y=y)

        metrics_to_run = [
            _classification.accuracy_score,
            _classification.roc_auc_score
        ]

        results = model.evaluate(x=X, y=y, metrics=metrics_to_run)

        assert "roc_auc_score" in results
        assert 0.0 <= results["roc_auc_score"] <= 1.0
    
    def test_evaluate_skips_metrics_that_need_proba(self, binary_data):
        """
        Tests that evaluate() correctly returns None for metrics like roc_auc_score
        when the model does not support predict_proba.
        """
        X, y = binary_data
        # A standard SVC does not have .predict_proba()
        config = SklearnModelsConfig(model_type=ModelTypeEnum.SVM)
        model = SklearnModels(config)
        model.train(x=X, y=y)
        
        metrics_to_run = [
            _classification.accuracy_score,
            _classification.roc_auc_score # This one should result in None
        ]
        
        results = model.evaluate(x=X, y=y, metrics=metrics_to_run)
        
        # Assert that the key exists and its value is None.
        assert "roc_auc_score" in results
        assert results["roc_auc_score"] is None
        
        # Assert that the other metric was still calculated correctly.
        assert "accuracy_score" in results
        assert results["accuracy_score"] is not None

    def test_get_and_set_params(self):
        """Tests the .get_params() and .set_params() methods."""
        config = SklearnModelsConfig(model_type=ModelTypeEnum.DECISION_TREE, model_params={"max_depth": 3})
        model = SklearnModels(config)

        assert model.get_params()["max_depth"] == 3
        model.set_params(max_depth=10)
        assert model.get_params()["max_depth"] == 10
    
    def test_save_and_load(self, binary_data, tmp_path):
        """Tests that a model can be saved and loaded correctly."""
        X, y = binary_data
        config = SklearnModelsConfig(model_type=ModelTypeEnum.RANDOM_FOREST)
        
        # Train and save
        original_model = SklearnModels(config)
        original_model.train(x=X, y=y)
        model_path = tmp_path / "model.pkl"
        original_model.save(str(model_path))

        # Load and verify
        loaded_model = SklearnModels.load(str(model_path), config)
        original_preds = original_model.predict(x=X)
        loaded_preds = loaded_model.predict(x=X)
        
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_predict_proba_and_unsupported_model_error(self, binary_data):
        """Tests .predict_proba() success and failure paths."""
        X, y = binary_data

        # Success path
        config_lr = SklearnModelsConfig(model_type=ModelTypeEnum.LOGISTIC_REGRESSION)
        model_lr = SklearnModels(config_lr)
        model_lr.train(X, y)
        probabilities = model_lr.predict_proba(X)
        assert probabilities.shape == (len(y), 2)

        # Failure path
        config_svc = SklearnModelsConfig(model_type=ModelTypeEnum.SVM)
        model_svc = SklearnModels(config_svc)
        model_svc.train(X, y)
        with pytest.raises(NotImplementedError):
            model_svc.predict_proba(X)
