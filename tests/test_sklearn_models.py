from ThreeWToolkit.assessment.strategies.sklearn_prediction_strategy import (
    SklearnPredictionStrategy,
)
import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification

from ThreeWToolkit.core.enums import ModelTypeEnum
from ThreeWToolkit.models.sklearn_models import SklearnModels, SklearnModelsConfig


@pytest.fixture
def base_config():
    """Creates a default configuration for sklearn models."""
    return SklearnModelsConfig(
        model_type=ModelTypeEnum.LOGISTIC_REGRESSION, random_seed=42
    )


@pytest.fixture
def base_model(base_config):
    """Creates a SklearnModels instance using the base configuration."""
    return SklearnModels(base_config)


class TestSklearnModels:
    """Unit tests for the SklearnModels wrapper class.

    This test suite verifies correct initialization, parameter handling,
    model persistence, and prediction strategy integration for sklearn models.
    """

    @pytest.fixture
    def binary_data(self):
        """Provides a small binary classification dataset.

        Returns:
            tuple[np.ndarray, np.ndarray]: Feature matrix and target labels.
        """
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 0, 1, 1])
        return X, y

    @pytest.fixture
    def multiclass_data(self):
        """Provides a multiclass classification dataset.

        Returns:
            tuple[np.ndarray, np.ndarray]: Feature matrix and target labels
            generated with sklearn.make_classification.
        """
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42,
        )
        return X, y

    def test_model_name(self, base_model):
        """Ensures the model_name property matches the underlying sklearn class."""
        assert base_model.model_name == base_model.model_class.__class__.__name__

    def test_forward_returns_none(self, base_model):
        """Ensures the forward method returns None for sklearn-based models.

        The forward interface exists for compatibility with other model types
        but is not used for sklearn estimators.
        """
        x = np.array([1, 2, 3])

        result = base_model.forward(x)

        assert result is None

    def test_initialization_with_random_seed(self):
        """Tests that a model is correctly instantiated with a random_seed."""
        config = SklearnModelsConfig(
            model_type=ModelTypeEnum.DECISION_TREE,
            random_seed=123,
            model_params={"max_depth": 2},
        )
        model = SklearnModels(config)
        assert isinstance(model.model_class, DecisionTreeClassifier)
        assert model.model_class.get_params()["random_state"] == 123
        assert model.model_class.get_params()["max_depth"] == 2

    def test_initialization_without_random_state(self):
        """Tests the __init__ path for a model without a 'random_state' parameter."""
        config = SklearnModelsConfig(model_type=ModelTypeEnum.KNN, random_seed=42)
        model = SklearnModels(config)
        assert isinstance(model.model_class, KNeighborsClassifier)

    def test_get_and_set_params(self):
        """Ensures model parameters can be retrieved and updated correctly."""
        config = SklearnModelsConfig(
            model_type=ModelTypeEnum.DECISION_TREE,
            random_seed=42,
            model_params={"max_depth": 3},
        )
        model = SklearnModels(config)

        params = model.get_params()
        assert params["max_depth"] == 3

        model.set_params(max_depth=10)
        params2 = model.get_params()
        assert params2["max_depth"] == 10

    def test_save_and_load(self, binary_data, tmp_path):
        """Ensures trained models can be saved and loaded correctly.

        After loading, predictions should match the original model.
        """
        X, y = binary_data
        config = SklearnModelsConfig(
            model_type=ModelTypeEnum.RANDOM_FOREST,
            random_seed=42,
        )

        model = SklearnModels(config)
        model.model_class.fit(X, y)

        path = tmp_path / "model.pkl"
        model.save(path)

        new_model = SklearnModels(config)
        new_model.load(path)

        np.testing.assert_array_equal(
            model.model_class.predict(X),
            new_model.model_class.predict(X),
        )

    def test_save_invalid_extension_raises(self, binary_data, tmp_path):
        """Ensures saving with an invalid file extension raises an error."""
        X, y = binary_data
        config = SklearnModelsConfig(
            model_type=ModelTypeEnum.RANDOM_FOREST,
            random_seed=42,
        )
        model = SklearnModels(config)
        model.model_class.fit(X, y)

        bad_path = tmp_path / "model.txt"
        with pytest.raises(ValueError):
            model.save(bad_path)

    def test_get_prediction_strategy(self, base_model):
        """Ensures the correct prediction strategy is returned."""
        assert base_model.get_prediction_strategy() is SklearnPredictionStrategy
