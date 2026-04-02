"""Tests for SklearnPredictionStrategy."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from ThreeWToolkit.assessment.strategies.sklearn_prediction_strategy import (
    SklearnPredictionStrategy,
)
from ThreeWToolkit.core.enums import TaskTypeEnum


class MockSklearnModel:
    """Mock sklearn model wrapper for testing prediction strategy."""

    def __init__(self, predictions=None, probas=None, has_predict_proba=True):
        self.model_class = MagicMock()
        self.model_class.predict = MagicMock(
            return_value=predictions if predictions is not None else np.array([0, 1, 0])
        )
        if has_predict_proba:
            self.model_class.predict_proba = MagicMock(
                return_value=(
                    probas
                    if probas is not None
                    else np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
                )
            )
        else:
            del self.model_class.predict_proba


class TestSklearnPredictionStrategy:
    """Tests for SklearnPredictionStrategy.predict() method."""

    @pytest.fixture
    def strategy(self):
        """Provide a fresh prediction strategy instance."""
        return SklearnPredictionStrategy()

    @pytest.fixture
    def mock_model(self):
        """Provide a mock sklearn model."""
        return MockSklearnModel()

    def test_predict_returns_numpy_array(self, strategy, mock_model):
        """Predict should return a numpy array."""
        x = np.random.randn(5, 4)
        result = strategy.predict(mock_model, TaskTypeEnum.CLASSIFICATION, x=x)

        assert isinstance(result, np.ndarray)

    def test_predict_calls_model_predict(self, strategy, mock_model):
        """Predict should call the underlying model's predict method."""
        x = np.random.randn(5, 4)
        strategy.predict(mock_model, TaskTypeEnum.CLASSIFICATION, x=x)

        mock_model.model_class.predict.assert_called_once()

    def test_predict_raises_without_input(self, strategy, mock_model):
        """Predict should raise ValueError when x is not provided."""
        with pytest.raises(ValueError, match="Input data must be provided"):
            strategy.predict(mock_model, TaskTypeEnum.CLASSIFICATION)

    def test_predict_raises_for_model_without_predict(self, strategy):
        """Predict should raise when model lacks predict method."""
        bad_model = MagicMock()
        bad_model.model_class = MagicMock(spec=[])  # No predict method

        x = np.random.randn(5, 4)
        with pytest.raises(ValueError, match="must implement a 'predict' method"):
            strategy.predict(bad_model, TaskTypeEnum.CLASSIFICATION, x=x)

    def test_predict_with_classification_task(self, strategy, mock_model):
        """Predict should work with classification task type."""
        x = np.random.randn(3, 4)
        result = strategy.predict(mock_model, TaskTypeEnum.CLASSIFICATION, x=x)

        assert len(result) == 3

    def test_predict_with_regression_task(self, strategy):
        """Predict should work with regression task type."""
        predictions = np.array([1.5, 2.3, 0.8])
        model = MockSklearnModel(predictions=predictions)

        x = np.random.randn(3, 4)
        result = strategy.predict(model, TaskTypeEnum.REGRESSION, x=x)

        assert np.array_equal(result, predictions)

    def test_predict_with_none_task_type(self, strategy, mock_model):
        """Predict should work when task type is None."""
        x = np.random.randn(3, 4)
        result = strategy.predict(mock_model, task=None, x=x)

        assert isinstance(result, np.ndarray)

    def test_predict_with_dataframe_input(self, strategy, mock_model):
        """Predict should work with pandas DataFrame input."""
        x = pd.DataFrame(np.random.randn(5, 4), columns=["a", "b", "c", "d"])
        result = strategy.predict(mock_model, TaskTypeEnum.CLASSIFICATION, x=x)

        assert isinstance(result, np.ndarray)

    def test_predict_preserves_dataframe_structure(self, strategy, mock_model):
        """DataFrame input should be passed to model preserving structure."""
        x = pd.DataFrame(
            np.random.randn(5, 4),
            columns=["feature_1", "feature_2", "feature_3", "feature_4"],
        )
        strategy.predict(mock_model, TaskTypeEnum.CLASSIFICATION, x=x)

        # Verify the model received a DataFrame
        call_args = mock_model.model_class.predict.call_args
        passed_x = call_args[0][0]
        assert isinstance(passed_x, pd.DataFrame)

    def test_predict_with_series_input(self, strategy, mock_model):
        """Predict should work with pandas Series input."""
        x = pd.Series(np.random.randn(5), name="feature")
        result = strategy.predict(mock_model, TaskTypeEnum.CLASSIFICATION, x=x)

        assert isinstance(result, np.ndarray)


class TestSklearnPredictionStrategyPredictProba:
    """Tests for SklearnPredictionStrategy.predict_proba() method."""

    @pytest.fixture
    def strategy(self):
        """Provide a fresh prediction strategy instance."""
        return SklearnPredictionStrategy()

    def test_predict_proba_returns_numpy_array(self, strategy):
        """predict_proba should return a numpy array."""
        model = MockSklearnModel()
        x = np.random.randn(3, 4)
        result = strategy.predict_proba(model, x=x)

        assert isinstance(result, np.ndarray)

    def test_predict_proba_shape_binary(self, strategy):
        """predict_proba should return (N, 2) shape for binary classification."""
        probas = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5], [0.9, 0.1]])
        model = MockSklearnModel(probas=probas)

        x = np.random.randn(4, 3)
        result = strategy.predict_proba(model, x=x)

        assert result.shape == (4, 2)

    def test_predict_proba_shape_multiclass(self, strategy):
        """predict_proba should return (N, C) shape for multiclass classification."""
        probas = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]])
        model = MockSklearnModel(probas=probas)

        x = np.random.randn(3, 4)
        result = strategy.predict_proba(model, x=x)

        assert result.shape == (3, 3)

    def test_predict_proba_raises_without_input(self, strategy):
        """predict_proba should raise ValueError when x is not provided."""
        model = MockSklearnModel()

        with pytest.raises(ValueError, match="Input data must be provided"):
            strategy.predict_proba(model)

    def test_predict_proba_raises_for_model_without_predict_proba(self, strategy):
        """predict_proba should raise NotImplementedError for models without it."""
        model = MockSklearnModel(has_predict_proba=False)
        x = np.random.randn(3, 4)

        with pytest.raises(NotImplementedError, match="does not support predict_proba"):
            strategy.predict_proba(model, x=x)

    def test_predict_proba_with_dataframe(self, strategy):
        """predict_proba should work with DataFrame input."""
        model = MockSklearnModel()
        x = pd.DataFrame(np.random.randn(3, 4), columns=["a", "b", "c", "d"])

        result = strategy.predict_proba(model, x=x)
        assert isinstance(result, np.ndarray)


class TestSklearnPredictionStrategyEnsureDataframe:
    """Tests for the _ensure_dataframe helper method."""

    @pytest.fixture
    def strategy(self):
        """Provide a fresh prediction strategy instance."""
        return SklearnPredictionStrategy()

    def test_dataframe_passed_unchanged(self, strategy):
        """DataFrame input should be returned unchanged."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = strategy._ensure_dataframe(df, {})

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]

    def test_series_converted_to_frame(self, strategy):
        """Series input should be converted to DataFrame."""
        series = pd.Series([1, 2, 3], name="feature")
        result = strategy._ensure_dataframe(series, {})

        assert isinstance(result, (pd.DataFrame, pd.Series))

    def test_numpy_array_with_feature_names(self, strategy):
        """numpy array with feature_names should become DataFrame."""
        arr = np.array([[1, 2], [3, 4]])
        kwargs = {"feature_names": ["col_a", "col_b"]}
        result = strategy._ensure_dataframe(arr, kwargs)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["col_a", "col_b"]

    def test_numpy_array_without_feature_names(self, strategy):
        """numpy array without feature_names should be returned as-is."""
        arr = np.array([[1, 2], [3, 4]])
        result = strategy._ensure_dataframe(arr, {})

        assert isinstance(result, np.ndarray)

    def test_other_types_passed_through(self, strategy):
        """Other input types should be passed through unchanged."""
        lst = [[1, 2], [3, 4]]
        result = strategy._ensure_dataframe(lst, {})

        assert result == lst
