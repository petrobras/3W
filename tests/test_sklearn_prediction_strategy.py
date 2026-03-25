import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from ThreeWToolkit.assessment.strategies.sklearn_prediction_strategy import (
    SklearnPredictionStrategy,
)


def mock_model(
    predictions: np.ndarray | None = None,
    probas: np.ndarray | None = None,
    has_predict: bool = True,
    has_predict_proba: bool = True,
):
    """Creates a mocked sklearn-like model.

    The returned object mimics a wrapper model containing a ``model_class``
    attribute with optional ``predict`` and ``predict_proba`` methods.

    Args:
        predictions (np.ndarray | None): Predictions returned by ``predict``.
        probas (np.ndarray | None): Probabilities returned by ``predict_proba``.
        has_predict (bool): Whether the model exposes a ``predict`` method.
        has_predict_proba (bool): Whether the model exposes a ``predict_proba`` method.

    Returns:
        MagicMock: Mocked model object.
    """
    model = MagicMock()
    if has_predict:
        model.model_class.predict.return_value = predictions or np.array([0, 1, 0])
    else:
        del model.model_class.predict
    if has_predict_proba:
        model.model_class.predict_proba.return_value = probas or np.array(
            [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]]
        )
    else:
        del model.model_class.predict_proba
    return model


def make_X(as_df: bool = False, n_samples: int = 3, cols: int = 2):
    """Generates random feature data for prediction tests.

    Args:
        as_df (bool): If True, returns a pandas DataFrame instead of a NumPy array.
        n_samples (int): Number of samples.
        cols (int): Number of features.

    Returns:
        np.ndarray | pandas.DataFrame: Generated feature matrix.
    """
    data = np.random.rand(n_samples, cols).astype(np.float32)
    return pd.DataFrame(data, columns=[f"f{i}" for i in range(cols)]) if as_df else data


class TestSklearnPredictionStrategy:
    """Unit tests for the SklearnPredictionStrategy class.

    This test suite verifies prediction behavior, probability predictions,
    input validation, and internal data conversion utilities.
    """

    @pytest.fixture
    def strategy(self):
        """Creates a SklearnPredictionStrategy instance for testing.

        Returns:
            SklearnPredictionStrategy: Strategy instance used in tests.
        """
        return SklearnPredictionStrategy()

    def test_predict_raises_if_no_X(self, strategy: SklearnPredictionStrategy):
        """Ensures predict raises an error when no input data is provided."""
        with pytest.raises(ValueError, match="Input data must be provided"):
            strategy.predict(mock_model())

    def test_predict_raises_if_no_predict_method(
        self, strategy: SklearnPredictionStrategy
    ):
        """Ensures predict raises an error if the model lacks a predict method."""
        model = MagicMock(spec=[])
        model.model_class = MagicMock(spec=[])
        with pytest.raises(ValueError, match="predict"):
            strategy.predict(model, x=make_X())

    def test_predict_returns_numpy_array(self, strategy: SklearnPredictionStrategy):
        """Ensures predict returns predictions as a NumPy array."""
        preds = strategy.predict(mock_model(), x=make_X())
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (3,)

    def test_predict_with_dataframe_input(self, strategy: SklearnPredictionStrategy):
        """Ensures predict works correctly when input is a pandas DataFrame."""
        preds = strategy.predict(mock_model(), x=make_X(as_df=True))
        assert isinstance(preds, np.ndarray)

    def test_predict_proba_raises_if_no_X(self, strategy: SklearnPredictionStrategy):
        """Ensures predict_proba raises an error when no input data is provided."""
        with pytest.raises(ValueError, match="Input data must be provided"):
            strategy.predict_proba(mock_model())

    def test_predict_proba_raises_if_not_implemented(
        self, strategy: SklearnPredictionStrategy
    ):
        """Ensures predict_proba raises an error if the model does not implement it."""
        model = MagicMock()
        model.model_class = MagicMock(spec=[])
        with pytest.raises(NotImplementedError, match="predict_proba"):
            strategy.predict_proba(model, x=make_X())

    def test_predict_proba_returns_numpy_array(
        self, strategy: SklearnPredictionStrategy
    ):
        """Ensures predict_proba returns probability estimates as a NumPy array."""
        probas = strategy.predict_proba(mock_model(), x=make_X())
        assert isinstance(probas, np.ndarray)
        assert probas.shape == (3, 2)

    def test_ensure_dataframe_preserves_dataframe(
        self, strategy: SklearnPredictionStrategy
    ):
        """Ensures DataFrame inputs remain unchanged."""
        df = make_X(as_df=True)
        result = strategy._ensure_dataframe(df, {})
        assert isinstance(result, pd.DataFrame)

    def test_ensure_dataframe_converts_series(
        self, strategy: SklearnPredictionStrategy
    ):
        """Ensures pandas Series inputs are converted to DataFrame."""
        s = pd.Series([1.0, 2.0, 3.0])
        result = strategy._ensure_dataframe(s, {})
        assert isinstance(result, pd.DataFrame)

    def test_ensure_dataframe_returns_array_unchanged(
        self, strategy: SklearnPredictionStrategy
    ):
        """Ensures NumPy array inputs remain unchanged."""
        arr = make_X()
        result = strategy._ensure_dataframe(arr, {})
        assert isinstance(result, np.ndarray)

    def test_ensure_dataframe_uses_feature_names_if_available(
        self, strategy: SklearnPredictionStrategy
    ):
        """Ensures feature names are applied when converting arrays to DataFrame."""
        arr = make_X()
        result = strategy._ensure_dataframe(arr, {"feature_names": ["f0", "f1"]})
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["f0", "f1"]
