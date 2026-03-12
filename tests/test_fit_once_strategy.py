import pytest
import numpy as np

from unittest.mock import MagicMock
from ThreeWToolkit.trainer.strategies.fit_once_strategy import FitOnceStrategy


@pytest.fixture
def strategy():
    """Creates a FitOnceStrategy instance for use in tests.

    Returns:
        FitOnceStrategy: A new training strategy instance.
    """
    return FitOnceStrategy()


@pytest.fixture
def mock_model():
    """Creates a mock model with a ``model_class`` attribute.

    The ``model_class`` attribute mimics an underlying estimator that
    exposes a ``fit`` method, allowing tests to verify that the strategy
    correctly delegates the training call.

    Returns:
        MagicMock: A mocked model object containing a mocked
            ``model_class`` attribute.
    """
    model = MagicMock()
    model.model_class = MagicMock()
    return model


class TestFitOnceStrategy:
    """Unit tests for the FitOnceStrategy training behavior.

    This test suite verifies that the strategy correctly delegates the
    training process to the underlying model's ``fit`` method and
    returns the expected training history.
    """

    def test_train_calls_fit_with_correct_args(self, strategy, mock_model):
        """Ensures that the strategy calls the model ``fit`` method with the correct arguments."""
        x = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        strategy.train(mock_model, x, y)

        mock_model.model_class.fit.assert_called_once_with(x, y)

    def test_train_returns_model_in_history(self, strategy, mock_model):
        """Verifies that the returned history dictionary contains the trained model."""
        x = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        history = strategy.train(mock_model, x, y)

        assert history["model"] is mock_model

    def test_train_ignores_validation_data(self, strategy, mock_model):
        """Ensures that validation data is ignored by the strategy.

        The FitOnceStrategy performs a single ``fit`` call and therefore
        should not pass validation data to the underlying model.
        """
        x_train = np.array([[1, 2]])
        y_train = np.array([0])
        x_val = np.array([[3, 4]])
        y_val = np.array([1])

        strategy.train(mock_model, x_train, y_train, x_val, y_val)

        mock_model.model_class.fit.assert_called_once_with(x_train, y_train)

    def test_train_raises_if_model_has_no_model_class(self, strategy):
        """Ensures that an error is raised when the model lacks ``model_class``."""
        bad_model = MagicMock(spec=[])
        with pytest.raises(AttributeError):
            strategy.train(bad_model, np.array([[1]]), np.array([0]))
