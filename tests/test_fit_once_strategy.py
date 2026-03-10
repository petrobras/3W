import pytest
import numpy as np

from unittest.mock import MagicMock
from ThreeWToolkit.trainer.strategies.fit_once_strategy import FitOnceStrategy


@pytest.fixture
def strategy():
    return FitOnceStrategy()


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.model_class = MagicMock()
    return model


class TestFitOnceStrategy:

    def test_train_calls_fit_with_correct_args(self, strategy, mock_model):
        x = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        strategy.train(mock_model, x, y)

        mock_model.model_class.fit.assert_called_once_with(x, y)

    def test_train_returns_model_in_history(self, strategy, mock_model):
        x = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        history = strategy.train(mock_model, x, y)

        assert history["model"] is mock_model

    def test_train_ignores_validation_data(self, strategy, mock_model):
        x_train = np.array([[1, 2]])
        y_train = np.array([0])
        x_val = np.array([[3, 4]])
        y_val = np.array([1])

        strategy.train(mock_model, x_train, y_train, x_val, y_val)

        mock_model.model_class.fit.assert_called_once_with(x_train, y_train)

    def test_train_raises_if_model_has_no_model_class(self, strategy):
        bad_model = MagicMock(spec=[])  # sem atributo model_class
        with pytest.raises(AttributeError):
            strategy.train(bad_model, np.array([[1]]), np.array([0]))
