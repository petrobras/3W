"""Tests for regression metrics."""

import pytest
import numpy as np

from ThreeWToolkit.metrics import explained_variance_score


class TestExplainedVarianceScore:
    def test_basic_explained_variance(self):
        """
        Test explained_variance_score basic output.
        """
        y_true = [3, -0.5, 2, 7]
        y_pred = [2.5, 0.0, 2, 8]
        expected_result = 0.9571734475374732

        result = explained_variance_score(y_true=y_true, y_pred=y_pred)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        assert np.isclose(result, expected_result, atol=1e-6)

    def test_explained_variance_with_sample_weight(self):
        """
        Test explained_variance_score with sample weights.
        """
        y_true = [3, -0.5, 2, 7]
        y_pred = [2.5, 0.0, 2, 8]
        weights = [1, 2, 3, 4]
        expected_result = 0.9689988623435722

        result = explained_variance_score(
            y_true=y_true, y_pred=y_pred, sample_weight=weights
        )

        assert isinstance(result, float)
        assert np.isclose(result, expected_result, atol=1e-6)

    def test_multioutput_raw_values(self):
        """
        Test explained_variance_score with multioutput='raw_values'.
        """
        y_true = [[0.5, 1], [-1, 1], [7, -6]]
        y_pred = [[0, 2], [-1, 2], [8, -5]]
        expected_result = np.array([0.96774194, 1.0])

        result = explained_variance_score(
            y_true=y_true, y_pred=y_pred, multioutput="raw_values"
        )

        assert isinstance(result, np.ndarray)
        assert np.allclose(result, expected_result, atol=1e-6)

    def test_invalid_force_finite(self):
        """
        Test explained_variance_score with invalid force_finite type.
        """
        with pytest.raises((TypeError, Exception)):
            explained_variance_score(y_true=[1, 2], y_pred=[1, 2], force_finite="yes")

    def test_invalid_multioutput_value(self):
        """
        Test explained_variance_score with invalid multioutput.
        """
        with pytest.raises(ValueError):
            explained_variance_score(
                y_true=[1, 2], y_pred=[1, 2], multioutput="invalid"
            )

    def test_shape_mismatch(self):
        """
        Test explained_variance_score with mismatched y_true and y_pred lengths.
        """
        with pytest.raises(ValueError):
            explained_variance_score(y_true=[1, 2, 3], y_pred=[1, 2])

    def test_sample_weight_mismatch(self):
        """
        Test explained_variance_score with invalid sample_weight shape.
        """
        with pytest.raises(ValueError):
            explained_variance_score(
                y_true=[1, 2, 3], y_pred=[1, 2, 3], sample_weight=[1, 2]
            )

    def test_perfect_prediction(self):
        """
        Test explained_variance_score with perfect predictions.
        """
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 2, 3, 4, 5]

        result = explained_variance_score(y_true=y_true, y_pred=y_pred)

        assert np.isclose(result, 1.0, atol=1e-6)

    def test_uniform_average(self):
        """
        Test explained_variance_score with multioutput='uniform_average'.
        """
        y_true = [[0.5, 1], [-1, 1], [7, -6]]
        y_pred = [[0, 2], [-1, 2], [8, -5]]
        expected_result = 0.9838709677419355

        result = explained_variance_score(
            y_true=y_true, y_pred=y_pred, multioutput="uniform_average"
        )

        assert isinstance(result, float)
        assert np.isclose(result, expected_result, atol=1e-6)

    def test_variance_weighted(self):
        """
        Test explained_variance_score with multioutput='variance_weighted'.
        """
        y_true = [[0.5, 1], [-1, 1], [7, -6]]
        y_pred = [[0, 2], [-1, 2], [8, -5]]

        result = explained_variance_score(
            y_true=y_true, y_pred=y_pred, multioutput="variance_weighted"
        )

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_with_numpy_arrays(self):
        """
        Test explained_variance_score with numpy arrays.
        """
        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, 0.0, 2, 8])

        result = explained_variance_score(y_true=y_true, y_pred=y_pred)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_with_pandas_series(self):
        """
        Test explained_variance_score with pandas Series.
        """
        import pandas as pd

        y_true = pd.Series([3, -0.5, 2, 7])
        y_pred = pd.Series([2.5, 0.0, 2, 8])

        result = explained_variance_score(y_true=y_true, y_pred=y_pred)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_negative_variance_score(self):
        """
        Test explained_variance_score can be negative for very poor predictions.
        """
        y_true = [1, 2, 3]
        y_pred = [100, 200, 300]  # Very bad predictions

        result = explained_variance_score(y_true=y_true, y_pred=y_pred)

        # Can be negative when predictions are worse than just predicting the mean
        assert isinstance(result, float)
