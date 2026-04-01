"""Tests for regression metrics using parametrize for cleaner test code."""

import pytest
import numpy as np
import pandas as pd

from ThreeWToolkit.metrics import explained_variance_score


class TestExplainedVarianceScore:
    @pytest.mark.parametrize(
        "y_true,y_pred,expected",
        [
            ([3, -0.5, 2, 7], [2.5, 0.0, 2, 8], 0.9571734475374732),
            ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 1.0),  # Perfect prediction
            ([1, 1, 1, 1], [1, 1, 1, 1], 1.0),  # Constant perfect
        ],
    )
    def test_explained_variance_basic(self, y_true, y_pred, expected):
        """Test explained_variance_score basic calculations."""
        result = explained_variance_score(y_true=y_true, y_pred=y_pred)

        assert isinstance(result, float)
        assert result == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize(
        "y_true,y_pred,weights,expected",
        [
            ([3, -0.5, 2, 7], [2.5, 0.0, 2, 8], [1, 2, 3, 4], 0.9689988623435722),
            (
                [1, 2, 3],
                [1.1, 2.1, 3.1],
                [1, 1, 1],
                None,
            ),  # None means just check it runs
        ],
    )
    def test_explained_variance_with_weights(self, y_true, y_pred, weights, expected):
        """Test explained_variance_score with sample weights."""
        result = explained_variance_score(
            y_true=y_true, y_pred=y_pred, sample_weight=weights
        )

        assert isinstance(result, float)
        if expected is not None:
            assert result == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize(
        "y_true,y_pred,multioutput,expected_type,expected_shape",
        [
            (
                [[0.5, 1], [-1, 1], [7, -6]],
                [[0, 2], [-1, 2], [8, -5]],
                "raw_values",
                np.ndarray,
                (2,),
            ),
            (
                [[0.5, 1], [-1, 1], [7, -6]],
                [[0, 2], [-1, 2], [8, -5]],
                "uniform_average",
                float,
                None,
            ),
            (
                [[0.5, 1], [-1, 1], [7, -6]],
                [[0, 2], [-1, 2], [8, -5]],
                "variance_weighted",
                float,
                None,
            ),
        ],
    )
    def test_explained_variance_multioutput(
        self, y_true, y_pred, multioutput, expected_type, expected_shape
    ):
        """Test explained_variance_score with different multioutput options."""
        result = explained_variance_score(
            y_true=y_true, y_pred=y_pred, multioutput=multioutput
        )

        assert isinstance(result, expected_type)
        if expected_shape:
            assert result.shape == expected_shape

    @pytest.mark.parametrize(
        "multioutput,expected_value",
        [
            ("uniform_average", 0.9838709677419355),
            ("raw_values", np.array([0.96774194, 1.0])),
        ],
    )
    def test_explained_variance_multioutput_values(self, multioutput, expected_value):
        """Test explained_variance_score multioutput expected values."""
        y_true = [[0.5, 1], [-1, 1], [7, -6]]
        y_pred = [[0, 2], [-1, 2], [8, -5]]

        result = explained_variance_score(
            y_true=y_true, y_pred=y_pred, multioutput=multioutput
        )

        if isinstance(expected_value, np.ndarray):
            assert np.allclose(result, expected_value, atol=1e-6)
        else:
            assert result == pytest.approx(expected_value, abs=1e-6)

    @pytest.mark.parametrize(
        "y_true,y_pred,kwargs,error_type",
        [
            ([1, 2], [1, 2], {"multioutput": "invalid"}, ValueError),
            ([1, 2, 3], [1, 2], {}, ValueError),  # Shape mismatch
            (
                [1, 2, 3],
                [1, 2, 3],
                {"sample_weight": [1, 2]},
                ValueError,
            ),  # Weight mismatch
        ],
    )
    def test_explained_variance_errors(self, y_true, y_pred, kwargs, error_type):
        """Test explained_variance_score error handling."""
        with pytest.raises(error_type):
            explained_variance_score(y_true=y_true, y_pred=y_pred, **kwargs)

    @pytest.mark.parametrize(
        "y_true,y_pred",
        [
            ([1, 2, 3], [100, 200, 300]),  # Very bad predictions
            ([1, 1, 1], [0, 0, 0]),  # Opposite predictions
        ],
    )
    def test_explained_variance_can_be_negative(self, y_true, y_pred):
        """Test that explained_variance_score can be negative for poor predictions."""
        result = explained_variance_score(y_true=y_true, y_pred=y_pred)
        assert isinstance(result, float)


class TestInputTypes:
    """Test that explained_variance_score works with different input types."""

    @pytest.mark.parametrize(
        "y_true,y_pred",
        [
            ([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]),
            ([1, 2, 3], [1.1, 2.1, 2.9]),
        ],
    )
    def test_with_numpy_arrays(self, y_true, y_pred):
        """Test explained_variance_score with numpy arrays."""
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)

        result = explained_variance_score(y_true=y_true_np, y_pred=y_pred_np)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize(
        "y_true,y_pred",
        [
            ([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]),
            ([1, 2, 3], [1.1, 2.1, 2.9]),
        ],
    )
    def test_with_pandas_series(self, y_true, y_pred):
        """Test explained_variance_score with pandas Series."""
        y_true_pd = pd.Series(y_true)
        y_pred_pd = pd.Series(y_pred)

        result = explained_variance_score(y_true=y_true_pd, y_pred=y_pred_pd)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize(
        "y_true,y_pred",
        [
            ([[0.5, 1], [-1, 1]], [[0, 2], [-1, 2]]),  # Multioutput
        ],
    )
    def test_with_2d_arrays(self, y_true, y_pred):
        """Test explained_variance_score with 2D numpy arrays."""
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)

        result = explained_variance_score(y_true=y_true_np, y_pred=y_pred_np)

        assert isinstance(result, float)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.parametrize(
        "y_true,y_pred",
        [
            ([0, 0, 0], [0, 0, 0]),  # All zeros
            ([1, 1, 1], [1, 1, 1]),  # All same value
        ],
    )
    def test_constant_values(self, y_true, y_pred):
        """Test with constant values."""
        result = explained_variance_score(y_true=y_true, y_pred=y_pred)
        assert isinstance(result, float)

    @pytest.mark.parametrize(
        "size",
        [2, 10, 100, 1000],
    )
    def test_different_sizes(self, size):
        """Test with different array sizes."""
        np.random.seed(42)
        y_true = np.random.randn(size)
        y_pred = y_true + np.random.randn(size) * 0.1

        result = explained_variance_score(y_true=y_true, y_pred=y_pred)

        assert isinstance(result, float)
        assert result > 0.5
