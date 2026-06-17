"""Tests for classification metrics using parametrize for cleaner test code."""

import pytest
import numpy as np
import pandas as pd

from ThreeWToolkit.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
)


class TestAccuracyScore:
    @pytest.mark.parametrize(
        "y_true,y_pred,expected",
        [
            ([1, 0, 1, 1], [1, 0, 0, 1], 0.75),  # Basic case: 3/4 correct
            ([1, 0, 1, 1], [1, 0, 1, 1], 1.0),  # Perfect prediction
            ([0, 0, 0, 1], [1, 1, 1, 0], 0.0),  # All wrong
            ([1, 0], [1, 0], 1.0),  # Simple binary
        ],
    )
    def test_accuracy_basic(self, y_true: list, y_pred: list, expected: float):
        """Test accuracy_score with various input combinations."""
        result = accuracy_score(y_true=y_true, y_pred=y_pred)
        assert isinstance(result, float)
        assert 0 <= result <= 1
        assert result == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize(
        "y_true,y_pred,weights",
        [
            ([1, 0, 1, 1], [1, 0, 0, 1], [0.5, 0.2, 0.1, 0.2]),
            ([1, 1, 0], [1, 0, 1], [1.0, 2.0, 3.0]),
        ],
    )
    def test_accuracy_with_sample_weight(
        self, y_true: list, y_pred: list, weights: list
    ):
        """Test accuracy_score with sample weights."""
        result = accuracy_score(y_true=y_true, y_pred=y_pred, sample_weight=weights)
        assert isinstance(result, float)
        assert 0 <= result <= 1

    @pytest.mark.parametrize(
        "y_true,y_pred,error_type",
        [
            (123, [1, 0, 1], (TypeError, Exception)),  # Invalid y_true type
            ([1, 0, 1], "wrong", (TypeError, Exception)),  # Invalid y_pred type
            ([1, 0], [1, 0, 1], ValueError),  # Shape mismatch
            (
                [1, 0, 1],
                [1, 0, 1],
                ValueError,
            ),  # Sample weight shape mismatch (tested separately)
        ],
    )
    def test_accuracy_errors(self, y_true, y_pred, error_type: type | tuple):
        """Test accuracy_score error handling."""
        if (error_type is ValueError) and (len(y_true) == len(y_pred)):
            # This is the sample weight case
            with pytest.raises(ValueError):
                accuracy_score(y_true=y_true, y_pred=y_pred, sample_weight=[0.1, 0.2])
        else:
            with pytest.raises(error_type):
                accuracy_score(y_true=y_true, y_pred=y_pred)


class TestBalancedAccuracyScore:
    @pytest.mark.parametrize(
        "y_true,y_pred,expected",
        [
            ([1, 0, 1, 1], [1, 0, 0, 1], 0.8333333333333333),
            ([0, 1, 1, 0], [0, 0, 1, 1], 0.5),
            ([1, 0, 0, 1], [1, 0, 0, 1], 1.0),
        ],
    )
    def test_balanced_accuracy_basic(self, y_true: list, y_pred: list, expected: float):
        """Test balanced_accuracy_score with binary classification."""
        result = balanced_accuracy_score(y_true, y_pred)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        assert result == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize(
        "adjusted,y_true,y_pred",
        [
            (True, [0, 1, 1, 0], [0, 0, 1, 1]),
            (False, [0, 1, 1, 0], [0, 0, 1, 1]),
        ],
    )
    def test_balanced_accuracy_adjusted(
        self, adjusted: bool, y_true: list, y_pred: list
    ):
        """Test balanced_accuracy_score with adjusted parameter."""
        result = balanced_accuracy_score(y_true, y_pred, adjusted=adjusted)
        assert isinstance(result, float)

    @pytest.mark.parametrize(
        "y_true,y_pred,sample_weight",
        [
            ([1, 0, 1, 0], [1, 1, 1, 0], [1.0, 0.5, 1.5, 2.0]),
            ([0, 1, 1], [0, 1, 0], [1, 1, 1]),
        ],
    )
    def test_balanced_accuracy_with_weights(
        self, y_true: list, y_pred: list, sample_weight: list
    ):
        """Test balanced_accuracy_score with sample weights."""
        score = balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight)
        assert isinstance(score, float)

    @pytest.mark.parametrize(
        "y_true,y_pred,kwargs,error_type",
        [
            ("invalid", [1, 0, 1], {}, (TypeError, Exception)),
            ([1, 0], [0, 1, 0], {}, ValueError),  # Length mismatch
            ([0, 1], [0, 1], {"sample_weight": "invalid"}, (TypeError, Exception)),
            ([0, 1, 1], [0, 1, 1], {"sample_weight": [1.0, 2.0]}, ValueError),
        ],
    )
    def test_balanced_accuracy_errors(
        self, y_true, y_pred, kwargs: dict, error_type: type | tuple
    ):
        """Test balanced_accuracy_score error handling."""
        with pytest.raises(error_type):
            balanced_accuracy_score(y_true, y_pred, **kwargs)


class TestAveragePrecisionScore:
    @pytest.mark.parametrize(
        "y_true,y_pred,average,expected",
        [
            ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], "macro", 0.8333333333333333),
            ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], "micro", 0.8333333333333333),
        ],
    )
    def test_average_precision_basic(
        self, y_true: list, y_pred: list, average: str, expected: float
    ):
        """Test average_precision_score with different averaging methods."""
        result = average_precision_score(y_true=y_true, y_pred=y_pred, average=average)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        assert result == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize(
        "y_true,y_pred,pos_label,expected",
        [
            ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 1, 0.8333333333333333),
            ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 0, 0.5),
        ],
    )
    def test_average_precision_pos_label(
        self, y_true: list, y_pred: list, pos_label: int, expected: float
    ):
        """Test average_precision_score with explicit pos_label."""
        result = average_precision_score(
            y_true=y_true, y_pred=y_pred, pos_label=pos_label
        )
        assert isinstance(result, float)
        assert result == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize(
        "y_true,y_pred,kwargs,error_type",
        [
            ("not_array", [0.8, 0.6, 0.7], {}, (TypeError, Exception)),
            ([1, 0, 1], [0.8, 0.5], {}, ValueError),  # Shape mismatch
            ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], {"average": 123}, ValueError),
            (
                [0, 0, 1, 1],
                [0.1, 0.4, 0.35, 0.8],
                {"sample_weight": [0.5, 0.5]},
                ValueError,
            ),
        ],
    )
    def test_average_precision_errors(
        self, y_true, y_pred, kwargs: dict, error_type: type | tuple
    ):
        """Test average_precision_score error handling."""
        with pytest.raises(error_type):
            average_precision_score(y_true=y_true, y_pred=y_pred, **kwargs)


class TestPrecisionScore:
    @pytest.mark.parametrize(
        "y_true,y_pred,average,expected",
        [
            ([1, 1, 0, 0], [1, 0, 0, 0], "binary", 1.0),
            ([0, 1, 2, 0, 1, 2], [0, 2, 1, 0, 0, 1], "macro", 0.2222222222222222),
            ([0, 1, 2, 0, 1, 2], [0, 2, 1, 0, 0, 1], "micro", 0.3333333333333333),
        ],
    )
    def test_precision_basic(
        self, y_true: list, y_pred: list, average: str, expected: float
    ):
        """Test precision_score with different averaging methods."""
        result = precision_score(y_true=y_true, y_pred=y_pred, average=average)
        assert isinstance(result, (float, np.floating))
        assert result == pytest.approx(expected, abs=1e-6)

    def test_precision_zero_division(self):
        """Test precision_score with zero_division parameter."""
        y_true = [0, 0, 0]
        y_pred = [1, 1, 1]

        result = precision_score(
            y_true=y_true, y_pred=y_pred, average="binary", zero_division=0
        )
        assert isinstance(result, (float, np.floating))


class TestRecallScore:
    @pytest.mark.parametrize(
        "y_true,y_pred,average,expected",
        [
            ([1, 1, 0, 0], [1, 0, 0, 0], "binary", 0.5),
            ([0, 1, 2, 0, 1, 2], [0, 2, 1, 0, 0, 1], "macro", 0.3333333333333333),
            ([0, 1, 2, 0, 1, 2], [0, 2, 1, 0, 0, 1], "micro", 0.3333333333333333),
        ],
    )
    def test_recall_basic(
        self, y_true: list, y_pred: list, average: str, expected: float
    ):
        """Test recall_score with different averaging methods."""
        result = recall_score(y_true=y_true, y_pred=y_pred, average=average)
        assert isinstance(result, (float, np.floating))
        assert result == pytest.approx(expected, abs=1e-6)

    def test_recall_zero_division(self):
        """Test recall_score with zero_division parameter."""
        y_true = [0, 0, 0]
        y_pred = [0, 0, 0]

        result = recall_score(
            y_true=y_true, y_pred=y_pred, average="binary", zero_division=0
        )
        assert isinstance(result, (float, np.floating))


class TestF1Score:
    @pytest.mark.parametrize(
        "y_true,y_pred,average,expected",
        [
            ([1, 1, 0, 0], [1, 0, 0, 0], "binary", 0.6666666666666666),
            ([0, 1, 2, 0, 1, 2], [0, 2, 1, 0, 0, 1], "macro", 0.26666666666666666),
            ([0, 1, 2, 0, 1, 2], [0, 2, 1, 0, 0, 1], "micro", 0.3333333333333333),
        ],
    )
    def test_f1_basic(self, y_true: list, y_pred: list, average: str, expected: float):
        """Test f1_score with different averaging methods."""
        result = f1_score(y_true=y_true, y_pred=y_pred, average=average)
        assert isinstance(result, (float, np.floating))
        assert result == pytest.approx(expected, abs=1e-6)

    def test_f1_zero_division(self):
        """Test f1_score with zero_division parameter."""
        y_true = [0, 0, 0]
        y_pred = [1, 1, 1]

        result = f1_score(
            y_true=y_true, y_pred=y_pred, average="binary", zero_division=0
        )
        assert isinstance(result, (float, np.floating))


class TestROCAUCScore:
    @pytest.mark.parametrize(
        "y_true,y_pred,expected",
        [
            ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 0.75),
            ([0, 1], [0, 1], 1.0),
            ([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], 1.0),
        ],
    )
    def test_roc_auc_basic(self, y_true: list, y_pred: list, expected: float):
        """Test roc_auc_score with binary classification."""
        result = roc_auc_score(y_true=y_true, y_pred=y_pred)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        assert result == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize(
        "y_true,y_pred,average",
        [
            ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], "macro"),
            ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], "micro"),
        ],
    )
    def test_roc_auc_average(self, y_true: list, y_pred: list, average: str):
        """Test roc_auc_score with different averaging methods."""
        result = roc_auc_score(y_true=y_true, y_pred=y_pred, average=average)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize(
        "y_true,y_pred,error_type",
        [
            ("invalid", [0.1, 0.2], (TypeError, Exception)),
            ([0, 1], [0.1, 0.2, 0.3], ValueError),  # Shape mismatch
        ],
    )
    def test_roc_auc_errors(self, y_true: list, y_pred: list, error_type: type | tuple):
        """Test roc_auc_score error handling."""
        with pytest.raises(error_type):
            roc_auc_score(y_true=y_true, y_pred=y_pred)


class TestMatthewsCorrcoef:
    @pytest.fixture(scope="class")
    def sk_mcc(self):
        from sklearn.metrics import matthews_corrcoef

        return matthews_corrcoef

    def test_binary_classification(self, sk_mcc):
        y_true = [1, 1, 0, 0]
        y_pred = [1, 0, 0, 0]

        expected = sk_mcc(y_true, y_pred)
        result = matthews_corrcoef(y_true, y_pred)

        assert result == pytest.approx(expected)

    def test_multiclass_classification(self, sk_mcc):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 1, 0, 2, 2]

        expected = sk_mcc(y_true, y_pred)
        result = matthews_corrcoef(y_true, y_pred)

        assert result == pytest.approx(expected)

    def test_numpy_arrays(self, sk_mcc):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 0])

        expected = sk_mcc(y_true, y_pred)
        result = matthews_corrcoef(y_true, y_pred)

        assert result == pytest.approx(expected)

    def test_pandas_series(self, sk_mcc):
        y_true = pd.Series([1, 1, 0, 0])
        y_pred = pd.Series([1, 0, 0, 0])

        expected = sk_mcc(y_true, y_pred)
        result = matthews_corrcoef(y_true, y_pred)

        assert result == pytest.approx(expected)

    def test_sample_weight(self, sk_mcc):
        y_true = [1, 1, 0, 0]
        y_pred = [1, 0, 0, 0]
        weights = [1.0, 2.0, 1.0, 1.0]

        expected = sk_mcc(
            y_true,
            y_pred,
            sample_weight=weights,
        )

        result = matthews_corrcoef(
            y_true,
            y_pred,
            sample_weight=weights,
        )

        assert result == pytest.approx(expected)

    def test_mismatched_lengths_raises_value_error(self):
        y_true = [1, 0, 1]
        y_pred = [1, 0]

        with pytest.raises(ValueError):
            matthews_corrcoef(y_true, y_pred)

    @pytest.mark.parametrize(
        "y_true,y_pred",
        [
            ("invalid", [1, 0, 1]),
            ([1, 0, 1], "invalid"),
            (123, [1, 0, 1]),
            ([1, 0, 1], 123),
        ],
    )
    def test_invalid_types_raise_error(self, y_true, y_pred):
        with pytest.raises(Exception):
            matthews_corrcoef(y_true, y_pred)


class TestInputTypes:
    """Test that metrics work with different input types (lists, numpy, pandas)."""

    @pytest.mark.parametrize(
        "metric_func,y_true,y_pred",
        [
            (accuracy_score, [1, 0, 1, 1], [1, 0, 0, 1]),
            (balanced_accuracy_score, [1, 0, 1, 1], [1, 0, 0, 1]),
            (precision_score, [1, 1, 0, 0], [1, 0, 0, 0]),
            (recall_score, [1, 1, 0, 0], [1, 0, 0, 0]),
            (f1_score, [1, 1, 0, 0], [1, 0, 0, 0]),
        ],
    )
    def test_metrics_with_numpy_arrays(self, metric_func, y_true: list, y_pred: list):
        """Test metrics with numpy arrays."""
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)

        kwargs = (
            {"average": "binary"}
            if metric_func in [precision_score, recall_score, f1_score]
            else {}
        )
        result = metric_func(y_true=y_true_np, y_pred=y_pred_np, **kwargs)

        assert isinstance(result, (float, np.floating))

    @pytest.mark.parametrize(
        "metric_func,y_true,y_pred",
        [
            (accuracy_score, [1, 0, 1, 1], [1, 0, 0, 1]),
            (balanced_accuracy_score, [1, 0, 1, 1], [1, 0, 0, 1]),
        ],
    )
    def test_metrics_with_pandas_series(self, metric_func, y_true, y_pred):
        """Test metrics with pandas Series."""
        y_true_pd = pd.Series(y_true)
        y_pred_pd = pd.Series(y_pred)

        result = metric_func(y_true=y_true_pd, y_pred=y_pred_pd)
        assert isinstance(result, (float, np.floating))
