import pytest
import numpy as np

from ThreeWToolkit.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    explained_variance_score,
)


class TestAccuracyScore:
    def test_accuracy_score_basic(self):
        """
        Test basic functionality of the accuracy_score function with correct inputs.
        """
        y_true = [1, 0, 1, 1]
        y_pred = [1, 0, 0, 1]
        expected_result = 0.75

        result = accuracy_score(
            y_true=y_true, y_pred=y_pred, sample_weight=None, normalize=True
        )

        assert isinstance(result, float)
        assert 0 <= result <= 1
        assert np.isclose(result, expected_result, atol=1e-6)  # 3 of 4

    def test_accuracy_score_with_sample_weight(self):
        """
        Test accuracy_score function using sample weights.
        Checks if the function returns a float value when sample weights are applied.
        """
        y_true = np.array([1, 0, 1, 1])
        y_pred = np.array([1, 0, 0, 1])
        sample_weight = [0.5, 0.2, 0.1, 0.2]

        result = accuracy_score(
            y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
        )

        assert isinstance(result, float)

    def test_accuracy_score_invalid_types(self):
        """
        Test accuracy_score function with invalid input types.
        Ensures that the function raises TypeError for invalid y_true, y_pred, or normalize arguments.
        """
        with pytest.raises(TypeError):
            accuracy_score(y_true=123, y_pred=[1, 0, 1])  # invalid y_true

        with pytest.raises(TypeError):
            accuracy_score(y_true=[1, 0, 1], y_pred="wrong type")  # invalid y_pred

        with pytest.raises(TypeError):
            accuracy_score(y_true=[1, 0, 1], y_pred=[1, 0, 1], normalize="true")

    def test_accuracy_score_shape_mismatch(self):
        """
        Test accuracy_score function with mismatched lengths of y_true and y_pred.
        """
        with pytest.raises(ValueError):
            accuracy_score(y_true=[1, 0], y_pred=[1, 0, 1])

    def test_accuracy_score_sample_weight_shape_mismatch(self):
        """
        Test accuracy_score function with mismatched sample_weight length.
        """
        with pytest.raises(ValueError):
            accuracy_score(y_true=[1, 0, 1], y_pred=[1, 0, 1], sample_weight=[0.1, 0.2])


class TestBalancedAccuracyScore:
    def test_balanced_accuracy_score_basic(self):
        """
        Test the balanced_accuracy_score with valid binary classification inputs.
        """
        y_true = [1, 0, 1, 1]
        y_pred = [1, 0, 0, 1]
        expected_result = 0.8333333333333333

        result = balanced_accuracy_score(y_true, y_pred)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        assert np.isclose(result, expected_result, atol=1e-6)

    def test_balanced_accuracy_with_sample_weight(self):
        """
        Test the balanced_accuracy_score with sample weights provided.
        """
        y_true = [1, 0, 1, 0]
        y_pred = [1, 1, 1, 0]
        sample_weight = [1.0, 0.5, 1.5, 2.0]

        score = balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight)
        assert isinstance(score, float)

    def test_balanced_accuracy_adjusted(self):
        """
        Test the balanced_accuracy_score with the `adjusted` parameter set to True.
        """
        y_true = [0, 1, 1, 0]
        y_pred = [0, 0, 1, 1]

        adjusted_score = balanced_accuracy_score(y_true, y_pred, adjusted=True)
        assert isinstance(adjusted_score, float)

    def test_balanced_accuracy_invalid_type_y_true(self):
        """
        Test passing an invalid data type for y_true (string instead of list/array).
        """
        with pytest.raises(TypeError):
            balanced_accuracy_score("invalid", [1, 0, 1])

    def test_balanced_accuracy_mismatched_lengths(self):
        """
        Test the case when y_true and y_pred have mismatched lengths.
        """
        with pytest.raises(ValueError):
            balanced_accuracy_score([1, 0], [0, 1, 0])

    def test_balanced_accuracy_invalid_sample_weight_type(self):
        """
        Test passing an invalid type for sample_weight (string instead of list/array).
        """
        with pytest.raises(TypeError):
            balanced_accuracy_score([0, 1], [0, 1], sample_weight="invalid")

    def test_balanced_accuracy_invalid_sample_weight_shape(self):
        """
        Test sample_weight with a different length than y_true/y_pred.
        """
        with pytest.raises(ValueError):
            balanced_accuracy_score([0, 1, 1], [0, 1, 1], sample_weight=[1.0, 2.0])

    def test_balanced_accuracy_invalid_adjusted_type(self):
        """
        Test passing an invalid type for the `adjusted` parameter (string instead of boolean).
        """
        with pytest.raises(TypeError):
            balanced_accuracy_score([1, 0], [1, 0], adjusted="yes")


class TestAveragePrecisionScore:
    def test_ap_average_score_basic(self):
        """
        Test average_precision_score with binary data using average='macro'.
        """
        y_true = [0, 0, 1, 1]
        y_pred = [0.1, 0.4, 0.35, 0.8]
        expected_result = 0.8333333333333333

        result = average_precision_score(y_true=y_true, y_pred=y_pred, average="macro")

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        assert np.isclose(result, expected_result, atol=1e-6)

    def test_ap_with_sample_weight(self):
        """
        Test average_precision_score with sample weights applied.
        """
        y_true = [1, 0, 1]
        y_pred = [0.9, 0.8, 0.8]
        sample_weight = [1, 2, 1]
        expected_result = 0.75

        result = average_precision_score(
            y_true=y_true, y_pred=y_pred, average="macro", sample_weight=sample_weight
        )

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        assert np.isclose(result, expected_result, atol=1e-6)

    def test_ap_micro_average(self):
        """
        Test average_precision_score with average='micro'.
        """
        y_true = [0, 0, 1, 1]
        y_pred = [0.1, 0.4, 0.35, 0.8]
        expected_result = 0.8333333333333333

        result = average_precision_score(y_true=y_true, y_pred=y_pred, average="micro")

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        assert np.isclose(result, expected_result, atol=1e-6)

    def test_ap_pos_label_argument(self):
        """
        Test average_precision_score with explicitly set pos_label.
        """
        y_true = [0, 0, 1, 1]
        y_pred = [0.1, 0.4, 0.35, 0.8]
        expected_result = 0.8333333333333333

        result = average_precision_score(y_true=y_true, y_pred=y_pred, pos_label=1)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        assert np.isclose(result, expected_result, atol=1e-6)

    def test_ap_invalid_type(self):
        """
        Test average_precision_score with invalid y_true type and invalid pos_label type.
        """
        y_true_invalid = "not_array"
        y_pred = [0.8, 0.6, 0.7]
        pos_label = "1"

        with pytest.raises(TypeError):
            average_precision_score(y_true=y_true_invalid, y_pred=y_pred)

        with pytest.raises(TypeError):
            average_precision_score(
                y_true=[1, 0, 1], y_pred=y_pred, pos_label=pos_label
            )

    def test_ap_shape_mismatch(self):
        """
        Test average_precision_score with mismatched lengths of y_true and y_pred.
        """
        y_true, y_pred = [1, 0, 1], [0.8, 0.5]

        with pytest.raises(ValueError):
            average_precision_score(y_true=y_true, y_pred=y_pred)

    def test_ap_invalid_average_type(self):
        """
        Test average_precision_score with invalid type for average.
        """
        y_true = [0, 0, 1, 1]
        y_pred = [0.1, 0.4, 0.35, 0.8]

        with pytest.raises(ValueError):
            average_precision_score(y_true=y_true, y_pred=y_pred, average=123)

    def test_ap_invalid_sample_weight_shape(self):
        """
        Test average_precision_score with sample_weight of incorrect shape.
        """
        y_true = [0, 0, 1, 1]
        y_pred = [0.1, 0.4, 0.35, 0.8]
        sample_weight = [0.5, 0.5]

        with pytest.raises(ValueError):
            average_precision_score(
                y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
            )


class TestPrecisionScore:
    def test_precision_score_basic(self):
        """
        Test precision_score with binary classification and average='binary'.
        """
        y_true = [0, 1, 0, 1, 0]
        y_pred = [0, 0, 1, 1, 0]
        expected_result = 0.5

        result = precision_score(y_true=y_true, y_pred=y_pred)

        assert isinstance(result, float)
        assert np.isclose(result, expected_result, atol=1e-6)

    def test_precision_macro_average(self):
        """
        Test precision_score with multiclass and average='macro'.
        """
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 2, 1, 0, 0, 1]
        expected_result = 0.2222222222222222

        result = precision_score(y_true=y_true, y_pred=y_pred, average="macro")

        assert isinstance(result, float)
        assert np.isclose(result, expected_result, atol=1e-6)

    def test_precision_with_labels_argument(self):
        """
        Test precision_score using a subset of labels.
        """
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 2, 1, 0, 0, 1]
        labels = [0, 2]
        expected_result = 0.3333333333333333

        result = precision_score(
            y_true=y_true, y_pred=y_pred, labels=labels, average="macro"
        )

        assert isinstance(result, float)
        assert np.isclose(result, expected_result, atol=1e-6)

    def test_precision_with_sample_weight(self):
        """
        Test precision_score with sample weights.
        """
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]
        weights = [1, 1, 5, 1]
        expected_result = 0.5

        result = precision_score(y_true=y_true, y_pred=y_pred, sample_weight=weights)

        assert isinstance(result, float)
        assert np.isclose(result, expected_result, atol=1e-6)

    def test_precision_zero_division(self):
        """
        Test precision_score when zero division occurs.
        """
        y_true = [1, 1, 1, 1]
        y_pred = [0, 0, 0, 0]

        result = precision_score(y_true=y_true, y_pred=y_pred, zero_division=0)

        assert isinstance(result, float)
        assert result == 0.0

        result = precision_score(y_true=y_true, y_pred=y_pred, zero_division=1)

        assert isinstance(result, float)
        assert result == 1.0

    def test_precision_invalid_types(self):
        """
        Test precision_score with invalid input types.
        """
        with pytest.raises(TypeError):
            precision_score(y_true="not_array", y_pred=[0, 1])

        with pytest.raises(TypeError):
            precision_score(y_true=[0, 1], y_pred=[0, 1], pos_label="positive")

        with pytest.raises(TypeError):
            precision_score(y_true=[0, 1], y_pred=[0, 1], labels="not list")

    def test_precision_shape_mismatch(self):
        """
        Test precision_score when y_true and y_pred have different lengths.
        """
        with pytest.raises(ValueError):
            precision_score(y_true=[0, 1, 1], y_pred=[1, 0])

        with pytest.raises(ValueError):
            precision_score(y_true=[1, 0, 1], y_pred=[1, 1, 1], sample_weight=[1, 1])

    def test_precision_invalid_average_type(self):
        """
        Test precision_score with invalid average type.
        """
        with pytest.raises(ValueError):
            precision_score(y_true=[0, 1], y_pred=[0, 1], average="invalid")

    def test_precision_invalid_zero_division(self):
        """
        Test precision_score with invalid zero_division value.
        """
        with pytest.raises(ValueError):
            precision_score(y_true=[1, 0], y_pred=[0, 1], zero_division="bad_value")


class TestRecallScore:
    def test_recall_score_basic(self):
        """
        Test recall_score with binary classification and average='binary'.
        """
        y_true = [0, 1, 1, 1]
        y_pred = [0, 1, 0, 1]
        expected_result = 2 / 3

        result = recall_score(y_true=y_true, y_pred=y_pred, average="binary")

        assert isinstance(result, float)
        assert np.isclose(result, expected_result, atol=1e-6)

    def test_recall_macro_average(self):
        """
        Test recall_score with multiclass and average='macro'.
        """
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 2, 1, 0, 0, 1]
        expected_result = 0.3333333333333333

        result = recall_score(y_true=y_true, y_pred=y_pred, average="macro")

        assert isinstance(result, float)
        assert np.isclose(result, expected_result, atol=1e-6)

    def test_recall_with_labels_argument(self):
        """
        Test recall_score using a subset of labels.
        """
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 2, 1, 0, 0, 1]
        labels = [0, 2]
        expected_result = 0.5

        result = recall_score(
            y_true=y_true, y_pred=y_pred, labels=labels, average="macro"
        )

        assert isinstance(result, float)
        assert np.isclose(result, expected_result, atol=1e-6)

    def test_recall_with_sample_weight(self):
        """
        Test recall_score with sample weights.
        """
        y_true = [0, 1, 1, 1]
        y_pred = [0, 1, 0, 1]
        weights = [1, 1, 4, 1]
        expected_result = 0.3333333333333333

        result = recall_score(y_true=y_true, y_pred=y_pred, sample_weight=weights)

        assert isinstance(result, float)
        assert np.isclose(result, expected_result, atol=1e-6)

    def test_recall_zero_division(self):
        """
        Test recall_score when zero division occurs.
        """
        y_true = [0, 0, 0, 0]
        y_pred = [1, 1, 1, 1]

        result = recall_score(y_true=y_true, y_pred=y_pred, zero_division=0)
        assert result == 0.0

        result = recall_score(y_true=y_true, y_pred=y_pred, zero_division=1)
        assert result == 1.0

    def test_recall_invalid_types(self):
        """
        Test recall_score with invalid input types.
        """
        with pytest.raises(TypeError):
            recall_score(y_true="invalid", y_pred=[0, 1])

        with pytest.raises(TypeError):
            recall_score(y_true=[1, 0], y_pred=[1, 0], pos_label="wrong")

        with pytest.raises(TypeError):
            recall_score(y_true=[0, 1], y_pred=[0, 1], labels="not list")

    def test_recall_shape_mismatch(self):
        """
        Test recall_score when y_true and y_pred have different lengths.
        """
        with pytest.raises(ValueError):
            recall_score(y_true=[1, 0], y_pred=[1])

        with pytest.raises(ValueError):
            recall_score(y_true=[1, 0, 1], y_pred=[1, 0, 1], sample_weight=[1, 1])

    def test_recall_invalid_average_type(self):
        """
        Test recall_score with invalid average type.
        """
        with pytest.raises(ValueError):
            recall_score(y_true=[0, 1], y_pred=[0, 1], average="invalid")

    def test_recall_invalid_zero_division(self):
        """
        Test recall_score with invalid zero_division value.
        """
        with pytest.raises(ValueError):
            recall_score(y_true=[1, 0], y_pred=[0, 1], zero_division="bad_value")


class TestF1Score:
    def test_f1_score_basic(self):
        """
        Test f1_score with binary classification and average='binary'.
        """
        y_true = [0, 1, 1, 1]
        y_pred = [0, 1, 0, 1]
        expected_result = 0.8

        result = f1_score(y_true=y_true, y_pred=y_pred, average="binary")

        assert isinstance(result, float)
        assert np.isclose(result, expected_result, atol=1e-6)

    def test_f1_macro_average(self):
        """
        Test f1_score with multiclass and average='macro'.
        """
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 2, 1, 0, 0, 1]
        expected_result = 0.26666666666666666

        result = f1_score(y_true=y_true, y_pred=y_pred, average="macro")

        assert isinstance(result, float)
        assert np.isclose(result, expected_result, atol=1e-6)

    def test_f1_with_labels_argument(self):
        """
        Test f1_score using a subset of labels.
        """
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 2, 1, 0, 0, 1]
        labels = [0, 2]
        expected_result = 0.4

        result = f1_score(y_true=y_true, y_pred=y_pred, labels=labels, average="macro")

        assert isinstance(result, float)
        assert np.isclose(result, expected_result, atol=1e-6)

    def test_f1_with_sample_weight(self):
        """
        Test f1_score with sample weights.
        """
        y_true = [0, 1, 1, 1]
        y_pred = [0, 1, 0, 1]
        weights = [1, 1, 5, 1]
        expected_result = 0.4444444444444444

        result = f1_score(y_true=y_true, y_pred=y_pred, sample_weight=weights)

        assert isinstance(result, float)
        assert np.isclose(result, expected_result, atol=1e-6)

    def test_f1_zero_division(self):
        """
        Test f1_score when zero division occurs.
        """
        y_true = [0, 0, 0, 0]
        y_pred = [0, 0, 0, 0]

        result = f1_score(y_true=y_true, y_pred=y_pred, zero_division=0)
        assert result == 0.0

        result = f1_score(y_true=y_true, y_pred=y_pred, zero_division=1)
        assert result == 1.0

    def test_f1_invalid_types(self):
        """
        Test f1_score with invalid input types.
        """
        with pytest.raises(TypeError):
            f1_score(y_true="invalid", y_pred=[0, 1])

        with pytest.raises(TypeError):
            f1_score(y_true=[1, 0], y_pred=[1, 0], pos_label="wrong")

        with pytest.raises(TypeError):
            f1_score(y_true=[0, 1], y_pred=[0, 1], labels="not list")

    def test_f1_shape_mismatch(self):
        """
        Test f1_score when y_true and y_pred have different lengths.
        """
        with pytest.raises(ValueError):
            f1_score(y_true=[1, 0], y_pred=[1])

        with pytest.raises(ValueError):
            f1_score(y_true=[1, 0, 1], y_pred=[1, 0, 1], sample_weight=[1, 1])

    def test_f1_invalid_average_type(self):
        """
        Test f1_score with invalid average type.
        """
        with pytest.raises(ValueError):
            f1_score(y_true=[0, 1], y_pred=[0, 1], average="invalid")

    def test_f1_invalid_zero_division(self):
        """
        Test f1_score with invalid zero_division value.
        """
        with pytest.raises(ValueError):
            f1_score(y_true=[1, 0], y_pred=[0, 1], zero_division="bad_value")


class TestRocAucScore:
    def test_roc_auc_score_basic(self):
        """
        Test roc_auc_score for binary classification.
        """
        y_true = [0, 0, 1, 1]
        y_pred = [0.1, 0.4, 0.35, 0.8]
        expected = 0.75

        result = roc_auc_score(y_true=y_true, y_pred=y_pred)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        assert np.isclose(result, expected, atol=1e-6)

    def test_weighted_roc_auc(self):
        """
        Test roc_auc_score with sample weights.
        """
        y_true = [0, 0, 1, 1]
        y_pred = [0.1, 0.4, 0.35, 0.8]
        weights = [0.5, 0.5, 1, 1]
        expected_result = 0.75

        result = roc_auc_score(y_true=y_true, y_pred=y_pred, sample_weight=weights)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        assert np.isclose(result, expected_result, atol=1e-6)

    def test_multiclass_roc_auc_ovr(self):
        """
        Test roc_auc_score for multiclass with multi_class='ovr'.
        """
        y_true = [0, 1, 2, 2]
        y_pred = [[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.2, 0.6], [0.2, 0.6, 0.2]]
        expected_result = 0.9583333333333334

        result = roc_auc_score(y_true=y_true, y_pred=y_pred, multi_class="ovr")

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        assert np.isclose(result, expected_result, atol=1e-6)

    def test_invalid_y_pred_type(self):
        """
        Test roc_auc_score with invalid type for y_pred.
        """
        with pytest.raises(TypeError):
            roc_auc_score(y_true=[0, 1], y_pred="invalid")

    def test_invalid_average_type(self):
        """
        Test roc_auc_score with invalid type for average.
        """
        y_true = [0, 0, 1, 1]
        y_pred = [0.1, 0.4, 0.35, 0.8]

        with pytest.raises(ValueError):
            roc_auc_score(y_true=y_true, y_pred=y_pred, average=123)

    def test_mismatched_lengths(self):
        """
        Test roc_auc_score with different lengths for y_true and y_pred.
        """
        with pytest.raises(ValueError):
            roc_auc_score(y_true=[0, 1], y_pred=[0.9])

    def test_invalid_max_fpr(self):
        """
        Test roc_auc_score with invalid max_fpr value.
        """
        with pytest.raises(ValueError):
            roc_auc_score(y_true=[0, 1], y_pred=[0.8, 0.9], max_fpr=1.5)

    def test_invalid_multi_class(self):
        """
        Test roc_auc_score with invalid multi_class value.
        """
        with pytest.raises(ValueError):
            roc_auc_score(y_true=[0, 1], y_pred=[0.8, 0.9], multi_class="invalid")


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
        with pytest.raises(TypeError):
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
