import pytest
import numpy as np

from ThreeWToolkit.metrics import (
    accuracy_score,
    balanced_accuracy_score
)

class TestAccuracyScore:
    def test_accuracy_score_basic(self):
        """
        Test basic functionality of the accuracy_score function with correct inputs.
        """
        y_true = [1, 0, 1, 1]
        y_pred = [1, 0, 0, 1]
        expected_result = 0.75

        result = accuracy_score(y_true = y_true, y_pred = y_pred, sample_weight = None, normalize = True)
        
        assert isinstance(result, float)
        assert 0 <= result <= 1
        assert np.isclose(result, expected_result, atol = 1e-6)  # 3 of 4 

    def test_accuracy_score_with_sample_weight(self):
        """
        Test accuracy_score function using sample weights.
        Checks if the function returns a float value when sample weights are applied.
        """
        y_true = np.array([1, 0, 1, 1])
        y_pred = np.array([1, 0, 0, 1])
        sample_weight = [0.5, 0.2, 0.1, 0.2]

        result = accuracy_score(y_true = y_true, y_pred = y_pred, sample_weight = sample_weight)
        
        assert isinstance(result, float)
    
    def test_accuracy_score_invalid_types(self):
        """
        Test accuracy_score function with invalid input types.
        Ensures that the function raises TypeError for invalid y_true, y_pred, or normalize arguments.
        """
        with pytest.raises(TypeError):
            accuracy_score(y_true = 123, y_pred = [1, 0, 1])  # invalid y_true

        with pytest.raises(TypeError):
            accuracy_score(y_true = [1, 0, 1], y_pred = "wrong type")  # invalid y_pred

        with pytest.raises(TypeError):
            accuracy_score(y_true = [1, 0, 1], y_pred = [1, 0, 1], normalize = "true")

    def test_accuracy_score_shape_mismatch(self):
        """
        Test accuracy_score function with mismatched lengths of y_true and y_pred.
        """
        with pytest.raises(ValueError):
            accuracy_score(y_true = [1, 0], y_pred = [1, 0, 1]) 

    def test_accuracy_score_sample_weight_shape_mismatch(self):
        """
        Test accuracy_score function with mismatched sample_weight length.
        """
        with pytest.raises(ValueError):
            accuracy_score(
                y_true = [1, 0, 1],
                y_pred = [1, 0, 1],
                sample_weight = [0.1, 0.2] 
            )

class TestBalancedAccuracyScore:
    def test_balanced_accuracy_score_basic(self):
        """
        Test the balanced_accuracy_score with valid binary classification inputs.
        """
        y_true = [1, 0, 1, 1]
        y_pred = [1, 0, 0, 1]
        except_result = 0.8333333333333333

        result = balanced_accuracy_score(y_true, y_pred)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        assert np.isclose(result, except_result, atol = 1e-6)

    def test_balanced_accuracy_with_sample_weight(self):
        """
        Test the balanced_accuracy_score with sample weights provided.
        """
        y_true = [1, 0, 1, 0]
        y_pred = [1, 1, 1, 0]
        sample_weight = [1.0, 0.5, 1.5, 2.0]
        
        score = balanced_accuracy_score(y_true, y_pred, sample_weight = sample_weight)
        assert isinstance(score, float)

    def test_balanced_accuracy_adjusted(self):
        """
        Test the balanced_accuracy_score with the `adjusted` parameter set to True.
        """
        y_true = [0, 1, 1, 0]
        y_pred = [0, 0, 1, 1]
        
        adjusted_score = balanced_accuracy_score(y_true, y_pred, adjusted = True)
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
            balanced_accuracy_score([0, 1], [0, 1], sample_weight = "invalid")

    def test_balanced_accuracy_invalid_sample_weight_shape(self):
        """
        Test sample_weight with a different length than y_true/y_pred.
        """
        with pytest.raises(ValueError):
            balanced_accuracy_score([0, 1, 1], [0, 1, 1], sample_weight = [1.0, 2.0])

    def test_balanced_accuracy_invalid_adjusted_type(self):
        """
        Test passing an invalid type for the `adjusted` parameter (string instead of boolean).
        """
        with pytest.raises(TypeError):
            balanced_accuracy_score([1, 0], [1, 0], adjusted = "yes")
