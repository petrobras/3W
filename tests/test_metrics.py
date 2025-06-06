import pytest
import numpy as np

from ThreeWToolkit.metrics import accuracy_score  

def test_accuracy_score_basic():
    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 0, 1]

    result = accuracy_score(y_true = y_true, y_pred = y_pred, sample_weight = None, normalize = True)
    
    assert isinstance(result, float)
    assert 0 <= result <= 1
    assert abs(result - 0.75) < 1e-6  # 3 of 4 

def test_accuracy_score_with_sample_weight():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])
    sample_weight = [0.5, 0.2, 0.1, 0.2]

    result = accuracy_score(y_true = y_true, y_pred = y_pred, sample_weight = sample_weight)
    
    assert isinstance(result, float)

def test_accuracy_score_invalid_types():

    with pytest.raises(TypeError):
        accuracy_score(y_true = 123, y_pred = [1, 0, 1])  # invalid y_true

    with pytest.raises(TypeError):
        accuracy_score(y_true = [1, 0, 1], y_pred = "wrong type")  # invalid y_pred

    with pytest.raises(TypeError):
        accuracy_score(y_true = [1, 0, 1], y_pred = [1, 0, 1], normalize = "true")

def test_accuracy_score_shape_mismatch():
    
    with pytest.raises(ValueError):
        accuracy_score(y_true = [1, 0], y_pred = [1, 0, 1]) 

def test_accuracy_score_sample_weight_shape_mismatch():
    
    with pytest.raises(ValueError):
        accuracy_score(
            y_true = [1, 0, 1],
            y_pred = [1, 0, 1],
            sample_weight = [0.1, 0.2] 
        )