import pytest
import pandas as pd
import numpy as np
import os

@pytest.fixture
def mock_plots_dir(tmp_path, monkeypatch):
    """
    Fixture to create a temporary directory for plots and monkeypatch
    the os functions to use it, preventing plot file creation during tests.
    """
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    
    # Monkeypatch the path creation to point to our temp dir
    monkeypatch.setattr(os.path, 'abspath', lambda x: str(tmp_path))
    # We will also mock plt.savefig directly in tests to avoid actual plotting
    return plots_dir

@pytest.fixture
def mock_time_series_data():
    """
    Fixture to generate consistent time series data for tests.
    """
    time = np.arange(0, 50, 1)
    signal = np.sin(2 * np.pi * time / 10) + np.random.normal(0, 0.1, len(time))
    series = pd.Series(signal, index=time, name="test_signal")
    
    data = {
        "X_train": series.iloc[:40],
        "y_train": series.iloc[:40],
        "X_test": series.iloc[40:],
        "y_test": series.iloc[40:]
    }
    return data

@pytest.fixture
def mock_model():
    """
    Fixture to create a mock model that mimics a scikit-learn model.
    """
    class MockModel:
        def get_params(self):
            return {"param_a": 10, "param_b": "auto", "param_c": [1, 2, 3]}

        def predict(self, X_data):
            # A simple prediction: return the data shifted by one
            return X_data.shift(-1).bfill()

    return MockModel()
