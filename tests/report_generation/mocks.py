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
    monkeypatch.setattr(os.path, "abspath", lambda x: str(tmp_path))
    # We will also mock plt.savefig directly in tests to avoid actual plotting
    return plots_dir


@pytest.fixture
def mock_time_series_data():
    """
    Fixture to generate consistent time series data with a prolonged anomaly
    in the test set.
    """
    # Use a seed for reproducible random noise
    np.random.seed(42)

    time = np.arange(0, 50, 1)
    signal = np.sin(2 * np.pi * time / 10) + np.random.normal(0, 0.1, len(time))
    series = pd.Series(signal, index=time, name="test_signal")

    # --- CHANGES START HERE ---

    # 1. Create the ground truth labels (0 for normal, 1 for anomaly)
    y_labels = pd.Series(0, index=series.index, name="is_anomaly")

    # 2. Inject a prolonged anomaly into the test section of the data
    # This event lasts for 3 time steps (from index 42 to 44)
    series.iloc[42:45] += 3.0
    y_labels.iloc[42:45] = 1  # Mark these indices as anomalies in the labels

    # 3. The returned dictionary now uses the signal for X and labels for Y
    data = {
        "X_train": series.iloc[:40],
        "y_train": y_labels.iloc[:40],  # y_train contains only normal points
        "X_test": series.iloc[40:],
        "y_test": y_labels.iloc[40:],  # y_test contains the anomaly
    }
    # --- CHANGES END HERE ---

    return data


@pytest.fixture
def mock_model():
    """
    Fixture to create a mock model that performs simple anomaly detection.
    """

    class MockModel:
        def get_params(self):
            # Parameters now reflect the anomaly detection method
            return {"method": "rolling_threshold", "window": 3, "threshold": 1.5}

        def predict(self, X_data):
            # A simple anomaly detection logic:
            # 1. Calculate the rolling mean (the "expected" value)
            smoothed_signal = (
                X_data.rolling(window=3, center=True).mean().bfill().ffill()
            )
            # 2. Find the deviation from the mean
            residual = (X_data - smoothed_signal).abs()
            # 3. If deviation is above a threshold, it's an anomaly (1), else normal (0)
            return (residual > 1.5).astype(int)

    return MockModel()
