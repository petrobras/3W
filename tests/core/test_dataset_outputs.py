"""Tests for DatasetOutputs class."""

import pytest
import pandas as pd
import numpy as np

from ThreeWToolkit.core import DatasetOutputs


class TestDatasetOutputsCreation:
    """Test DatasetOutputs instantiation."""

    def test_create_with_signal_and_label(self):
        """Test creating DatasetOutputs with signal and label."""
        signal = pd.DataFrame({"sensor_0": [1.0, 2.0], "sensor_1": [3.0, 4.0]})
        label = pd.Series([0, 1], name="label")

        output = DatasetOutputs(signal=signal, label=label)

        assert isinstance(output.signal, pd.DataFrame)
        assert isinstance(output.label, pd.Series)
        assert output.metadata == {}
        assert len(output.signal) == 2

    def test_create_with_none_label(self):
        """Test creating DatasetOutputs with None label (unsupervised case)."""
        signal = pd.DataFrame({"sensor_0": [1.0, 2.0]})

        output = DatasetOutputs(signal=signal, label=None)

        assert output.label is None
        assert isinstance(output.signal, pd.DataFrame)

    def test_create_with_metadata(self):
        """Test creating DatasetOutputs with metadata."""
        signal = pd.DataFrame({"sensor_0": [1.0]})
        label = pd.Series([0])
        metadata = {"file_name": "test.parquet", "event_id": 42}

        output = DatasetOutputs(signal=signal, label=label, metadata=metadata)

        assert output.metadata["file_name"] == "test.parquet"
        assert output.metadata["event_id"] == 42

    def test_create_empty_metadata_default(self):
        """Test that metadata defaults to empty dict."""
        signal = pd.DataFrame({"sensor_0": [1.0]})
        label = pd.Series([0])

        output = DatasetOutputs(signal=signal, label=label)

        assert output.metadata == {}
        assert isinstance(output.metadata, dict)


class TestDatasetOutputsValidation:
    """Test DatasetOutputs validation behavior."""

    def test_signal_required(self):
        """Test that signal is a required field."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            DatasetOutputs(label=pd.Series([0, 1]))

    def test_accepts_numpy_like_data(self):
        """Test that signal can be created from numpy-like data."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        signal = pd.DataFrame(data, columns=["a", "b"])
        label = pd.Series([0, 1])

        output = DatasetOutputs(signal=signal, label=label)

        assert output.signal.shape == (2, 2)


class TestDatasetOutputsUsage:
    """Test DatasetOutputs in practical usage scenarios."""

    def test_access_signal_columns(self):
        """Test accessing signal columns."""
        signal = pd.DataFrame({
            "P-PDG": [100.0, 101.0],
            "T-TPT": [50.0, 51.0],
            "P-MON-CKP": [200.0, 201.0],
        })
        label = pd.Series([0, 0])

        output = DatasetOutputs(signal=signal, label=label)

        assert list(output.signal.columns) == ["P-PDG", "T-TPT", "P-MON-CKP"]
        assert output.signal["P-PDG"].iloc[0] == 100.0

    def test_access_label_values(self):
        """Test accessing label values."""
        signal = pd.DataFrame({"sensor_0": [1.0, 2.0, 3.0]})
        label = pd.Series([0, 1, 1])

        output = DatasetOutputs(signal=signal, label=label)

        assert output.label.sum() == 2
        assert list(output.label.values) == [0, 1, 1]

    def test_signal_with_nan_values(self):
        """Test that DatasetOutputs accepts NaN values in signal."""
        signal = pd.DataFrame({
            "sensor_0": [1.0, np.nan, 3.0],
            "sensor_1": [np.nan, np.nan, np.nan],
        })
        label = pd.Series([0, 0, 1])

        output = DatasetOutputs(signal=signal, label=label)

        assert output.signal.isna().sum().sum() == 4

    def test_metadata_modification(self):
        """Test modifying metadata after creation."""
        signal = pd.DataFrame({"sensor_0": [1.0]})
        label = pd.Series([0])

        output = DatasetOutputs(signal=signal, label=label, metadata={"key": "value"})
        output.metadata["new_key"] = "new_value"

        assert output.metadata["new_key"] == "new_value"
        assert output.metadata["key"] == "value"
