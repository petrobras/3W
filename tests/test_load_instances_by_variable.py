import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from ThreeWToolkit.dataset.parquet_dataset import ParquetDataset


class TestLoadInstancesByVariable:
    """Test suite for ParquetDataset.load_instances_by_variable()."""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock ParquetDataset that bypasses file system access."""
        fake_signals = [
            pd.DataFrame(
                {
                    "P-MON-CKP": [1.0, 2.0, 3.0],
                    "T-TPT": [10.0, 20.0, 30.0],
                }
            ),
            pd.DataFrame(
                {
                    "P-MON-CKP": [4.0, 5.0, 6.0],
                    "T-TPT": [40.0, 50.0, 60.0],
                }
            ),
        ]

        with patch.object(ParquetDataset, "__init__", lambda self, *a, **kw: None):
            ds = ParquetDataset.__new__(ParquetDataset)

        ds.config = MagicMock()
        ds.config.columns = ["P-MON-CKP", "T-TPT"]

        ds.files_events = [f"file_{i}" for i in range(len(fake_signals))]
        ds.load_data = MagicMock(side_effect=[{"signal": sig} for sig in fake_signals])

        return ds

    def test_returns_all_configured_variables(self, mock_dataset):
        result = mock_dataset.load_instances_by_variable()

        assert "P-MON-CKP" in result
        assert "T-TPT" in result
        assert len(result) == 2

    def test_returns_correct_number_of_instances(self, mock_dataset):
        result = mock_dataset.load_instances_by_variable()

        assert len(result["P-MON-CKP"]) == 2
        assert len(result["T-TPT"]) == 2

    def test_arrays_contain_correct_values(self, mock_dataset):
        result = mock_dataset.load_instances_by_variable()

        np.testing.assert_array_equal(result["P-MON-CKP"][0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result["T-TPT"][1], [40.0, 50.0, 60.0])

    def test_returns_numpy_arrays(self, mock_dataset):
        result = mock_dataset.load_instances_by_variable()

        for var_name, instances in result.items():
            for arr in instances:
                assert isinstance(arr, np.ndarray)

    def test_filters_to_specific_variables(self, mock_dataset):
        result = mock_dataset.load_instances_by_variable(variables=["T-TPT"])

        assert "T-TPT" in result
        assert "P-MON-CKP" not in result
        assert len(result["T-TPT"]) == 2

    def test_missing_variable_returns_empty_list(self, mock_dataset):
        result = mock_dataset.load_instances_by_variable(variables=["NONEXISTENT"])

        assert "NONEXISTENT" in result
        assert len(result["NONEXISTENT"]) == 0

    def test_calls_load_data_for_each_file(self, mock_dataset):
        mock_dataset.load_instances_by_variable()

        assert mock_dataset.load_data.call_count == 2
