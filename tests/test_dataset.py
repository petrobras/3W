from pathlib import Path
import pytest

import numpy as np
import pandas as pd

from ThreeWToolkit.dataset import DatasetConfig, ParquetDataset

_NUM_SIGNALS  = 11
_NUM_ROWS     = 100
_SIGNAL_NAMES = [f"signal-{i}" for i in range(_NUM_SIGNALS)]
_LABEL_NAME   = "class"
_NUM_CLASSES  = 4

def make_parquet_event(path):
    data  = np.random.randn(_NUM_ROWS, _NUM_SIGNALS).astype(np.float32)
    label = np.random.randint(low=0, high=99, size=_NUM_ROWS)
    df = pd.DataFrame(data, columns=_SIGNAL_NAMES)
    df[_LABEL_NAME] = label
    df.to_parquet(path, engine="pyarrow", compression="brotli")

@pytest.fixture(scope="session")
def parquet_dataset_path(tmp_path_factory):
    base_path = tmp_path_factory.mktemp("data")
    for c in range(_NUM_CLASSES):
        dir_c = base_path / str(c)
        dir_c.mkdir()
        make_parquet_event(dir_c / "WELL_test.parquet")
        make_parquet_event(dir_c / "DRAWN_test.parquet")
        make_parquet_event(dir_c / "SIMULATED_test.parquet")
    return base_path


class TestParquetDataset:
    def test_full_loading(self, parquet_dataset_path):
        """
        Load all files, without target column separation.
        """

        config  = DatasetConfig(path=parquet_dataset_path, file_type="parquet", target_column=None)
        dataset = ParquetDataset(config)

        # check correct number of files were loaded
        assert len(dataset) == _NUM_CLASSES * 3

        for e in dataset:
            assert list(e["signal"].columns) == _SIGNAL_NAMES + [_LABEL_NAME]
            assert e["signal"].shape == (_NUM_ROWS, _NUM_SIGNALS + 1)
            assert "label" not in e

    def test_target_loading(self, parquet_dataset_path):
        """
        Load all files, with target column separation.
        """
        config  = DatasetConfig(path=parquet_dataset_path, file_type="parquet", target_column=_LABEL_NAME)
        dataset = ParquetDataset(config)

        assert len(dataset) == _NUM_CLASSES * 3

        for e in dataset:
            assert list(e["signal"].columns) == _SIGNAL_NAMES
            assert e["signal"].shape == (_NUM_ROWS, _NUM_SIGNALS)
            assert e["label"].shape == (_NUM_ROWS, 1)

    def test_file_listing(self, parquet_dataset_path):
        pass
