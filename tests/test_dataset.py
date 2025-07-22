import pytest

import numpy as np
import pandas as pd

from ThreeWToolkit.dataset import DatasetConfig, ParquetDataset
from ThreeWToolkit.core.base_dataset import EventPrefixEnum


_NUM_SIGNALS = 11
_NUM_ROWS = 100
_SIGNAL_NAMES = [f"signal-{i}" for i in range(_NUM_SIGNALS)]
_LABEL_NAME = "class"
_NUM_CLASSES = 6


def make_parquet_event(path):
    data = np.random.randn(_NUM_ROWS, _NUM_SIGNALS).astype(np.float32)
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
        for prefix in EventPrefixEnum:
            make_parquet_event(dir_c / f"{prefix.value}_test.parquet")
    return base_path


class TestParquetDataset:
    def test_wrong_file_type(self, parquet_dataset_path):
        """
        Reports wrong file_type.
        """
        config = DatasetConfig(path=parquet_dataset_path, file_type="csv")
        with pytest.raises(ValueError):
            _ = ParquetDataset(config)

    def test_full_loading(self, parquet_dataset_path):
        """
        Load all files, without target column separation.
        """
        config = DatasetConfig(
            path=parquet_dataset_path, file_type="parquet", target_column=None
        )
        dataset = ParquetDataset(config)

        # check all files were loaded
        assert len(dataset) == _NUM_CLASSES * len(EventPrefixEnum)

        for e in dataset:
            assert list(e["signal"].columns) == _SIGNAL_NAMES + [
                _LABEL_NAME
            ]  # _LABEL_NAME maps to signals
            assert e["signal"].shape == (_NUM_ROWS, _NUM_SIGNALS + 1)
            assert "label" not in e

    def test_target_loading(self, parquet_dataset_path):
        """
        Load all files, with target column separation.
        """
        config = DatasetConfig(
            path=parquet_dataset_path, file_type="parquet", target_column=_LABEL_NAME
        )
        dataset = ParquetDataset(config)

        assert len(dataset) == _NUM_CLASSES * len(EventPrefixEnum)

        for e in dataset:
            assert list(e["signal"].columns) == _SIGNAL_NAMES
            assert e["signal"].shape == (_NUM_ROWS, _NUM_SIGNALS)
            assert e["label"].shape == (_NUM_ROWS, 1)

    def test_file_splitting(self, parquet_dataset_path):
        """
        Load only files in flist.
        """

        flist = [
            "0/WELL_test.parquet",
            "1/DRAWN_test.parquet",
            "3/SIMULATED_test.parquet",
        ]
        config = DatasetConfig(
            path=parquet_dataset_path,
            file_type="parquet",
            target_column=_LABEL_NAME,
            split="list",
            file_list=flist,
        )
        dataset = ParquetDataset(config)
        assert len(dataset) == len(flist)

    def test_missing_file(self, parquet_dataset_path):
        """
        Reports missing file.
        """
        flist = ["0/WELL_missing.parquet"]
        config = DatasetConfig(
            path=parquet_dataset_path,
            file_type="parquet",
            target_column=_LABEL_NAME,
            split="list",
            file_list=flist,
        )
        with pytest.raises(RuntimeError):
            _ = ParquetDataset(config)

    def test_event_filtering(self, parquet_dataset_path):
        """
        Load only event_type files.
        """
        event_type = [EventPrefixEnum.REAL]
        config = DatasetConfig(
            path=parquet_dataset_path,
            file_type="parquet",
            target_column=_LABEL_NAME,
            split=None,
            event_type=event_type,
        )
        dataset = ParquetDataset(config)

        assert len(dataset) == _NUM_CLASSES * len(event_type)

        event_type = [EventPrefixEnum.REAL, EventPrefixEnum.SIMULATED]
        config = DatasetConfig(
            path=parquet_dataset_path,
            file_type="parquet",
            target_column=_LABEL_NAME,
            split=None,
            event_type=event_type,
        )
        dataset = ParquetDataset(config)

        assert len(dataset) == _NUM_CLASSES * len(event_type)

    def test_target_filtering(self, parquet_dataset_path):
        """
        Load only target classes.
        """
        target_class = [
            0,
        ]
        config = DatasetConfig(
            path=parquet_dataset_path,
            file_type="parquet",
            target_column=_LABEL_NAME,
            split=None,
            target_class=target_class,
        )
        dataset = ParquetDataset(config)
        assert len(dataset) == len(target_class) * len(EventPrefixEnum)

        target_class = [0, 2]
        config = DatasetConfig(
            path=parquet_dataset_path,
            file_type="parquet",
            target_column=_LABEL_NAME,
            split=None,
            target_class=target_class,
        )
        dataset = ParquetDataset(config)
        assert len(dataset) == len(target_class) * len(EventPrefixEnum)
