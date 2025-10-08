from pathlib import Path
import pytest

import numpy as np
import pandas as pd

from ThreeWToolkit.dataset import ParquetDataset
from ThreeWToolkit.core.base_dataset import EventPrefixEnum, ParquetDatasetConfig
from ThreeWToolkit.utils.data_utils import GLOBAL_AVERAGES


_NUM_ROWS = 10
_NUM_CLASSES = 3
_NUM_EVENTS = 1
_LABEL_NAME = "class"


def make_parquet_event(path: Path):
    """
    Create a single parquet file with random signals and labels, using columns
    compatible with GLOBAL_AVERAGES/STD.
    """
    signal_columns = list(GLOBAL_AVERAGES.keys())
    data = np.random.randn(_NUM_ROWS, len(signal_columns)).astype(np.float32)
    df = pd.DataFrame(data, columns=signal_columns)

    # add label column if not already present
    df[_LABEL_NAME] = np.random.randint(low=0, high=_NUM_CLASSES, size=_NUM_ROWS)

    df.to_parquet(path, engine="pyarrow", compression="brotli")


@pytest.fixture(scope="session")
def parquet_dataset_path(tmp_path_factory):
    base_path = tmp_path_factory.mktemp("data")
    for class_idx in range(_NUM_CLASSES):
        dir_c = base_path / str(class_idx)
        dir_c.mkdir()
        for prefix in EventPrefixEnum:
            for event_idx in range(_NUM_EVENTS):
                file_path = dir_c / f"{prefix.value}_event{event_idx}.parquet"
                make_parquet_event(file_path)
    return base_path


@pytest.mark.skip(
    reason="This test class was disabled so that we can think about a better way to test the dataset download."
)
class TestParquetDataset:
    def test_full_loading(self, parquet_dataset_path):
        """
        Load all files, without target column separation.
        """
        config = ParquetDatasetConfig(
            path=str(parquet_dataset_path), target_column=_LABEL_NAME, clean_data=False
        )
        dataset = ParquetDataset(config)

        # Ajuste: 1 evento por classe/prefixo
        expected_files = _NUM_CLASSES * len(EventPrefixEnum)
        assert len(dataset) == expected_files

        for e in dataset:
            signal_cols = list(e["signal"].columns)
            expected_cols = list(GLOBAL_AVERAGES.keys())

            assert all(
                col in expected_cols or col == _LABEL_NAME for col in signal_cols
            )

    def test_file_splitting(self, parquet_dataset_path):
        """
        Load only files in file_list.
        """
        # Usar apenas os arquivos existentes
        flist = [f"0/{prefix.value}_event0.parquet" for prefix in EventPrefixEnum]
        config = ParquetDatasetConfig(
            path=str(parquet_dataset_path),
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
        flist = ["0/UNKNOWN_event0.parquet"]
        config = ParquetDatasetConfig(
            path=str(parquet_dataset_path),
            target_column=_LABEL_NAME,
            split="list",
            file_list=flist,
        )
        with pytest.raises(RuntimeError):
            _ = ParquetDataset(config)

    def test_event_filtering(self, parquet_dataset_path):
        """
        Load only files matching event_type.
        """
        event_type = [EventPrefixEnum.REAL]
        config = ParquetDatasetConfig(
            path=str(parquet_dataset_path),
            target_column=_LABEL_NAME,
            split=None,
            event_type=event_type,
        )
        dataset = ParquetDataset(config)
        assert len(dataset) == _NUM_CLASSES * len(event_type)

        # Múltiplos tipos de evento
        event_type = [EventPrefixEnum.REAL, EventPrefixEnum.SIMULATED]
        config = ParquetDatasetConfig(
            path=str(parquet_dataset_path),
            target_column=_LABEL_NAME,
            split=None,
            event_type=event_type,
        )
        dataset = ParquetDataset(config)
        assert len(dataset) == _NUM_CLASSES * len(event_type)

    def test_target_filtering(self, parquet_dataset_path):
        """
        Load only specific target classes.
        """
        target_class = [0]
        config = ParquetDatasetConfig(
            path=str(parquet_dataset_path),
            target_column=_LABEL_NAME,
            split=None,
            target_class=target_class,
        )
        dataset = ParquetDataset(config)
        assert len(dataset) == len(target_class) * len(EventPrefixEnum)

        target_class = [0, 2]
        config = ParquetDatasetConfig(
            path=str(parquet_dataset_path),
            target_column=_LABEL_NAME,
            split=None,
            target_class=target_class,
        )
        dataset = ParquetDataset(config)
        assert len(dataset) == len(target_class) * len(EventPrefixEnum)

    def test_clean_data_processing(self, parquet_dataset_path):
        """
        Validate that pre_process cleans and fills the target column correctly
        when `clean_data=True`.
        """
        config = ParquetDatasetConfig(
            path=str(parquet_dataset_path), target_column=_LABEL_NAME, clean_data=True
        )
        dataset = ParquetDataset(config)

        for e in dataset:
            assert "signal" in e
            assert "label" in e
            # Todas as colunas de sinal devem estar em GLOBAL_AVERAGES
            for col in e["signal"].columns:
                assert col in GLOBAL_AVERAGES
            # Nenhum NaN após limpeza
            assert not e["signal"].isna().any().any()
