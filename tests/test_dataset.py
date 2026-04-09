from pathlib import Path
import pytest

from ThreeWToolkit.dataset import ParquetDatasetConfig


@pytest.fixture(scope="session")
def parquet_dataset_path() -> Path:
    """Point to the directory where the 3W dataset is stored."""
    return Path(__file__).parent.parent / "dataset"


_COLUMNS = [
    "T-TPT",
    "P-TPT",
    "P-JUS-CKGL",
    "P-MON-CKGL",
]  # some signal variables in the 2.0.0 version of the dataset
_LABEL_NAME = "class"


class TestParquetDataset:
    def test_full_loading(self, parquet_dataset_path: Path):
        """
        Load all files, without target column separation.
        """
        config = ParquetDatasetConfig(
            path=parquet_dataset_path,
            columns=_COLUMNS,
            target_column=_LABEL_NAME,
        )
        dataset = config.build()

        assert (
            len(dataset) == 2228
        )  # known number of events in the dataset (2.0.0 version)

        for e in dataset:
            signal_cols = list(e.signal.columns)
            expected_cols = _COLUMNS

            assert all(
                col in expected_cols or col == _LABEL_NAME for col in signal_cols
            )

    def test_file_splitting(self, parquet_dataset_path):
        """
        Load only files in file_list.
        """
        # Usar apenas os arquivos existentes
        flist = ["0/WELL-00001_20170201010207.parquet", "1/DRAWN_00001.parquet"]
        config = ParquetDatasetConfig(
            path=parquet_dataset_path,
            target_column=_LABEL_NAME,
            split="list",
            file_list=flist,
        )
        dataset = config.build()
        assert len(dataset) == len(flist)

    def test_missing_file(self, parquet_dataset_path):
        """
        Reports missing file.
        """
        flist = ["0/UNKNOWN_event0.parquet"]
        config = ParquetDatasetConfig(
            path=parquet_dataset_path,
            target_column=_LABEL_NAME,
            split="list",
            file_list=flist,
        )
        with pytest.raises(RuntimeError):
            _ = config.build()

    def test_event_filtering(self, parquet_dataset_path):
        """
        Load only files matching event_type.
        """
        config = ParquetDatasetConfig(
            path=parquet_dataset_path,
            target_column=_LABEL_NAME,
            split=None,
            event_type=["real"],
        )
        dataset = config.build()
        assert (
            len(dataset) == 1119
        )  # known number of real events in the dataset (2.0.0 version)

        # Múltiplos tipos de evento
        config = ParquetDatasetConfig(
            path=parquet_dataset_path,
            target_column=_LABEL_NAME,
            split=None,
            event_type=["real", "simulated"],
        )
        dataset = config.build()
        assert (
            len(dataset) == 2208
        )  # known number of real + simulated events in the dataset (2.0.0 version)

    def test_target_filtering(self, parquet_dataset_path):
        """
        Load only specific target classes.
        """
        target_class = [0]
        config = ParquetDatasetConfig(
            path=parquet_dataset_path,
            target_column=_LABEL_NAME,
            split=None,
            target_class=target_class,
        )
        dataset = config.build()
        assert (
            len(dataset) == 594
        )  # known number of events in class 0 in the dataset (2.0.0 version)

        target_class = [0, 2]
        config = ParquetDatasetConfig(
            path=parquet_dataset_path,
            target_column=_LABEL_NAME,
            split=None,
            target_class=target_class,
        )
        dataset = config.build()
        assert (
            len(dataset) == 632
        )  # known number of events in classes 0 and 2 in the dataset (2.0.0 version)
