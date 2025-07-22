import zipfile
from pathlib import Path
from typing import Any, Dict

from pandas import read_parquet
from ..utils.downloader import get_figshare_data

from ..core.base_dataset import BaseDataset, DatasetConfig


class ParquetDataset(BaseDataset):
    def __init__(self, config: DatasetConfig, download: bool = False):
        """
        Lazy loading of event files. Checks split consistency.
        Download from figshare if needed. Compatible with version 2.0.0.
        """
        super().__init__(config)
        if config.file_type != "parquet":
            raise ValueError("Incompatible file_type.")

        if download:
            dl_path = Path(config.path) / "download"
            dl_path.mkdir(exist_ok=True, parents=True)
            downloaded = get_figshare_data(dl_path, version="2.0.0")
            with zipfile.ZipFile(downloaded[0], "r") as zip_file:
                zip_file.extractall(config.path)

        # search all events
        root = Path(config.path)
        found_events = [e.relative_to(root) for e in root.rglob("*.parquet")]
        found_events = self.filter_events(found_events)

        if config.split not in [None, "list"]:
            raise ValueError("Dataset splitting not implemented.")

        if config.file_list is None:  # TODO: train/val/test splitting
            self.events = found_events
        else:
            not_found = set(Path(p) for p in config.file_list) - set(found_events)
            if len(not_found) > 0:
                raise RuntimeError('"file_list" contains files not found in root path.')
            self.events = [Path(p) for p in config.file_list]

    def __len__(self) -> int:
        """
        Return number of events in dataset.
        """
        return len(self.events)

    def load_data(self, idx: int) -> Dict[str, Any]:
        """
        Return dict for loaded file.
        """
        path = Path(self.config.path) / self.events[idx]
        ret = {}
        ret["signal"] = read_parquet(
            path, columns=self.config.columns, engine="pyarrow"
        )
        if self.config.target_column is not None:
            ret["signal"].drop(columns=[self.config.target_column], inplace=True)
            ret["label"] = read_parquet(
                path, columns=[self.config.target_column], engine="pyarrow"
            )
        return ret
