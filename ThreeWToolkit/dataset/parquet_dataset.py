from pathlib import Path
from typing import Any, Dict

from pandas import read_parquet

from ..core.base_dataset import BaseDataset, DatasetConfig


class ParquetDataset(BaseDataset):
    def __init__(self, config: DatasetConfig):
        """
        Lazy loading of event files. Checks split consistency.
        """
        super().__init__(config)

        if config.file_type != "parquet":
            raise ValueError("Incompatible file_type.")

        # search all events
        found_events = list(Path(config.path).glob("**/*.parquet"))

        if config.split not in [None, "list"]:
            raise ValueError("Dataset plitting not implemented.")

        if config.file_list is None: # TODO: train/val/test splitting
            self.events = found_events
        else:
            not_found = set(Path(p) for p in config.file_list) - set(found_events)
            if len(not_found) > 0:
                raise RuntimeError("\"file_list\" contains paths not found in root path.")
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
        ret = {}
        ret["signal"] = read_parquet(self.events[idx], columns=self.config.columns, engine="pyarrow")
        if self.config.target_column is not None:
            ret["signal"].drop(columns=[self.config.target_column], inplace=True)
            ret["label"] = read_parquet(self.events[idx], columns=[self.config.target_column], engine="pyarrow")
        return ret
