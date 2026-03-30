from pydantic import Field

from ..core.base_preprocessing import BasePreprocessing, BasePreprocessingConfig
from ..core.dataset_outputs import DatasetOutputs


class RenameColumnsConfig(BasePreprocessingConfig):
    columns_map: dict[str, str]
    target_: type = Field(default_factory=lambda: RenameColumns)

class RenameColumns(BasePreprocessing):
    """
    A simple data processing step that renames DataFrame columns according to a mapping.

    This class provides a clean interface for renaming columns in a pandas DataFrame
    using a dictionary mapping from old names to new names.

    Attributes:
        config (RenameColumnsConfig): Configuration object containing the column mapping
    """

    def __init__(self, config: RenameColumnsConfig):
        """
        Initialize the RenameColumns step with the provided configuration.

        Args:
            config (RenameColumnsConfig): Configuration containing the columns_map dictionary
        """
        self.config = config

    def transform(self, data: DatasetOutputs) -> DatasetOutputs:
        """
        Rename columns according to the configured mapping.

        Args:
            data: DatasetOutputs object containing signal DataFrame

        Returns:
            DatasetOutputs: Data with renamed signal columns
        """

        signal_columns = data.signal.columns
        missing_columns = [col for col in self.config.columns_map.keys() if col not in signal_columns]
        if missing_columns:
            raise ValueError(f"RenameColumns: The following columns specified in columns_map are not present in the signal DataFrame: {missing_columns}")

        data.signal = data.signal.rename(columns=self.config.columns_map)
        return data
