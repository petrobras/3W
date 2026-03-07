import pandas as pd
from ..core.base_step import BaseStep
from pydantic import BaseModel, Field, ValidationInfo, field_validator


class RenameColumnsConfig(BaseModel):
    columns_map: dict[str, str]
    target: type = Field(default_factory=lambda: RenameColumns)

    @field_validator("columns_map")
    def validate_columns_exist(cls, columns_map: dict[str, str], info: ValidationInfo):
        """
        Validate that all columns to be renamed exist in the DataFrame.

        Args:
            cls: The class reference.
            columns_map (dict[str, str]): Mapping of columns to rename.
            info: Validation info, containing the data attribute.

        Raises:
            ValueError: If any column in columns_map does not exist in the DataFrame.

        Returns:
            dict[str, str]: The validated columns_map.
        """
        df: pd.DataFrame | None = info.data.get("data")
        if df is not None:
            missing = [col for col in columns_map if col not in df.columns]
            if missing:
                raise ValueError(f"Columns not found in DataFrame: {missing}")
        return columns_map

    @field_validator("columns_map")
    def validate_unique_new_column_names(cls, columns_map: dict[str, str]):
        """
        Validate that new column names are unique.

        Args:
            cls: The class reference.
            columns_map (dict[str, str]): Mapping of columns to rename.

        Raises:
            ValueError: If there are duplicate new column names.

        Returns:
            dict[str, str]: The validated columns_map.
        """
        new_names = list(columns_map.values())
        if len(new_names) != len(set(new_names)):
            raise ValueError("Duplicate new column names are not allowed.")
        return columns_map


class RenameColumns(BaseStep):
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

    def run(self, data: dict) -> dict:
        """
        Rename columns according to the configured mapping.

        This method applies the column renaming using pandas' rename method with the
        mapping provided in the configuration.

        Args:
            data (pd.DataFrame): DataFrame with columns to be renamed

        Returns:
            dict: Event data with renamed columns
        """
        signal_df = data.get("signal")
        if signal_df is None or not isinstance(signal_df, pd.DataFrame):
            return data  # Return unchanged if no signal data

        if signal_df.columns.duplicated().any():
            duplicated = (
                signal_df.columns[signal_df.columns.duplicated()].unique().tolist()
            )
            raise ValueError(f"Duplicate column names found in DataFrame: {duplicated}")

        missing = [
            col for col in self.config.columns_map if col not in signal_df.columns
        ]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")

        signal_df = signal_df.rename(columns=self.config.columns_map)
        result = data.copy()
        result["signal"] = signal_df
        return result
