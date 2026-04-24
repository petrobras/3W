import pandas as pd
from pathlib import Path


def load_csv(
    file_path: str, column_names: list[str], date_column: list[str], parse_dates: bool
) -> pd.DataFrame:
    """
    Loads a CSV file into a DataFrame with selected columns.

    Args:
        file_path (str): Path to the CSV file.
        column_names (list[str]): List of column names to load.
        date_column (list[str]): List of columns to be parsed as dates.
        parse_dates (bool): If True, parse the date columns.

    Returns:
        pd.DataFrame: The loaded DataFrame with selected columns.
    """
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    if not isinstance(column_names, list) or not all(
        isinstance(c, str) for c in column_names
    ):
        raise ValueError("`column_names` must be a list of strings.")
    if not isinstance(date_column, list) or not all(
        isinstance(c, str) for c in date_column
    ):
        raise ValueError("`date_column` must be a list of strings.")
    if not isinstance(parse_dates, bool):
        raise ValueError("`parse_dates` must be a boolean.")

    try:
        df = pd.read_csv(
            file_path,
            usecols=column_names,
            parse_dates=date_column if parse_dates else False,
        )
        return df

    except Exception as e:
        raise RuntimeError(f"Failed to load CSV '{file_path}': {str(e)}")
