"""Tests for RenameColumns preprocessing class."""

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from ThreeWToolkit.preprocessing import RenameColumns, RenameColumnsConfig


# Module-level fixtures

@pytest.fixture
def df_full():
    """Full DataFrame with multiple columns."""
    return pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})


@pytest.fixture
def df_partial():
    """Partial DataFrame with fewer columns."""
    return pd.DataFrame({"A": [10, 20], "B": [30, 40]})


@pytest.fixture
def df_empty():
    """Empty DataFrame with only column headers."""
    return pd.DataFrame(columns=["A", "B"])


@pytest.fixture
def df_with_duplicates():
    """DataFrame with duplicate column names."""
    return pd.DataFrame([[1, 2]], columns=["A", "A"])


# Tests for column renaming


class TestRenameColumnsFunctionality:
    """Test basic renaming functionality."""

    def test_functional_case(self, df_full):
        """Test renaming multiple columns while keeping others unchanged."""
        columns_map = {"A": "X", "B": "Y"}
        rename = RenameColumns(RenameColumnsConfig(columns_map=columns_map))
        result = rename(df_full)

        assert list(result.columns) == ["X", "Y", "C"]
        assert result["X"].tolist() == [1, 2]
        assert result["Y"].tolist() == [3, 4]
        assert result["C"].tolist() == [5, 6]

    def test_empty_map_returns_same_dataframe(self, df_partial):
        """Test that an empty mapping returns the original DataFrame unchanged."""
        rename = RenameColumns(RenameColumnsConfig(columns_map={}))
        result = rename(df_partial)
        assert_frame_equal(result, df_partial)

    def test_empty_dataframe(self, df_empty):
        """Test renaming columns in an empty DataFrame (only headers, no rows)."""
        columns_map = {"A": "X"}
        rename = RenameColumns(RenameColumnsConfig(columns_map=columns_map))
        result = rename(df_empty)

        assert list(result.columns) == ["X", "B"]
        assert result.empty

    def test_same_name_in_mapping(self, df_partial):
        """Test that mapping a column to itself leaves the DataFrame unchanged."""
        columns_map = {"A": "A"}
        rename = RenameColumns(RenameColumnsConfig(columns_map=columns_map))
        result = rename(df_partial)
        assert_frame_equal(result, df_partial)

    @pytest.mark.parametrize(
        "columns_map,expected_columns",
        [
            ({"A": "X"}, ["X", "B", "C"]),
            ({"A": "X", "B": "Y"}, ["X", "Y", "C"]),
            ({"A": "X", "B": "Y", "C": "Z"}, ["X", "Y", "Z"]),
            ({"C": "Z"}, ["A", "B", "Z"]),
        ],
    )
    def test_various_rename_patterns(self, df_full, columns_map, expected_columns):
        """Test different renaming patterns parametrized."""
        rename = RenameColumns(RenameColumnsConfig(columns_map=columns_map))
        result = rename(df_full)
        assert list(result.columns) == expected_columns


class TestRenameColumnsEdgeCases:
    """Test edge cases and error handling."""

    def test_column_not_found_raises_error(self, df_partial):
        """Test that a ValueError is raised when a column does not exist."""
        columns_map = {"Z": "W"}
        with pytest.raises(
            ValueError, match="Columns not found in DataFrame: \\['Z'\\]"
        ):
            rename = RenameColumns(RenameColumnsConfig(columns_map=columns_map))
            _ = rename(df_partial)

    def test_duplicate_new_column_names(self, df_partial):
        """Test that duplicate new column names raise a ValueError."""
        columns_map = {"A": "X", "B": "X"}
        with pytest.raises(
            ValueError, match="Duplicate new column names are not allowed."
        ):
            rename = RenameColumns(RenameColumnsConfig(columns_map=columns_map))
            _ = rename(df_partial)

    def test_duplicated_columns_in_dataframe_raise_error(self, df_with_duplicates):
        """Test that a ValueError is raised when DataFrame has duplicate column names."""
        columns_map = {"A": "X"}

        with pytest.raises(
            ValueError, match="Duplicate column names found in DataFrame: \\['A'\\]"
        ):
            rename = RenameColumns(RenameColumnsConfig(columns_map=columns_map))
            _ = rename(df_with_duplicates)

    @pytest.mark.parametrize(
        "columns_map",
        [
            {"NonExistent": "X"},
            {"A": "X", "NonExistent": "Y"},
            {"Missing1": "X", "Missing2": "Y"},
        ],
    )
    def test_multiple_missing_columns(self, df_partial, columns_map):
        """Test error when multiple non-existent columns are in the mapping."""
        with pytest.raises(ValueError, match="Columns not found in DataFrame"):
            rename = RenameColumns(RenameColumnsConfig(columns_map=columns_map))
            _ = rename(df_partial)
