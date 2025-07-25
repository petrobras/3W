import pytest
import pandas as pd

from ThreeWToolkit.preprocessing._data_processing import rename_columns


class TestRenameColumns:
    def setup_method(self):
        """
        Prepare example DataFrames used in the test cases.
        """
        self.df_full = pd.DataFrame({
            "A": [1, 2],
            "B": [3, 4],
            "C": [5, 6]
        })

        self.df_partial = pd.DataFrame({
            "A": [10, 20],
            "B": [30, 40]
        })

        self.df_empty = pd.DataFrame(columns=["A", "B"])

    def test_functional_case(self):
        """
        Test renaming multiple columns while keeping others unchanged.
        """
        columns_map = {"A": "X", "B": "Y"}
        result = rename_columns(self.df_full, columns_map)

        assert list(result.columns) == ["X", "Y", "C"]
        assert result["X"].tolist() == [1, 2]
        assert result["Y"].tolist() == [3, 4]
        assert result["C"].tolist() == [5, 6]

    def test_column_not_found_raises_error(self):
        """
        Test that a ValueError is raised when a column does not exist.
        """
        columns_map = {"Z": "W"}
        with pytest.raises(
            ValueError,
            match="Columns not found in DataFrame: \\['Z'\\]"
        ):
            rename_columns(self.df_partial, columns_map)

    def test_empty_map_returns_same_dataframe(self):
        """
        Test that an empty mapping returns the original DataFrame unchanged.
        """
        result = rename_columns(self.df_partial, {})
        pd.testing.assert_frame_equal(result, self.df_partial)

    def test_empty_dataframe(self):
        """
        Test renaming columns in an empty DataFrame (only headers, no rows).
        """
        columns_map = {"A": "X"}
        result = rename_columns(self.df_empty, columns_map)

        assert list(result.columns) == ["X", "B"]
        assert result.empty

    def test_duplicate_new_column_names(self):
        """
        Test that duplicate new column names raise a ValueError.
        """
        columns_map = {"A": "X", "B": "X"}
        with pytest.raises(
            ValueError,
            match="Duplicate new column names are not allowed."
        ):
            rename_columns(self.df_partial, columns_map)

    def test_same_name_in_mapping(self):
        """
        Test that mapping a column to itself leaves the DataFrame unchanged.
        """
        columns_map = {"A": "A"}
        result = rename_columns(self.df_partial, columns_map)
        pd.testing.assert_frame_equal(result, self.df_partial)
