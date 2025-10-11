import pytest
import pandas as pd
import numpy as np

from pandas.testing import assert_series_equal, assert_frame_equal
from numpy.testing import assert_allclose

from ThreeWToolkit.core.base_preprocessing import (
    ImputeMissingConfig,
    NormalizeConfig,
    RenameColumnsConfig,
    WindowingConfig,
)
from ThreeWToolkit.preprocessing._data_processing import (
    ImputeMissing,
    Normalize,
    RenameColumns,
    Windowing,
)


class TestImputeMissingData:
    def test_impute_mean_dataframe(self):
        """
        Test that imputing with strategy 'mean' replaces NaNs in all DataFrame columns with the column mean.
        """
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
        imp_missing = ImputeMissing(ImputeMissingConfig(strategy="mean"))
        result = imp_missing(data=df)
        expected_result = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})

        assert_frame_equal(result, expected_result)

    def test_impute_median_dataframe_specific_column(self):
        """
        Test that imputing with strategy 'median' replaces NaNs only in specified columns.
        Other columns remain unchanged.
        """
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 5.0, 6.0]})
        imp_missing = ImputeMissing(
            ImputeMissingConfig(strategy="median", columns=["a"])
        )
        result = imp_missing(data=df)
        expected_result = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [np.nan, 5.0, 6.0]})

        assert_frame_equal(result, expected_result)

    def test_impute_constant_series(self):
        """
        Test that imputing a Series with strategy 'constant' replaces NaNs with the provided fill value.
        """
        series = pd.Series([1.0, np.nan, 3.0])
        imp_missing = ImputeMissing(
            ImputeMissingConfig(strategy="constant", fill_value=99.0)
        )
        result = imp_missing(data=series)

        expected_result = pd.Series([1.0, 99.0, 3.0], name="__temp__")

        assert_series_equal(result, expected_result)

    def test_impute_multiple_columns(self):
        """
        Test imputing multiple columns with strategy 'mean', ensuring NaNs are replaced correctly.
        """
        df = pd.DataFrame(
            {
                "a": [np.nan, 2.0, np.nan],
                "b": [5.0, np.nan, 7.0],
                "c": [9.0, 10.0, 11.0],
            }
        )

        imp_missing = ImputeMissing(ImputeMissingConfig(strategy="mean"))
        result = imp_missing(data=df)

        expected_result = pd.DataFrame(
            {"a": [2.0, 2.0, 2.0], "b": [5.0, 6.0, 7.0], "c": [9.0, 10.0, 11.0]}
        )

        assert_frame_equal(result, expected_result)

    def test_impute_multiple_columns_constant(self):
        """
        Test imputing multiple columns with strategy 'constant', checking that NaNs are replaced with the provided constant fill value.
        """
        df = pd.DataFrame(
            {"x": [np.nan, 2, 3], "y": [4, np.nan, np.nan], "z": [1, 1, 1]}
        )

        imp_missing = ImputeMissing(
            ImputeMissingConfig(strategy="constant", fill_value=-1)
        )
        result = imp_missing(data=df)

        expected_result = pd.DataFrame(
            {"x": [-1, 2, 3], "y": [4, -1, -1], "z": [1, 1, 1]}
        )

        assert_frame_equal(result, expected_result, check_dtype=False)

    def test_returns_input_when_column_not_found(self):
        """
        Test that the DataFrame is returned unchanged when none of the specified
        columns for imputation exist in the DataFrame.
        """
        df = pd.DataFrame({"a": [1.0, np.nan]})

        imp_missing = ImputeMissing(
            ImputeMissingConfig(strategy="mean", columns=["missing_column"])
        )
        result = imp_missing(data=df.copy())

        pd.testing.assert_frame_equal(result, df)

    def test_raises_error_on_non_numeric_column(self):
        """
        Test that a TypeError is raised when attempting to impute a non-numeric column.
        """
        df = pd.DataFrame({"a": [1.0, np.nan], "b": ["x", "y"]})

        with pytest.raises(TypeError, match="Only numeric columns can be imputed"):
            imp_missing = ImputeMissing(
                ImputeMissingConfig(strategy="mean", columns=["b"])
            )
            _ = imp_missing(data=df)

    def test_raises_error_if_fill_value_not_provided(self):
        """
        Test that a ValueError is raised if strategy is 'constant' but no fill_value is provided.
        """
        df = pd.DataFrame({"a": [1.0, np.nan]})

        with pytest.raises(
            ValueError, match="You must provide `fill_value` when strategy='constant'"
        ):
            imp_missing = ImputeMissing(ImputeMissingConfig(strategy="constant"))
            _ = imp_missing(data=df)


class TestNormalize:
    def test_normalize_dataframe_l2_axis1(self):
        """Test L2 normalization across rows of a DataFrame."""
        df = pd.DataFrame({"x": [3.0, 0.0], "y": [4.0, 0.0]})
        norm = Normalize(NormalizeConfig(norm="l2", axis=1))
        result = norm(data=df)
        expected_result = pd.DataFrame({"x": [0.6, 0.0], "y": [0.8, 0.0]})

        assert_frame_equal(result, expected_result)

    def test_normalize_dataframe_l1_axis0(self):
        """Test L1 normalization across columns of a DataFrame."""
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 6.0]})
        norm = Normalize(NormalizeConfig(norm="l1", axis=0))
        result = norm(data=df)
        expected_result = pd.DataFrame({"a": [1 / 3, 2 / 3], "b": [1 / 3, 2 / 3]})

        assert_frame_equal(result, expected_result)

    def test_normalize_series_l2(self):
        """Test L2 normalization of a Series."""
        s = pd.Series([3.0, 4.0], name="s")
        norm = Normalize(NormalizeConfig(norm="l2", axis=0))
        result = norm(data=s)
        expected_result = pd.Series([0.6, 0.8], name="s")

        assert_series_equal(result, expected_result)

    def test_normalize_series_max(self):
        """Test max normalization of a Series."""
        s = pd.Series([2.0, 8.0])
        norm = Normalize(NormalizeConfig(norm="max", axis=0))
        result = norm(data=s)
        expected_result = pd.Series([0.25, 1.0])

        assert_series_equal(result, expected_result)

    def test_normalize_return_norm(self):
        """Test return of normalization + norm values."""
        df = pd.DataFrame({"x": [3.0, 0.0], "y": [4.0, 0.0]})
        norm = Normalize(NormalizeConfig(norm="l2", axis=1, return_norm_values=True))
        result, norm = norm(data=df)
        expected_result = pd.DataFrame({"x": [0.6, 0.0], "y": [0.8, 0.0]})
        expected_norm = np.array([[5.0], [0.0]])

        assert_frame_equal(result, expected_result)
        assert_allclose(norm, expected_norm)

    def test_non_numeric_dataframe_raises_type_error(self):
        """Test error raised when DataFrame has non-numeric column."""
        df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})

        with pytest.raises(TypeError, match="Non-numeric columns"):
            norm = Normalize(NormalizeConfig())
            _ = norm(data=df)

    def test_non_numeric_series_raises_type_error(self):
        """Test error raised when Series is non-numeric."""
        s = pd.Series(["a", "b"])

        with pytest.raises(TypeError, match="Series must be numeric"):
            norm = Normalize(NormalizeConfig())
            _ = norm(data=s)


class TestWindowingPercentOverlap:
    def test_validate_window_with_correct_tuple_params(self):
        """Test that windowing works correctly with valid tuple window parameters."""
        series = pd.Series(np.arange(50))
        wind = Windowing(WindowingConfig(window=("kaiser", 0.5), window_size=5))
        result = wind(data=series)

        assert result.shape[0] == 10

    def test_basic_windowing_series_no_overlap(self):
        """Test windowing with no overlap on a Series. The number of rows should be total_len // window_size."""
        series = pd.Series(np.arange(40))
        win_size = 4

        wind = Windowing(WindowingConfig(window_size=win_size, overlap=0.0))
        result = wind(data=series)

        expected_shape = (
            10,
            win_size + 1,
        )  # 10 windows, each with 4 values + win column

        assert result.shape == expected_shape

    def test_basic_windowing_series_normalize(self):
        """Test if normalization is applied correctly when using a 'boxcar' window."""
        series = pd.Series(np.arange(12))
        win_size = 4

        wind = Windowing(
            WindowingConfig(
                window="boxcar", window_size=win_size, overlap=0.0, normalize=True
            )
        )
        result = wind(data=series)

        expected_result = [
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [1.0, 1.25, 1.5, 1.75, 2.0],
            [2.0, 2.25, 2.5, 2.75, 3.0],
        ]

        assert np.allclose(result.values, expected_result, atol=1e-6)

    def test_basic_windowing_series_50_percent_overlap_without_padding(self):
        """Test windowing with 50% overlap and no padding. Shape should match expected window count."""
        series = pd.Series(np.arange(40))
        win_size = 4

        wind = Windowing(WindowingConfig(window_size=win_size, overlap=0.5))
        result = wind(data=series)

        expected_shape = (19, win_size + 1)

        assert result.shape == expected_shape

    def test_windowing_dataframe_overlap(self):
        """Test windowing applied to a column from a DataFrame, validating shape."""
        df = pd.DataFrame({"a": np.arange(10), "b": np.arange(10, 20)})
        win_size = 4

        wind = Windowing(WindowingConfig(window_size=win_size, overlap=0.5))
        result = wind(data=df["a"])

        step = int(3 * (1 - 0.33))
        expected_rows = (10 - 3) // step + 1
        expected_cols = win_size + 1

        assert result.shape == (expected_rows, expected_cols)

    def test_no_padding_when_even_division(self):
        """Ensure no extra row is added when the signal is evenly divisible by step size."""
        series = pd.Series(np.arange(12))
        wind = Windowing(
            WindowingConfig(window_size=4, overlap=0.0, pad_last_window=True)
        )
        result = wind(data=series)

        assert result.shape[0] == 3

    def test_invalid_window_name_string(self):
        """Should raise ValueError if string window name is unknown."""
        with pytest.raises(ValueError, match="Invalid window name 'unknownwin'"):
            wind = Windowing(WindowingConfig(window="unknownwin", window_size=4))
            _ = wind(data=pd.Series(np.arange(10)))

    def test_window_size_larger_than_signal_raises(self):
        """Test that an error is raised if the window size is larger than the series length."""
        series = pd.Series(np.arange(5))

        with pytest.raises(ValueError, match="window_size.*length of X"):
            wind = Windowing(WindowingConfig(window_size=10, overlap=0.0))
            _ = wind(data=series)

    def test_tuple_with_missing_param_raises(self):
        """Tuple with missing required parameters should raise ValueError."""

        with pytest.raises(ValueError, match="requires 1 parameter.*got 0"):
            wind = Windowing(WindowingConfig(window=("kaiser",), window_size=10))
            _ = wind(data=pd.Series(np.arange(100)))

    def test_tuple_with_unknown_window_raises(self):
        """Unknown window name in tuple should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown window name"):
            wind = Windowing(
                WindowingConfig(window=("unknownwin", 1.0), window_size=10)
            )
            _ = wind(data=pd.Series(np.arange(100)))

    def test_string_with_required_param_raises(self):
        """Using a param-required window as string should raise ValueError."""
        with pytest.raises(ValueError, match="requires parameter.*tuple like"):
            wind = Windowing(WindowingConfig(window="kaiser", window_size=10))
            _ = wind(data=pd.Series(np.arange(100)))

    def test_empty_tuple_raises(self):
        """An empty tuple should raise ValueError."""
        with pytest.raises(ValueError, match="Tuple window must start with"):
            wind = Windowing(WindowingConfig(window=(), window_size=10))
            _ = wind(data=pd.Series(np.arange(100)))

    def test_valid_optional_param_window_as_str(self):
        """Windows that can be used as string only should work."""
        series = pd.Series(np.arange(50))
        wind = Windowing(WindowingConfig(window="hann", window_size=5))
        result = wind(data=series)

        assert isinstance(result, pd.DataFrame)

    def test_valid_optional_param_window_as_tuple(self):
        """Windows that accept optional parameters should also work in tuple form."""
        series = pd.Series(np.arange(50))
        wind = Windowing(WindowingConfig(window="hann", window_size=5))
        result = wind(data=series)

        assert isinstance(result, pd.DataFrame)


class TestRenameColumns:
    def setup_method(self):
        """
        Prepare example DataFrames used in the test cases.
        """
        self.df_full = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})

        self.df_partial = pd.DataFrame({"A": [10, 20], "B": [30, 40]})

        self.df_empty = pd.DataFrame(columns=["A", "B"])

    def test_functional_case(self):
        """
        Test renaming multiple columns while keeping others unchanged.
        """
        columns_map = {"A": "X", "B": "Y"}
        rename = RenameColumns(RenameColumnsConfig(columns_map=columns_map))
        result = rename(self.df_full)

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
            ValueError, match="Columns not found in DataFrame: \\['Z'\\]"
        ):
            rename = RenameColumns(RenameColumnsConfig(columns_map=columns_map))
            _ = rename(self.df_partial)

    def test_empty_map_returns_same_dataframe(self):
        """
        Test that an empty mapping returns the original DataFrame unchanged.
        """
        rename = RenameColumns(RenameColumnsConfig(columns_map={}))
        result = rename(self.df_partial)
        pd.testing.assert_frame_equal(result, self.df_partial)

    def test_empty_dataframe(self):
        """
        Test renaming columns in an empty DataFrame (only headers, no rows).
        """
        columns_map = {"A": "X"}
        rename = RenameColumns(RenameColumnsConfig(columns_map=columns_map))
        result = rename(self.df_empty)

        assert list(result.columns) == ["X", "B"]
        assert result.empty

    def test_duplicate_new_column_names(self):
        """
        Test that duplicate new column names raise a ValueError.
        """
        columns_map = {"A": "X", "B": "X"}
        with pytest.raises(
            ValueError, match="Duplicate new column names are not allowed."
        ):
            rename = RenameColumns(RenameColumnsConfig(columns_map=columns_map))
            _ = rename(self.df_partial)

    def test_same_name_in_mapping(self):
        """
        Test that mapping a column to itself leaves the DataFrame unchanged.
        """
        columns_map = {"A": "A"}
        rename = RenameColumns(RenameColumnsConfig(columns_map=columns_map))
        result = rename(self.df_partial)
        pd.testing.assert_frame_equal(result, self.df_partial)

    def test_duplicated_columns_in_dataframe_raise_error(self):
        """
        Test that a ValueError is raised when the DataFrame has duplicate column names.
        """
        df_with_duplicates = pd.DataFrame([[1, 2]], columns=["A", "A"])
        columns_map = {"A": "X"}

        with pytest.raises(
            ValueError, match="Duplicate column names found in DataFrame: \\['A'\\]"
        ):
            rename = RenameColumns(RenameColumnsConfig(columns_map=columns_map))
            _ = rename(df_with_duplicates)
