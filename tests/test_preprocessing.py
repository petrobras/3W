import pytest
import pandas as pd
import numpy as np

from pandas.testing import assert_series_equal, assert_frame_equal
from numpy.testing import assert_allclose

from ThreeWToolkit.preprocessing import impute_missing_data, normalize, windowing


class TestImputeMissingData:
    def test_impute_mean_dataframe(self):
        """
        Test that imputing with strategy 'mean' replaces NaNs in all DataFrame columns with the column mean.
        """
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
        result = impute_missing_data(data=df, strategy="mean")
        expected_result = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})

        assert_frame_equal(result, expected_result)

    def test_impute_median_dataframe_specific_column(self):
        """
        Test that imputing with strategy 'median' replaces NaNs only in specified columns.
        Other columns remain unchanged.
        """
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 5.0, 6.0]})
        result = impute_missing_data(data=df, strategy="median", columns=["a"])
        expected_result = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [np.nan, 5.0, 6.0]})

        assert_frame_equal(result, expected_result)

    def test_impute_constant_series(self):
        """
        Test that imputing a Series with strategy 'constant' replaces NaNs with the provided fill value.
        """
        series = pd.Series([1.0, np.nan, 3.0])
        result = impute_missing_data(data=series, strategy="constant", fill_value=99.0)
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

        result = impute_missing_data(data=df, strategy="mean")

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

        result = impute_missing_data(data=df, strategy="constant", fill_value=-1)

        expected_result = pd.DataFrame(
            {"x": [-1, 2, 3], "y": [4, -1, -1], "z": [1, 1, 1]}
        )

        assert_frame_equal(result, expected_result, check_dtype=False)

    def test_raises_error_when_column_not_found(self):
        """
        Test that a ValueError is raised when a specified column for imputation does not exist in the DataFrame.
        """
        df = pd.DataFrame({"a": [1.0, np.nan]})

        with pytest.raises(ValueError, match="Columns not found"):
            impute_missing_data(data=df, strategy="mean", columns=["missing_column"])

    def test_raises_error_on_non_numeric_column(self):
        """
        Test that a TypeError is raised when attempting to impute a non-numeric column.
        """
        df = pd.DataFrame({"a": [1.0, np.nan], "b": ["x", "y"]})

        with pytest.raises(TypeError, match="Only numeric columns can be imputed"):
            impute_missing_data(data=df, strategy="mean", columns=["b"])

    def test_raises_error_if_fill_value_not_provided(self):
        """
        Test that a ValueError is raised if strategy is 'constant' but no fill_value is provided.
        """
        df = pd.DataFrame({"a": [1.0, np.nan]})

        with pytest.raises(ValueError, match="You must provide `fill_value`"):
            impute_missing_data(data=df, strategy="constant")


class TestNormalize:
    def test_normalize_dataframe_l2_axis1(self):
        """Test L2 normalization across rows of a DataFrame."""
        df = pd.DataFrame({"x": [3.0, 0.0], "y": [4.0, 0.0]})
        result = normalize(X=df, norm="l2", axis=1)
        expected_result = pd.DataFrame({"x": [0.6, 0.0], "y": [0.8, 0.0]})

        assert_frame_equal(result, expected_result)

    def test_normalize_dataframe_l1_axis0(self):
        """Test L1 normalization across columns of a DataFrame."""
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 6.0]})
        result = normalize(X=df, norm="l1", axis=0)
        expected_result = pd.DataFrame({"a": [1 / 3, 2 / 3], "b": [1 / 3, 2 / 3]})

        assert_frame_equal(result, expected_result)

    def test_normalize_series_l2(self):
        """Test L2 normalization of a Series."""
        s = pd.Series([3.0, 4.0], name="s")
        result = normalize(X=s, norm="l2", axis=0)
        expected_result = pd.Series([0.6, 0.8], name="s")

        assert_series_equal(result, expected_result)

    def test_normalize_series_max(self):
        """Test max normalization of a Series."""
        s = pd.Series([2.0, 8.0])
        result = normalize(X=s, norm="max", axis=0)
        expected_result = pd.Series([0.25, 1.0])

        assert_series_equal(result, expected_result)

    def test_normalize_return_norm(self):
        """Test return of normalization + norm values."""
        df = pd.DataFrame({"x": [3.0, 0.0], "y": [4.0, 0.0]})
        result, norm = normalize(X=df, norm="l2", axis=1, return_norm_values=True)
        expected_result = pd.DataFrame({"x": [0.6, 0.0], "y": [0.8, 0.0]})
        expected_norm = np.array([[5.0], [0.0]])

        assert_frame_equal(result, expected_result)
        assert_allclose(norm, expected_norm)

    def test_non_numeric_dataframe_raises_type_error(self):
        """Test error raised when DataFrame has non-numeric column."""
        df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})

        with pytest.raises(TypeError, match="Non-numeric columns"):
            normalize(X=df)

    def test_non_numeric_series_raises_type_error(self):
        """Test error raised when Series is non-numeric."""
        s = pd.Series(["a", "b"])

        with pytest.raises(TypeError, match="Series must be numeric"):
            normalize(X=s)


class TestWindowingPercentOverlap:
    def test_basic_windowing_series_no_overlap(self):
        """Test windowing with no overlap on a Series. The number of rows should be total_len // window_size."""
        series = pd.Series(np.arange(40))
        win_size = 4

        result = windowing(X=series, window_size=win_size, overlap=0.0)
        expected_shape = (
            10,
            win_size + 1,
        )  # 10 windows, each with 4 values + win column

        assert result.shape == expected_shape

    def test_basic_windowing_series_normalize(self):
        """Test if normalization is applied correctly when using a 'boxcar' window."""
        series = pd.Series(np.arange(12))
        win_size = 4

        result = windowing(
            X=series, window="boxcar", window_size=win_size, overlap=0.0, normalize=True
        )
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

        result = windowing(X=series, window_size=win_size, overlap=0.5)
        expected_shape = (19, win_size + 1)

        assert result.shape == expected_shape

    def test_windowing_dataframe_overlap(self):
        """Test windowing applied to a column from a DataFrame, validating shape."""
        df = pd.DataFrame({"a": np.arange(10), "b": np.arange(10, 20)})
        win_size = 4

        result = windowing(X=df["a"], window_size=win_size, overlap=0.5)
        step = int(3 * (1 - 0.33))
        expected_rows = (10 - 3) // step + 1
        expected_cols = win_size + 1

        assert result.shape == (expected_rows, expected_cols)

    def test_padding_adds_last_window_series(self):
        """Test if padding adds an additional window when signal length is not divisible by step."""
        series = pd.Series(np.arange(103))
        win_size = 4
        overlap = 0

        res_no_pad = windowing(
            X=series,
            window="boxcar",
            window_size=win_size,
            overlap=overlap,
            pad_last_window=False,
        )

        res_pad = windowing(
            X=series,
            window="boxcar",
            window_size=win_size,
            overlap=overlap,
            pad_last_window=True,
            pad_value=-1.0,
        )

        # The padded version should have one more row
        assert res_pad.shape[0] == res_no_pad.shape[0] + 1
        assert len([x for x in res_pad.columns if "val_" in x]) == win_size

        # The last row should contain padding values at the end
        last_row = res_pad.iloc[-1].values
        original_remainder = 10 - ((res_no_pad.shape[0] - 1) * 2 + win_size)
        assert (last_row[-original_remainder:] == -1.0).all()

    def test_padding_dataframe_last_window(self):
        """Test if the last padded row has the specified padding value for all trailing elements."""
        df = pd.DataFrame({"x": np.arange(13), "y": np.linspace(0, 1, 13)})

        res = windowing(
            X=df["x"], window_size=5, overlap=0.2, pad_last_window=True, pad_value=999
        )
        last_row = res.iloc[-1]

        # Check if the last column in the last row is the pad value
        colnames = [col for col in res.columns if col.endswith("_t4")]
        for col in colnames:
            assert last_row[col] == 999

    def test_no_padding_when_even_division(self):
        """Ensure no extra row is added when the signal is evenly divisible by step size."""
        series = pd.Series(np.arange(12))
        res = windowing(X=series, window_size=4, overlap=0.0, pad_last_window=True)

        assert res.shape[0] == 3

    def test_window_size_larger_than_signal_raises(self):
        """Test that an error is raised if the window size is larger than the series length."""
        series = pd.Series(np.arange(5))

        with pytest.raises(ValueError, match="window_size.*length of X"):
            windowing(X=series, window_size=10, overlap=0.0)

    def test_non_numeric_column_raises(self):
        """Test that a TypeError is raised when trying to window a non-numeric column."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        with pytest.raises(TypeError, match="Series must be numeric."):
            windowing(X=df["b"], window_size=2, overlap=0.0)
