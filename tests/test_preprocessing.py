import pytest
import pandas as pd
import numpy as np

from pandas.testing import assert_series_equal, assert_frame_equal

from ThreeWToolkit.preprocessing import (
    impute_missing_data
)

class TestImputeMissingData:
    def test_impute_mean_dataframe(self):
        """
        Test that imputing with strategy 'mean' replaces NaNs in all DataFrame columns with the column mean.
        """
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
        result = impute_missing_data(data = df, strategy = "mean")
        expected_result = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        
        assert_frame_equal(result, expected_result)

    def test_impute_median_dataframe_specific_column(self):
        """
        Test that imputing with strategy 'median' replaces NaNs only in specified columns.
        Other columns remain unchanged.
        """
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 5.0, 6.0]})
        result = impute_missing_data(data = df, strategy = "median", columns = ["a"])
        expected_result = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [np.nan, 5.0, 6.0]})

        assert_frame_equal(result, expected_result)

    def test_impute_constant_series(self):
        """
        Test that imputing a Series with strategy 'constant' replaces NaNs with the provided fill value.
        """
        series = pd.Series([1.0, np.nan, 3.0])
        result = impute_missing_data(data = series, strategy = "constant", fill_value = 99.0)
        expected_result = pd.Series([1.0, 99.0, 3.0], name = "__temp__")

        assert_series_equal(result, expected_result) 

    def test_impute_multiple_columns(self):
        """
        Test imputing multiple columns with strategy 'mean', ensuring NaNs are replaced correctly.
        """
        df = pd.DataFrame({
            "a": [np.nan, 2.0, np.nan],
            "b": [5.0, np.nan, 7.0],
            "c": [9.0, 10.0, 11.0]
        })

        result = impute_missing_data(data = df, strategy = "mean")

        expected_result = pd.DataFrame({
            "a": [2.0, 2.0, 2.0], 
            "b": [5.0, 6.0, 7.0], 
            "c": [9.0, 10.0, 11.0]
        })

        assert_frame_equal(result, expected_result)

    def test_impute_multiple_columns_constant(self):
        """
        Test imputing multiple columns with strategy 'constant', checking that NaNs are replaced with the provided constant fill value.
        """
        df = pd.DataFrame({
            "x": [np.nan, 2, 3],
            "y": [4, np.nan, np.nan],
            "z": [1, 1, 1]
        })

        result = impute_missing_data(data = df, strategy = "constant", fill_value = -1)

        expected_result = pd.DataFrame({
            "x": [-1, 2, 3],
            "y": [4, -1, -1],
            "z": [1, 1, 1]
        })

        assert_frame_equal(result, expected_result, check_dtype = False)

    def test_raises_error_when_column_not_found(self):
        """
        Test that a ValueError is raised when a specified column for imputation does not exist in the DataFrame.
        """
        df = pd.DataFrame({"a": [1.0, np.nan]})
        
        with pytest.raises(ValueError, match = "Columns not found"):
            impute_missing_data(data = df, strategy = "mean", columns = ["missing_column"])

    def test_raises_error_on_non_numeric_column(self):
        """
        Test that a TypeError is raised when attempting to impute a non-numeric column.
        """
        df = pd.DataFrame({"a": [1.0, np.nan], "b": ["x", "y"]})
        
        with pytest.raises(TypeError, match = "Only numeric columns can be imputed"):
            impute_missing_data(data = df, strategy = "mean", columns = ["b"])

    def test_raises_error_if_fill_value_not_provided(self):
        """
        Test that a ValueError is raised if strategy is 'constant' but no fill_value is provided.
        """
        df = pd.DataFrame({"a": [1.0, np.nan]})
        
        with pytest.raises(ValueError, match = "You must provide `fill_value`"):
            impute_missing_data(data = df, strategy = "constant")