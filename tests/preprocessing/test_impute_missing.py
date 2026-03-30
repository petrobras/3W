"""Tests for ImputeMissing preprocessing class."""

import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal, assert_frame_equal

from ThreeWToolkit.preprocessing import ImputeMissing, ImputeMissingConfig


# Module-level fixtures

@pytest.fixture
def simple_df_with_nan():
    """DataFrame with NaN values in one column."""
    return pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})


@pytest.fixture
def multi_col_df_with_nan():
    """DataFrame with NaN values in multiple columns."""
    return pd.DataFrame(
        {
            "a": [np.nan, 2.0, np.nan],
            "b": [5.0, np.nan, 7.0],
            "c": [9.0, 10.0, 11.0],
        }
    )


@pytest.fixture
def simple_series_with_nan():
    """Series with NaN values."""
    return pd.Series([1.0, np.nan, 3.0])


@pytest.fixture
def df_with_non_numeric():
    """DataFrame with non-numeric column."""
    return pd.DataFrame({"a": [1.0, np.nan], "b": ["x", "y"]})


# Tests for different imputation strategies


class TestImputeMissingStrategies:
    """Test different imputation strategies."""

    def test_impute_mean_dataframe(self, simple_df_with_nan):
        """Test that imputing with strategy 'mean' replaces NaNs with column mean."""
        imp_missing = ImputeMissingConfig(strategy="mean").build()
        result = imp_missing(data=simple_df_with_nan)
        expected_result = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})

        assert_frame_equal(result, expected_result)

    def test_impute_median_dataframe_specific_column(self):
        """Test imputing with strategy 'median' on specified columns only."""
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 5.0, 6.0]})
        imp_missing = ImputeMissing(
            ImputeMissingConfig(strategy="median", columns=["a"])
        )
        result = imp_missing(data=df)
        expected_result = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [np.nan, 5.0, 6.0]})

        assert_frame_equal(result, expected_result)

    def test_impute_constant_series(self, simple_series_with_nan):
        """Test imputing a Series with strategy 'constant' and fill value."""
        imp_missing = ImputeMissing(
            ImputeMissingConfig(strategy="constant", fill_value=99.0)
        )
        result = imp_missing(data=simple_series_with_nan)
        expected_result = pd.Series([1.0, 99.0, 3.0], name="__temp__")

        assert_series_equal(result, expected_result)

    @pytest.mark.parametrize(
        "strategy,fill_value,expected_a,expected_b",
        [
            ("mean", None, [2.0, 2.0, 2.0], [5.0, 6.0, 7.0]),
            ("constant", -1, [-1, 2.0, -1], [5.0, -1, 7.0]),
            ("constant", 0, [0, 2.0, 0], [5.0, 0, 7.0]),
        ],
    )
    def test_impute_multiple_columns_parametrized(
        self, multi_col_df_with_nan, strategy, fill_value, expected_a, expected_b
    ):
        """Test imputing multiple columns with different strategies."""
        config_kwargs = {"strategy": strategy}
        if fill_value is not None:
            config_kwargs["fill_value"] = fill_value

        imp_missing = ImputeMissing(ImputeMissingConfig(**config_kwargs))
        result = imp_missing(data=multi_col_df_with_nan)

        expected_result = pd.DataFrame(
            {"a": expected_a, "b": expected_b, "c": [9.0, 10.0, 11.0]}
        )

        assert_frame_equal(result, expected_result, check_dtype=False)


class TestImputeMissingEdgeCases:
    """Test edge cases and error handling."""

    def test_returns_input_when_column_not_found(self):
        """Test that DataFrame is unchanged when specified columns don't exist."""
        df = pd.DataFrame({"a": [1.0, np.nan]})

        imp_missing = ImputeMissing(
            ImputeMissingConfig(strategy="mean", columns=["missing_column"])
        )
        result = imp_missing(data=df.copy())

        pd.testing.assert_frame_equal(result, df)

    def test_raises_error_on_non_numeric_column(self, df_with_non_numeric):
        """Test that TypeError is raised when imputing non-numeric columns."""
        with pytest.raises(TypeError, match="Only numeric columns can be imputed"):
            imp_missing = ImputeMissing(
                ImputeMissingConfig(strategy="mean", columns=["b"])
            )
            _ = imp_missing(data=df_with_non_numeric)

    def test_raises_error_if_fill_value_not_provided(self):
        """Test that ValueError is raised if strategy='constant' without fill_value."""
        df = pd.DataFrame({"a": [1.0, np.nan]})

        with pytest.raises(
            ValueError, match="You must provide `fill_value` when strategy='constant'"
        ):
            imp_missing = ImputeMissing(ImputeMissingConfig(strategy="constant"))
            _ = imp_missing(data=df)

    @pytest.mark.parametrize(
        "data",
        [
            pd.DataFrame({"a": [1.0, 2.0, 3.0]}),  # No NaN
            pd.Series([1.0, 2.0, 3.0]),  # Series without NaN
            pd.DataFrame({"a": [], "b": []}),  # Empty DataFrame
        ],
    )
    def test_impute_on_data_without_nan(self, data):
        """Test imputation on data without NaN values returns unchanged data."""
        imp_missing = ImputeMissingConfig(strategy="mean").build()
        result = imp_missing(data=data.copy())

        if isinstance(data, pd.DataFrame):
            assert_frame_equal(result, data)
        else:
            assert_series_equal(result, data)
