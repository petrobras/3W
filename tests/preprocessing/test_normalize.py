"""Tests for Normalize preprocessing class."""

import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal, assert_frame_equal
from numpy.testing import assert_allclose

from ThreeWToolkit.preprocessing import Normalize, NormalizeConfig


# Module-level fixtures

@pytest.fixture
def simple_df():
    """Simple DataFrame for normalization tests."""
    return pd.DataFrame({"x": [3.0, 0.0], "y": [4.0, 0.0]})


@pytest.fixture
def simple_series():
    """Simple Series for normalization tests."""
    return pd.Series([3.0, 4.0], name="s")


@pytest.fixture
def df_with_non_numeric():
    """DataFrame with non-numeric column."""
    return pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})


# Tests for different normalization methods


class TestNormalizeStrategies:
    """Test different normalization strategies."""

    def test_normalize_dataframe_l2_axis1(self, simple_df):
        """Test L2 normalization across rows of a DataFrame."""
        norm = Normalize(NormalizeConfig(norm="l2", axis=1))
        result = norm(data=simple_df)
        expected_result = pd.DataFrame({"x": [0.6, 0.0], "y": [0.8, 0.0]})

        assert_frame_equal(result, expected_result)

    def test_normalize_dataframe_l1_axis0(self):
        """Test L1 normalization across columns of a DataFrame."""
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 6.0]})
        norm = Normalize(NormalizeConfig(norm="l1", axis=0))
        result = norm(data=df)
        expected_result = pd.DataFrame({"a": [1 / 3, 2 / 3], "b": [1 / 3, 2 / 3]})

        assert_frame_equal(result, expected_result)

    def test_normalize_series_l2(self, simple_series):
        """Test L2 normalization of a Series."""
        norm = Normalize(NormalizeConfig(norm="l2", axis=0))
        result = norm(data=simple_series)
        expected_result = pd.Series([0.6, 0.8], name="s")

        assert_series_equal(result, expected_result)

    def test_normalize_series_max(self):
        """Test max normalization of a Series."""
        s = pd.Series([2.0, 8.0])
        norm = Normalize(NormalizeConfig(norm="max", axis=0))
        result = norm(data=s)
        expected_result = pd.Series([0.25, 1.0])

        assert_series_equal(result, expected_result)

    @pytest.mark.parametrize(
        "norm_type,axis,input_data,expected",
        [
            ("l2", 1, {"x": [3.0, 0.0], "y": [4.0, 0.0]}, {"x": [0.6, 0.0], "y": [0.8, 0.0]}),
            ("l1", 0, {"a": [1.0, 2.0], "b": [3.0, 6.0]}, {"a": [1/3, 2/3], "b": [1/3, 2/3]}),
            ("max", 1, {"x": [2.0, 4.0], "y": [1.0, 2.0]}, {"x": [1.0, 1.0], "y": [0.5, 0.5]}),
        ],
    )
    def test_normalize_parametrized(self, norm_type, axis, input_data, expected):
        """Test different normalization strategies with parametrization."""
        df = pd.DataFrame(input_data)
        norm = Normalize(NormalizeConfig(norm=norm_type, axis=axis))
        result = norm(data=df)
        expected_result = pd.DataFrame(expected)

        assert_frame_equal(result, expected_result)

    def test_normalize_return_norm(self, simple_df):
        """Test return of normalization + norm values."""
        norm = Normalize(NormalizeConfig(norm="l2", axis=1, return_norm_values=True))
        result, norm_values = norm(data=simple_df)
        expected_result = pd.DataFrame({"x": [0.6, 0.0], "y": [0.8, 0.0]})
        expected_norm = np.array([[5.0], [0.0]])

        assert_frame_equal(result, expected_result)
        assert_allclose(norm_values, expected_norm)


class TestNormalizeEdgeCases:
    """Test edge cases and error handling."""

    def test_non_numeric_dataframe_raises_type_error(self, df_with_non_numeric):
        """Test error raised when DataFrame has non-numeric column."""
        with pytest.raises(TypeError, match="Non-numeric columns"):
            norm = Normalize(NormalizeConfig())
            _ = norm(data=df_with_non_numeric)

    def test_non_numeric_series_raises_type_error(self):
        """Test error raised when Series is non-numeric."""
        s = pd.Series(["a", "b"])

        with pytest.raises(TypeError, match="Series must be numeric"):
            norm = Normalize(NormalizeConfig())
            _ = norm(data=s)

    @pytest.mark.parametrize(
        "data",
        [
            pd.DataFrame({"a": [0.0, 0.0], "b": [0.0, 0.0]}),  # All zeros
            pd.DataFrame({"a": [1.0], "b": [1.0]}),  # Single row
            pd.Series([0.0, 0.0]),  # Series with zeros
        ],
    )
    def test_normalize_edge_cases(self, data):
        """Test normalization on edge case data."""
        norm = Normalize(NormalizeConfig(norm="l2", axis=0))
        result = norm(data=data.copy())
        
        # Result should handle edge cases gracefully
        assert isinstance(result, type(data))
