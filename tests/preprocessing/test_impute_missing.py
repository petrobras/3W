"""Tests for ImputeMissing preprocessing class."""

import pytest
import numpy as np

from ThreeWToolkit.core.base_dataset import BaseDataset

from ThreeWToolkit.preprocessing import NormalizeConfig
from ThreeWToolkit.preprocessing import ImputeMissingConfig


@pytest.fixture
def simple_dataset(mock_dataset_factory) -> BaseDataset:
    """Simple dataset for normalization tests."""
    return mock_dataset_factory(num_sensors=10)


class TestImputeMissingStrategies:
    """Test different imputation strategies."""

    def test_impute_missing_dataset_mean(self, simple_dataset):
        """Test imputation with different strategies."""

        imputer = ImputeMissingConfig(strategy="mean").build()
        imputer.fit(simple_dataset)

        # use a normalizer to check that the imputed values are close to the mean of the non-missing values
        normalizer = NormalizeConfig(norm="l2").build()
        normalizer.fit(simple_dataset)

        for event in simple_dataset:
            imputed = imputer.transform(event).signal
            assert not imputed.isna().any().any(), (
                "Imputed dataset should not contain any NaN values."
            )

            original_na = event.signal.isna()

            # Check that imputed values are close to the mean of the non-missing values
            for col in event.signal.columns:
                assert np.isclose(
                    imputed.loc[original_na[col], col], normalizer.global_average[col]
                ).all()

    def test_impute_missing_dataset_constant(self, simple_dataset):
        """Test imputation with constant strategy."""

        imputer = ImputeMissingConfig(strategy="constant", fill_value=42).build()
        imputer.fit(simple_dataset)

        for event in simple_dataset:
            imputed = imputer.transform(event).signal
            assert not imputed.isna().any().any(), (
                "Imputed dataset should not contain any NaN values."
            )

            original_na = event.signal.isna()

            # Check that imputed values are equal to the fill value
            for col in event.signal.columns:
                assert np.isclose(imputed.loc[original_na[col], col], 42).all()

    def test_impute_missing_dataset_forward_fill(self, simple_dataset):
        """Test imputation with forward fill strategy."""

        imputer = ImputeMissingConfig(strategy="ffill").build()
        imputer.fit(simple_dataset)

        for event in simple_dataset:
            imputed = imputer.transform(event).signal
            assert not imputed.isna().any().any(), (
                "Imputed dataset should not contain any NaN values."
            )

            original_na = event.signal.isna()

            # Check that imputed values are equal to the previous non-missing value
            for col in event.signal.columns:
                for idx in np.where(original_na[col])[0]:
                    if idx > 0:
                        assert np.isclose(
                            imputed.loc[idx, col], imputed.loc[idx - 1, col]
                        ), (
                            f"Imputed value at index {idx} should be equal to the previous non-missing value."
                        )

    def test_impute_missing_dataset_backward_fill(self, simple_dataset):
        """Test imputation with backward fill strategy."""

        imputer = ImputeMissingConfig(strategy="bfill").build()
        imputer.fit(simple_dataset)

        for event in simple_dataset:
            imputed = imputer.transform(event).signal
            assert not imputed.isna().any().any(), (
                "Imputed dataset should not contain any NaN values."
            )

            original_na = event.signal.isna()

            # Check that imputed values are equal to the next non-missing value
            for col in event.signal.columns:
                for idx in np.where(original_na[col])[0]:
                    if idx < len(imputed) - 1:
                        assert np.isclose(
                            imputed.loc[idx, col], imputed.loc[idx + 1, col]
                        ), (
                            f"Imputed value at index {idx} should be equal to the next non-missing value."
                        )

    def test_impute_missing_dataset_interpolate(self, simple_dataset):
        """Test imputation with interpolation strategy."""

        imputer = ImputeMissingConfig(
            strategy="interpolate", interpolate_method="linear"
        ).build()
        imputer.fit(simple_dataset)

        for event in simple_dataset:
            imputed = imputer.transform(event).signal
            assert not imputed.isna().any().any(), (
                "Imputed dataset should not contain any NaN values."
            )

            original_na = event.signal.isna()

            # Check that imputed values are between the previous and next non-missing values
            for col in event.signal.columns:
                for idx in np.where(original_na[col])[0]:
                    if idx > 0 and idx < len(imputed) - 1:
                        prev_value = imputed.loc[idx - 1, col]
                        next_value = imputed.loc[idx + 1, col]
                        assert prev_value <= imputed.loc[idx, col] <= next_value, (
                            f"Imputed value at index {idx} should be between the previous and next non-missing values."
                        )
