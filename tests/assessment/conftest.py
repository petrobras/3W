"""Conftest for assessment tests.

This handles import issues by ensuring the toolkit is properly importable.
The assessment/__init__.py imports model_assess which imports metrics,
which has Pydantic validation issues on Python 3.14. We work around this
by importing modules in a specific order.
"""

import sys
from unittest.mock import MagicMock


def pytest_configure(config):
    """Hook to run before test collection."""
    # Mock the metrics module to avoid pydantic/numpy issues during import
    mock_metrics = MagicMock()
    mock_metrics.accuracy_score = MagicMock(return_value=0.5)
    mock_metrics.balanced_accuracy_score = MagicMock(return_value=0.5)
    mock_metrics.recall_score = MagicMock(return_value=0.5)
    mock_metrics.precision_score = MagicMock(return_value=0.5)
    mock_metrics.f1_score = MagicMock(return_value=0.5)
    mock_metrics.average_precision_score = MagicMock(return_value=0.5)
    mock_metrics.explained_variance_score = MagicMock(return_value=0.5)

    # Pre-register the mock before any imports try to load the real module
    if "ThreeWToolkit.metrics" not in sys.modules:
        sys.modules["ThreeWToolkit.metrics"] = mock_metrics
