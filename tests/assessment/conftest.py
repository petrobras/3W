"""Conftest for assessment tests."""

import sys
import pytest
from unittest.mock import MagicMock


@pytest.fixture(scope="module", autouse=True)
def mock_metrics_module():
    """Mock the metrics module for assessment tests only."""
    # Store the original module if it exists
    original_module = sys.modules.get("ThreeWToolkit.metrics")

    # Create mock
    mock_metrics = MagicMock()
    mock_metrics.accuracy_score = MagicMock(return_value=0.5)
    mock_metrics.balanced_accuracy_score = MagicMock(return_value=0.5)
    mock_metrics.recall_score = MagicMock(return_value=0.5)
    mock_metrics.precision_score = MagicMock(return_value=0.5)
    mock_metrics.f1_score = MagicMock(return_value=0.5)
    mock_metrics.average_precision_score = MagicMock(return_value=0.5)
    mock_metrics.explained_variance_score = MagicMock(return_value=0.5)

    # Install mock
    sys.modules["ThreeWToolkit.metrics"] = mock_metrics

    yield mock_metrics

    # Restore original module after tests
    if original_module is not None:
        sys.modules["ThreeWToolkit.metrics"] = original_module
    else:
        sys.modules.pop("ThreeWToolkit.metrics", None)
