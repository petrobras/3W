import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from ThreeWToolkit.utils.trainer_logger import TrainerLogger


@pytest.fixture
def sample_log():
    """Provide a sample optimization log for testing."""
    return {
        "model": "SVM",
        "params": {"C": 1.0, "kernel": "rbf"},
        "score": 0.87,
        "timestamp": datetime.now().isoformat(),
    }


@pytest.fixture
def temp_log_dir():
    """Create and clean up a temporary directory for log files."""
    log_dir = "temp_test_logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    yield log_dir
    shutil.rmtree(log_dir)


def test_log_json_format(sample_log, temp_log_dir):
    """Test saving the optimization log in JSON format."""
    path = TrainerLogger.log_optimization_progress(
        sample_log, log_dir=temp_log_dir, file_format="json"
    )
    assert Path(path).exists()
    assert Path(path).suffix == ".json"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["model"] == "SVM"


def test_log_pickle_format(sample_log, temp_log_dir):
    """Test saving the optimization log in Pickle format."""
    path = TrainerLogger.log_optimization_progress(
        sample_log, log_dir=temp_log_dir, file_format="pickle"
    )
    assert Path(path).exists()
    assert Path(path).suffix == ".pkl"

    with open(path, "rb") as f:
        data = pickle.load(f)

    assert data["params"]["C"] == 1.0


def test_invalid_format_raises_error(sample_log):
    """Test that an unsupported file format raises a ValueError."""
    with pytest.raises(ValueError):
        TrainerLogger.log_optimization_progress(sample_log, file_format="xml")


def test_non_dict_input_raises_type_error(tmp_path):
    """Test that passing a non-dictionary input raises a TypeError."""
    invalid_input = ["not", "a", "dict"]
    with pytest.raises(TypeError, match="progress_log must be a dictionary"):
        TrainerLogger.log_optimization_progress(
            progress_log=invalid_input, log_dir=str(tmp_path), file_format="json"
        )


def test_empty_dict_raises_value_error(tmp_path):
    """Test that passing an empty dictionary raises a ValueError."""
    empty_input = {}
    with pytest.raises(ValueError, match="progress_log must be a non-empty dictionary"):
        TrainerLogger.log_optimization_progress(
            progress_log=empty_input, log_dir=str(tmp_path), file_format="json"
        )
