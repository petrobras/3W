import os
import pytest
from pathlib import Path
import contextlib

from ThreeWToolkit.utils.latex_manager import latex_environment

def test_set_and_restore_texinputs_when_none_exists(monkeypatch, tmp_path):
    """
    Tests that TEXINPUTS is set correctly when it doesn't exist initially
    and is completely removed after the context exits.
    """
    # Ensure the environment variable is not set before the test
    monkeypatch.delenv("TEXINPUTS", raising=False)
    
    search_path = tmp_path / "custom_lib"
    search_path.mkdir()

    assert "TEXINPUTS" not in os.environ

    with latex_environment(search_path):
        # Inside the context, check if the variable is set correctly
        expected_path = f"{str(search_path)}{os.sep}{os.sep}{os.pathsep}"
        assert os.environ.get("TEXINPUTS") == expected_path

    # After the context, ensure the variable has been removed
    assert "TEXINPUTS" not in os.environ


def test_set_and_restore_texinputs_when_it_exists(monkeypatch, tmp_path):
    """
    Tests that the new path is prepended to an existing TEXINPUTS variable
    and that the original value is restored after the context exits.
    """
    original_path = f"/usr/local/texlive/texmf-local{os.pathsep}"
    monkeypatch.setenv("TEXINPUTS", original_path)

    search_path = tmp_path / "project_pkg"
    search_path.mkdir()
    
    assert os.environ.get("TEXINPUTS") == original_path

    with latex_environment(search_path):
        # Inside the context, check if the new path was prepended
        new_part = f"{str(search_path)}{os.sep}{os.sep}{os.pathsep}"
        expected_path = f"{new_part}{original_path}"
        assert os.environ.get("TEXINPUTS") == expected_path

    # After the context, ensure the original value is restored
    assert os.environ.get("TEXINPUTS") == original_path


def test_recursive_search_path_format(monkeypatch, tmp_path):
    """
    Verifies the special kpathsea path format is correctly constructed.
    It should end with a double separator and a path separator.
    """
    monkeypatch.delenv("TEXINPUTS", raising=False)
    search_path = tmp_path / "another_dir"
    search_path.mkdir()
    
    with latex_environment(search_path):
        texinputs = os.environ.get("TEXINPUTS")
        # Check for the recursive search marker '//' or '\\'
        assert texinputs.startswith(f"{str(search_path)}{os.sep}{os.sep}")
        # Check for the trailing path separator to include default paths
        assert texinputs.endswith(os.pathsep)


def test_context_manager_handles_exceptions(monkeypatch, tmp_path):
    """
    Ensures that the environment variable is restored to its original state
    even if an exception is raised within the 'with' block.
    """
    original_path = "some/initial/path:"
    monkeypatch.setenv("TEXINPUTS", original_path)

    search_path = tmp_path / "raises_error"
    
    with pytest.raises(ValueError, match="Test exception"):
        with latex_environment(search_path):
            # The environment should be updated here
            assert os.environ.get("TEXINPUTS") != original_path
            # Raise an exception to test the 'finally' block
            raise ValueError("Test exception")
            
    # After the exception is caught, the 'finally' block should have run.
    # Check that the environment variable is restored.
    assert os.environ.get("TEXINPUTS") == original_path

