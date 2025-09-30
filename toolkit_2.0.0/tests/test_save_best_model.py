import os
import tempfile
import pickle
from io import BytesIO

import pytest
import torch
import torch.nn as nn
import sklearn.linear_model

from ThreeWToolkit.utils import ModelRecorder


class SimpleTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


class TestModelRecorder:
    def setup_method(self):
        """
        Setup simple models for each supported framework.
        """
        self.torch_model = SimpleTorchModel()
        self.sklearn_model = sklearn.linear_model.LogisticRegression()

    def test_save_torch_model(self):
        """
        Test saving a PyTorch model (.pt).
        """
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            ModelRecorder.save_best_model(self.torch_model, tmp.name)
            assert os.path.exists(tmp.name)
            assert os.path.getsize(tmp.name) > 0
        os.remove(tmp.name)

    def test_save_sklearn_model(self):
        """
        Test saving a scikit-learn model (.pkl).
        """
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            ModelRecorder.save_best_model(self.sklearn_model, tmp.name)
            assert os.path.exists(tmp.name)
            with open(tmp.name, "rb") as f:
                loaded = pickle.load(f)
                assert isinstance(loaded, sklearn.linear_model.LogisticRegression)
        os.remove(tmp.name)

    def test_save_invalid_extension(self):
        """
        Test saving with an unsupported extension raises ValueError.
        """
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=True) as tmp:
            with pytest.raises(ValueError, match="Unsupported file extension"):
                ModelRecorder.save_best_model(self.sklearn_model, tmp.name)

    def test_invalid_type_as_filename(self):
        """
        Test passing a non-string and non-pathlike object raises TypeError.
        """
        filename = 12345
        with pytest.raises(
            TypeError,
            match=f"Invalid filename: `{filename}`. Expected a string, path-like object, or file-like object.",
        ):
            ModelRecorder.save_best_model(self.sklearn_model, filename)

    def test_file_like_object_not_supported(self):
        """
        Test passing a file-like object raises ValueError.
        """
        fake_file = BytesIO()
        with pytest.raises(
            ValueError,
            match=f"Saving to file-like object '{fake_file}' is not supported. Please provide a valid file path.",
        ):
            ModelRecorder.save_best_model(self.sklearn_model, fake_file)

    def test_exception_inside_torch_block(self, monkeypatch):
        """
        Simulate error inside torch.save to test exception handling.
        """

        def raise_error(*args, **kwargs):
            raise Exception("Simulated torch error")

        monkeypatch.setattr(torch, "save", raise_error)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            with pytest.raises(RuntimeError, match="Error saving PyTorch model"):
                ModelRecorder.save_best_model(self.torch_model, tmp.name)
        os.remove(tmp.name)

    def test_exception_inside_pickle_block(self, monkeypatch):
        """
        Simulate error inside pickle.dump to test exception handling.
        """

        def raise_error(*args, **kwargs):
            raise Exception("Simulated pickle error")

        monkeypatch.setattr(pickle, "dump", raise_error)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            with pytest.raises(RuntimeError, match="Error saving Pickle model"):
                ModelRecorder.save_best_model(self.sklearn_model, tmp.name)
        os.remove(tmp.name)
