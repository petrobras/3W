import os
import tempfile
import pickle
from io import BytesIO
from collections import OrderedDict

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


class TestModelRecorderLoad:
    def setup_method(self):
        """
        Setup simple models for each supported framework.
        """
        self.torch_model = SimpleTorchModel()
        self.sklearn_model = sklearn.linear_model.LogisticRegression()

    def _save_torch_state_dict_to_tmp(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        tmp.close()
        torch.save(self.torch_model.state_dict(), tmp.name)
        return tmp.name

    def _save_sklearn_pickle_to_tmp(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        tmp.close()
        with open(tmp.name, "wb") as f:
            pickle.dump(self.sklearn_model, f)
        return tmp.name

    def test_load_torch_model_into_instance(self):
        """
        Save a PyTorch state_dict and load it into a fresh model instance.
        """
        path = self._save_torch_state_dict_to_tmp()
        try:
            fresh_model = SimpleTorchModel()
            loaded = ModelRecorder.load_model(path, model=fresh_model)
            assert isinstance(loaded, SimpleTorchModel)

            # Compare parameters elementwise
            for p_loaded, p_saved in zip(
                loaded.state_dict().values(), self.torch_model.state_dict().values()
            ):
                assert torch.allclose(p_loaded, p_saved)
        finally:
            os.remove(path)

    def test_load_torch_returns_state_dict_when_model_none(self):
        """
        If no model is provided, load_model should return the state_dict.
        """
        path = self._save_torch_state_dict_to_tmp()
        try:
            state = ModelRecorder.load_model(path)
            assert isinstance(state, (dict, OrderedDict))
            # Must contain the same keys as original state_dict
            orig_keys = set(self.torch_model.state_dict().keys())
            assert set(state.keys()) == orig_keys
        finally:
            os.remove(path)

    def test_exception_inside_torch_load_block(self, monkeypatch):
        """
        Simulate error inside torch.load to test exception handling.
        """

        def raise_error(*args, **kwargs):
            raise Exception("Simulated torch load error")

        import torch as torch_module

        monkeypatch.setattr(torch_module, "load", raise_error)

        path = self._save_torch_state_dict_to_tmp()
        try:
            with pytest.raises(RuntimeError, match="Error loading PyTorch model"):
                ModelRecorder.load_model(path, model=SimpleTorchModel())
        finally:
            os.remove(path)

    def test_load_sklearn_pickled_model(self):
        """
        Save a sklearn model via pickle and load it back.
        """
        path = self._save_sklearn_pickle_to_tmp()
        try:
            loaded = ModelRecorder.load_model(path)
            assert isinstance(loaded, sklearn.linear_model.LogisticRegression)
        finally:
            os.remove(path)

    def test_exception_inside_pickle_load_block(self, monkeypatch):
        """
        Simulate error inside pickle.load to test exception handling.
        """

        def raise_error(*args, **kwargs):
            raise Exception("Simulated pickle load error")

        monkeypatch.setattr(pickle, "load", raise_error)

        path = self._save_sklearn_pickle_to_tmp()
        try:
            with pytest.raises(RuntimeError, match="Error loading Pickle model"):
                ModelRecorder.load_model(path)
        finally:
            os.remove(path)

    def test_load_invalid_extension(self):
        """
        Loading with an unsupported extension raises ValueError.
        """
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=True) as tmp:
            with pytest.raises(ValueError, match="Unsupported file extension"):
                ModelRecorder.load_model(tmp.name)

    def test_load_invalid_type_as_filename(self):
        """
        Passing a non-string and non-pathlike object raises TypeError.
        """
        filename = 98765
        with pytest.raises(
            TypeError,
            match=f"Invalid filename: `{filename}`. Expected a string, path-like object, or file-like object.",
        ):
            ModelRecorder.load_model(filename)

    def test_load_file_like_object_not_supported(self):
        """
        Passing a file-like object raises ValueError.
        """
        fake_file = BytesIO()
        with pytest.raises(
            ValueError,
            match=f"Loading from file-like object '{fake_file}' is not supported. Please provide a valid file path.",
        ):
            ModelRecorder.load_model(fake_file)

    def test_load_pickle_from_non_pickle_file_raises(self):
        """
        Attempt to load a .pkl that isn't a pickle object should raise RuntimeError.
        """
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp.write(b"not a real pickle payload")
            tmp.flush()
            tmp_path = tmp.name
        try:
            with pytest.raises(RuntimeError, match="Error loading Pickle model"):
                ModelRecorder.load_model(tmp_path)
        finally:
            os.remove(tmp_path)

    def test_load_torch_from_non_torch_file_raises(self):
        """
        Attempt to load a .pt that isn't a real torch state_dict should raise RuntimeError.
        """
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp.write(b"not a real torch payload")
            tmp.flush()
            tmp_path = tmp.name
        try:
            with pytest.raises(RuntimeError, match="Error loading PyTorch model"):
                ModelRecorder.load_model(tmp_path, model=SimpleTorchModel())
        finally:
            os.remove(tmp_path)
