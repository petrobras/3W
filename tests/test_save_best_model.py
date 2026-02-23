import tempfile
import pickle
import pytest
import torch
import sklearn.linear_model

from io import BytesIO
from pathlib import Path

from ThreeWToolkit.utils import ModelRecorder
from ThreeWToolkit.models.mlp import MLP, MLPConfig


def make_mlp() -> MLP:
    config = MLPConfig(
        input_size=10,
        hidden_sizes=(8,),
        output_size=1,
        random_seed=42,
    )
    return MLP(config)


class TestModelRecorder:
    def setup_method(self):
        """
        Setup simple models for each supported framework.
        """
        self.torch_model = make_mlp()
        self.sklearn_model = sklearn.linear_model.LogisticRegression()

    def test_save_torch_model(self):
        """
        Test saving a PyTorch model (.pt).
        """
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            ModelRecorder.save_best_model(self.torch_model, tmp.name)
            assert Path(tmp.name).exists()
            assert Path(tmp.name).stat().st_size > 0
        Path(tmp.name).unlink()

    def test_save_sklearn_model(self):
        """
        Test saving a scikit-learn model (.pkl).
        """
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            ModelRecorder.save_best_model(self.sklearn_model, tmp.name)
            assert Path(tmp.name).exists()
            with open(tmp.name, "rb") as f:
                loaded = pickle.load(f)
                assert isinstance(loaded, sklearn.linear_model.LogisticRegression)
        Path(tmp.name).unlink()

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
            match=f"Invalid filename: `{filename}`. Expected a string or path-like object.",
        ):
            ModelRecorder.save_best_model(self.sklearn_model, filename)

    def test_file_like_object_not_supported(self):
        """
        Test passing a file-like object raises TypeError.
        """
        fake_file = BytesIO()
        with pytest.raises(
            TypeError,
            match=f"Invalid filename: `{fake_file}`. Expected a string or path-like object.",
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
        Path(tmp.name).unlink()

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
        Path(tmp.name).unlink()
