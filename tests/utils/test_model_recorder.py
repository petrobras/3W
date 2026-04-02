import shutil
from pathlib import Path

from typing import cast
import pytest
import torch
from torch import nn

from ThreeWToolkit.utils.model_recorder import ModelRecorder
from ThreeWToolkit.models.sklearn_models import SklearnModels
from ThreeWToolkit.models.torch_models import TorchModels
from ThreeWToolkit.constants import CHECKPOINT_DIR


class SimpleTorchModel(TorchModels):
    """Simple PyTorch model for testing."""

    def __init__(self, input_size: int = 10, output_size: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def get_params(self):
        return self.parameters()


class SimpleSkLearnModel(SklearnModels):
    """Simple scikit-learn-like model for testing."""

    def __init__(self, param1: int = 5, param2: str = "default"):
        self.param1 = param1
        self.param2 = param2
        self.is_fitted = False

    def get_params(self):
        return {"param1": self.param1, "param2": self.param2}

    def fit(self, X, y):
        self.is_fitted = True


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create a temporary checkpoint directory for testing."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    yield checkpoint_dir
    # Cleanup
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)


@pytest.fixture
def pytorch_model():
    """Create a simple PyTorch model for testing."""
    return SimpleTorchModel(input_size=10, output_size=2)


@pytest.fixture
def sklearn_model():
    """Create a simple scikit-learn model for testing."""
    model = SimpleSkLearnModel(param1=42, param2="test")
    model.fit([[1, 2], [3, 4]], [0, 1]) # type: ignore
    return model


class TestModelRecorderSave:
    def test_save_pytorch_model(self, pytorch_model):
        """Test saving a PyTorch model."""
        filename = "test_torch_model.pt"
        path = ModelRecorder.save_model(pytorch_model, filename)

        assert path.exists()
        assert path.name == filename
        assert path.parent == CHECKPOINT_DIR
        assert path.suffix == ".pt"

        # Cleanup
        if path.exists():
            path.unlink()

    def test_save_sklearn_model(self, sklearn_model):
        """Test saving a scikit-learn model."""
        filename = "test_sklearn_model.pkl"
        path = ModelRecorder.save_model(sklearn_model, filename)

        assert path.exists()
        assert path.name == filename
        assert path.parent == CHECKPOINT_DIR
        assert path.suffix == ".pkl"

        # Cleanup
        if path.exists():
            path.unlink()

    def test_save_creates_checkpoint_dir(self, pytorch_model):
        """Test that save_model creates CHECKPOINT_DIR if it doesn't exist."""
        # Remove checkpoint dir if exists
        if CHECKPOINT_DIR.exists():
            shutil.rmtree(CHECKPOINT_DIR)

        filename = "test_model.pt"
        path = ModelRecorder.save_model(pytorch_model, filename)

        assert CHECKPOINT_DIR.exists()
        assert path.exists()

        # Cleanup
        if path.exists():
            path.unlink()

    def test_save_with_path_object(self, pytorch_model):
        """Test saving with Path object instead of string."""
        filename = Path("test_model_path.pt")
        path = ModelRecorder.save_model(pytorch_model, filename)

        assert path.exists()
        assert path.name == "test_model_path.pt"

        # Cleanup
        if path.exists():
            path.unlink()

    def test_save_unsupported_model_type(self):
        """Test that saving an unsupported model type raises ValueError."""

        class UnsupportedModel:
            pass

        model = UnsupportedModel()

        with pytest.raises(Exception):
            ModelRecorder.save_model(model, "test.pkl") # type: ignore


class TestModelRecorderLoad:
    def test_load_pytorch_model(self, pytorch_model):
        """Test loading a PyTorch model."""
        filename = "test_torch_load.pt"
        saved_path = ModelRecorder.save_model(pytorch_model, filename)

        # Create a new model instance
        loaded_model = ModelRecorder.load_model(saved_path)

        assert isinstance(loaded_model, SimpleTorchModel), f"Loaded model is not of type SimpleTorchModel: {type(loaded_model)}"

        # Check that state dict was loaded
        original_state = pytorch_model.state_dict()
        loaded_state = loaded_model.state_dict()
        assert set(original_state.keys()) == set(loaded_state.keys()), "State dict keys do not match"

        # Cleanup
        if saved_path.exists():
            saved_path.unlink()

    def test_load_sklearn_model(self, sklearn_model):
        """Test loading a scikit-learn model."""
        filename = "test_sklearn_load.pkl"
        saved_path = ModelRecorder.save_model(sklearn_model, filename)

        loaded_model = ModelRecorder.load_model(saved_path)

        assert isinstance(loaded_model, SimpleSkLearnModel)
        assert loaded_model.param1 == 42
        assert loaded_model.param2 == "test"
        assert loaded_model.is_fitted is True

        # Cleanup
        if saved_path.exists():
            saved_path.unlink()

    def test_load_from_filename_only(self, pytorch_model):
        """Test loading with just filename (looks in CHECKPOINT_DIR)."""
        filename = "test_filename_only.pt"
        saved_path = ModelRecorder.save_model(pytorch_model, filename)

        # Load using just the filename
        loaded_model = ModelRecorder.load_model(filename)

        assert isinstance(loaded_model, SimpleTorchModel)

        # Cleanup
        if saved_path.exists():
            saved_path.unlink()

    def test_load_pytorch_without_model_instance(self, pytorch_model):
        """Test loading PyTorch model without providing model instance."""
        filename = "test_no_instance.pt"
        saved_path = ModelRecorder.save_model(pytorch_model, filename)

        # Load without providing model instance (returns state dict or full model)
        loaded = ModelRecorder.load_model(saved_path)

        # Should return something (state dict or model)
        assert loaded is not None

        # Cleanup
        if saved_path.exists():
            saved_path.unlink()

    def test_load_with_pth_extension(self, pytorch_model):
        """Test loading PyTorch model with .pth extension."""
        filename = "test_model.pth"
        saved_path = ModelRecorder.save_model(pytorch_model, filename)

        loaded_model = ModelRecorder.load_model(saved_path)

        assert isinstance(loaded_model, SimpleTorchModel)

        # Cleanup
        if saved_path.exists():
            saved_path.unlink()

    def test_load_with_pickle_extension(self, sklearn_model):
        """Test loading scikit-learn model with .pickle extension."""
        filename = "test_model.pickle"
        saved_path = ModelRecorder.save_model(sklearn_model, filename)

        loaded_model = ModelRecorder.load_model(saved_path)

        assert isinstance(loaded_model, SimpleSkLearnModel)

        # Cleanup
        if saved_path.exists():
            saved_path.unlink()

    def test_load_unsupported_extension(self):
        """Test that loading a file with unsupported extension raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported file extension"):
            ModelRecorder.load_model("model.txt")

    def test_load_nonexistent_file(self):
        """Test that loading a nonexistent file raises RuntimeError."""
        with pytest.raises(Exception):
            ModelRecorder.load_model("nonexistent_model.pt")


class TestModelRecorderIntegration:
    def test_save_and_load_pytorch_roundtrip(self, pytorch_model):
        """Test full save/load cycle for PyTorch model."""
        filename = "roundtrip_torch.pt"

        # Generate some test data
        test_input = torch.randn(5, 10)
        original_output = pytorch_model(test_input)

        # Save the model
        saved_path = ModelRecorder.save_model(pytorch_model, filename)

        # Load the model
        loaded_model = cast(SimpleTorchModel, ModelRecorder.load_model(saved_path))

        # Test that loaded model produces same output
        loaded_output = loaded_model.forward(test_input)
        assert torch.allclose(original_output, loaded_output)

        # Cleanup
        if saved_path.exists():
            saved_path.unlink()

    def test_save_and_load_sklearn_roundtrip(self, sklearn_model):
        """Test full save/load cycle for scikit-learn model."""
        filename = "roundtrip_sklearn.pkl"

        # Save the model
        saved_path = ModelRecorder.save_model(sklearn_model, filename)

        # Load the model
        loaded_model = cast(SimpleSkLearnModel, ModelRecorder.load_model(saved_path))

        # Test that loaded model has same attributes
        assert loaded_model.param1 == sklearn_model.param1
        assert loaded_model.param2 == sklearn_model.param2
        assert loaded_model.is_fitted == sklearn_model.is_fitted

        # Cleanup
        if saved_path.exists():
            saved_path.unlink()
