import torch
import torch.nn as nn

from pathlib import Path
from pydantic import Field, field_validator, PrivateAttr
from ..core.base_models import ModelsConfig, BaseTorchModels, ParamsT
from ..utils.model_recorder import ModelRecorder
from ..core.enums import ModelTypeEnum, ActivationFunctionEnum


class MLPConfig(ModelsConfig):
    """MLP configuration. Use with TorchTrainer for training."""

    model_type: ModelTypeEnum = Field(default=ModelTypeEnum.MLP)
    input_size: int | None = Field(default=None)
    hidden_sizes: tuple[int, ...] = Field(..., min_length=1)
    output_size: int = Field(..., gt=0)
    activation_function: str = Field(default="relu")
    regularization: float | None = Field(default=None, ge=0)
    _target: type = PrivateAttr(default_factory=lambda: MLP)

    @field_validator("input_size")
    @classmethod
    def check_input_size(cls, input_size: int | None) -> int | None:
        if input_size is not None and input_size <= 0:
            raise ValueError("`input_size` must be > 0 when specified")
        return input_size

    @field_validator("activation_function")
    @classmethod
    def check_activation_function(cls, activation_function: str) -> str:
        valid = {
            ActivationFunctionEnum.RELU.value,
            ActivationFunctionEnum.SIGMOID.value,
            ActivationFunctionEnum.TANH.value,
        }
        if activation_function not in valid:
            raise ValueError(f"activation_function must be one of {valid}")
        return activation_function

    @field_validator("hidden_sizes")
    @classmethod
    def check_hidden_sizes(cls, hidden_sizes: tuple) -> tuple:
        if not hidden_sizes or not all(
            isinstance(h, int) and h > 0 for h in hidden_sizes
        ):
            raise ValueError("hidden_sizes must be a tuple of positive integers")
        return hidden_sizes

    def is_input_size_dynamic(self) -> bool:
        return self.input_size is None

    def set_inferred_input_size(self, input_size: int) -> None:
        if input_size <= 0:
            raise ValueError("Inferred input_size must be > 0")
        try:
            self.input_size = input_size
        except (AttributeError, ValueError):
            object.__setattr__(self, "input_size", input_size)


class MLP(BaseTorchModels):
    """Multi-Layer Perceptron. Use TorchTrainer for training."""

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.config: MLPConfig = config
        self.activation_func = self._get_activation_function(config.activation_function)
        self._layers_built = False
        self.model: nn.Sequential | None = None

        if config.input_size is not None:
            self._build_layers(config.input_size)
            self._layers_built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_model_built(x)
        assert self.model is not None
        return self.model(x)

    def save(self, filename: str | Path) -> None:
        """Save model to disk."""
        path = Path(filename)
        if path.exists() and path.suffix not in {".pt", ".pth"}:
            raise ValueError("PyTorch models must be saved with .pt or .pth extension")
        if not path.exists() and path.suffix == "":
            path = path.with_suffix(".pth")

        ModelRecorder.save_model(self, path)

    def load(self, filename: str | Path):
        """Load model from disk."""
        path = Path(filename)
        ModelRecorder.load_model(path, model=self)

    def _build_layers(self, input_size: int) -> None:
        layers: list[nn.Module] = []
        in_size = input_size
        for h in self.config.hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(self.activation_func)
            in_size = h

        layers.append(nn.Linear(in_size, self.config.output_size))
        self.model = nn.Sequential(*layers)

        if self.config.is_input_size_dynamic():
            self.config.set_inferred_input_size(input_size)

    def _ensure_model_built(self, x: torch.Tensor) -> None:
        if not self._layers_built:
            input_size = x.shape[-1]
            self._build_layers(input_size)
            assert self.model is not None
            self.model = self.model.to(x.device)
            self._layers_built = True

    def _get_activation_function(self, activation: str) -> nn.Module:
        if activation == ActivationFunctionEnum.RELU.value:
            return nn.ReLU()
        elif activation == ActivationFunctionEnum.SIGMOID.value:
            return nn.Sigmoid()
        elif activation == ActivationFunctionEnum.TANH.value:
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def get_params(self) -> ParamsT:
        if self.model is None:
            if not hasattr(self, "_dummy_param"):
                self._dummy_param = nn.Parameter(torch.tensor(0.0, requires_grad=True))
            return [self._dummy_param]
        return self.model.parameters()
