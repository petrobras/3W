import logging
from typing import Sequence

from pydantic import Field, PrivateAttr, field_validator, ConfigDict
import torch
import torch.nn as nn

from ..core.base_models import ParamsT
from .torch_models import TorchModels, TorchModelsConfig

logger = logging.getLogger(__name__)


class MLPConfig(TorchModelsConfig):
    """MLP configuration. Use with TorchTrainer for training."""

    model_type: type["MLP"] = Field(
        default_factory=lambda: MLP, description="Type of model to use."
    )
    hidden_sizes: Sequence[int] = Field(
        ..., min_length=1, description="Tuple of hidden layer sizes."
    )
    activation_function: nn.Module = Field(
        default=nn.ReLU(),
        description="PyTorch activation function module (e.g., ReLU, Tanh, Sigmoid) applied to hidden layers.",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _target: type = PrivateAttr(default_factory=lambda: MLP)

    @field_validator("hidden_sizes")
    @classmethod
    def check_hidden_sizes(cls, hidden_sizes: Sequence[int]) -> list[int]:
        hidden_sizes = list(hidden_sizes)  # convert to list for easier validation
        if any(h <= 0 for h in hidden_sizes):
            raise ValueError("All hidden layer sizes must be > 0")
        return hidden_sizes


class MLP(TorchModels):
    """Multi-Layer Perceptron. Use TorchTrainer for training."""

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.config: MLPConfig = config
        self.activation_func = self.config.activation_function
        self.model: nn.Sequential | None = None

        self._build_layers()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        return self.model(x)

    def _build_layers(self) -> None:
        layers: list[nn.Module] = []
        if self.config.input_size is None:
            raise ValueError("Input size must be specified to build layers")

        in_size = self.config.input_size
        for h in self.config.hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(self.activation_func)
            in_size = h

        layers.append(nn.Linear(in_size, self.config.output_size))
        self.model = nn.Sequential(*layers)

    def get_params(self) -> ParamsT:
        if self.model is None:
            if not hasattr(self, "_dummy_param"):
                self._dummy_param = nn.Parameter(torch.tensor(0.0, requires_grad=True))
            return [self._dummy_param]
        return self.model.parameters()
