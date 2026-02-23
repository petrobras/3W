import torch

from typing import Protocol, Iterable, runtime_checkable


@runtime_checkable
class SupportsOptimizerParams(Protocol):
    """
    Structural contract for models that expose parameters
    compatible with PyTorch optimizers via `get_params()`.
    """

    def get_params(self) -> Iterable[torch.nn.Parameter]: ...
