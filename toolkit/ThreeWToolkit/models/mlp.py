import torch.nn as nn
import torch

from pathlib import Path
from pydantic import Field, field_validator
from typing import Iterable, Type, TypeAlias

from ..core.base_models import ModelsConfig, BaseModels
from ..utils.model_recorder import ModelRecorder
from ..core.enums import ModelTypeEnum, ActivationFunctionEnum
from ..core.base_training_strategies import TrainingStrategy
from ..trainer.strategies.epoch_strategy import EpochTrainingStrategy
from ..assessment.strategies.torch_prediction_strategy import TorchPredictionStrategy
from ..core.base_prediction_strategies import PredictionStrategy

# Type alias for PyTorch model parameters
ParamsT: TypeAlias = (
    Iterable[torch.Tensor]
    | Iterable[dict[str, torch.Tensor | int | float | str | bool]]
    | Iterable[tuple[str, torch.Tensor]]
)


class MLPConfig(ModelsConfig):
    """Configuration class for Multi-Layer Perceptron (MLP) model.

    This class defines all the architectural parameters and hyperparameters
    needed to configure an MLP neural network. It extends the base ModelsConfig
    with MLP-specific parameters and validation rules.

    The input_size parameter can now be `None` to enable automatic inference
    from the first batch of input data during the forward pass.

    Args:
        model_type (ModelTypeEnum, optional): Type of model, automatically set to MLP.
        input_size (int | None): Number of input features. Must be greater than `0`
            if specified, or None for automatic inference from input data.
        hidden_sizes (tuple[int, ...]): Tuple specifying the size of each hidden layer.
            Must contain at least one positive integer.
        output_size (int): Number of output neurons. Must be greater than 0.
        activation_function (str, optional): Activation function identifier.
            Allowed values are defined in ActivationFunctionEnum: {"relu", "sigmoid", "tanh"}.
        regularization (float | None, optional): L2 regularization parameter.
            If specified, must be >= 0. Defaults to None (no regularization).

    Raises:
        ValueError: If activation_function is not supported, if hidden_sizes contains
            non-positive values, or if any size parameter is not positive.

    Example:
        MLP with explicit input size:
        >>> config = MLPConfig(
        ...     input_size=784,
        ...     hidden_sizes=(128, 64),
        ...     output_size=10,
        ...     activation_function="relu"
        ... )

        MLP with automatic input size inference:
        >>> config = MLPConfig(
        ...     input_size=None,  # Will be inferred from data
        ...     hidden_sizes=(128, 64),
        ...     output_size=10,
        ...     activation_function="relu"
        ... )

        MLP with regularization:
        >>> config = MLPConfig(
        ...     input_size=20,
        ...     hidden_sizes=(50, 25, 10),
        ...     output_size=1,
        ...     activation_function="sigmoid",
        ...     regularization=0.01
        ... )

    Attributes:
        target_ (type): Reference to the concrete model class associated with this configuration.

    Note:
        - When input_size is `None`, the model will infer it from the first forward pass;
        - The model automatically adds activation functions between hidden layers;
        - No activation is applied to the output layer (handled by loss function);
        - Input and output sizes must match your data dimensions;
        - After inference, the input_size field will be updated with the actual value.
    """

    model_type: ModelTypeEnum = Field(
        default=ModelTypeEnum.MLP,
        description="Type of model (automatically set to MLP).",
    )
    input_size: int | None = Field(
        default=None,
        description="Number of input features (must be > 0 if specified, or `None` for automatic inference).",
    )
    hidden_sizes: tuple[int, ...] = Field(
        ...,
        min_length=1,
        description="Sizes of hidden layers (at least one layer required).",
    )
    output_size: int = Field(
        ..., gt=0, description="Number of output features (must be > 0)."
    )
    activation_function: str = Field(
        default="relu", description="Activation function to use between hidden layers."
    )
    regularization: float | None = Field(
        default=None,
        ge=0,
        description="L2 regularization parameter (>=0 or None for no regularization).",
    )
    target_: type[BaseModels] = Field(default_factory=lambda: MLP)

    @field_validator("input_size")
    @classmethod
    def check_input_size(cls: type["MLPConfig"], input_size: int | None):
        """Validate that `input_size` is positive if specified.

        Args:
            cls (MLPConfig): The class reference.
            input_size (int | None): The input size to validate.

        Returns:
            int | None: The validated input size.

        Raises:
            ValueError: If `input_size` is specified but not positive.
        """
        if input_size is not None and input_size <= 0:
            raise ValueError("`input_size` must be > 0 when specified")
        return input_size

    @field_validator("activation_function")
    @classmethod
    def check_activation_function(cls: type["MLPConfig"], activation_function: str):
        """Validate that the activation function is supported.

        Args:
            cls (MLPConfig): The class reference.
            activation_function (str): The activation function name to validate.

        Returns:
            str: The validated activation function name.

        Raises:
            ValueError: If the activation function is not supported.
        """
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
    def check_hidden_sizes(cls: type["MLPConfig"], hidden_sizes: tuple):
        """Validate that hidden sizes are positive integers.

        Args:
            cls (MLPConfig): The class reference.
            hidden_sizes (tuple): The hidden layer sizes to validate.

        Returns:
            tuple: The validated hidden layer sizes.

        Raises:
            ValueError: If any hidden size is not a positive integer.
        """
        if not hidden_sizes or not all(
            isinstance(h, int) and h > 0 for h in hidden_sizes
        ):
            raise ValueError("hidden_sizes must be a tuple of positive integers")
        return hidden_sizes

    def is_input_size_dynamic(self) -> bool:
        """Check if input size will be inferred dynamically.

        Returns:
            bool: True if input_size is None (dynamic inference), False otherwise.
        """
        return self.input_size is None

    def set_inferred_input_size(self, input_size: int) -> None:
        """Set the input size after it has been inferred from data.

        This method is typically called by the model during the first forward pass
        when input_size was originally None.

        Args:
            input_size (int): The inferred input size from the data.

        Raises:
            ValueError: If input_size is not positive.
            RuntimeError: If assignment fails due to immutable configuration.
        """
        if input_size <= 0:
            raise ValueError("Inferred input_size must be > 0")

        # Use object.__setattr__ to bypass pydantic's frozen model restriction
        # if the model is frozen, or direct assignment if mutable
        try:
            self.input_size = input_size
        except (AttributeError, ValueError):
            # Handle frozen models
            object.__setattr__(self, "input_size", input_size)


class MLP(BaseModels, nn.Module):
    """Multi-Layer Perceptron (MLP) model implemented with PyTorch.

    This class defines a flexible fully connected neural network architecture
    with support for multiple hidden layers, configurable activation functions,
    and dynamic input size inference during the first forward pass.

    The MLP focuses exclusively on model architecture and forward computation.
    Training and prediction logic are fully decoupled and delegated to external
    strategy classes via the Strategy pattern:

    - Training is handled by a `TrainingStrategy` (e.g., `EpochTrainingStrategy`);
    - Inference is handled by a `PredictionStrategy` (e.g., `TorchPredictionStrategy`);
    - Model persistence (save/load) is delegated to `ModelRecorder`.

    This design improves modularity, testability, and extensibility, allowing
    different training or prediction backends without modifying the model code.

    Args:
        config (MLPConfig): Configuration object specifying the model architecture
            and hyperparameters. If `input_size` is None, the input dimensionality
            is inferred automatically from the first batch during the forward pass.

    Attributes:
        config (MLPConfig): Configuration used to build the model.
        model (nn.Sequential | None): Sequential container holding the network layers.
            Initialized lazily if input size is inferred dynamically.
        activation_func (nn.Module): Activation function applied between hidden layers.
        _layers_built (bool): Indicates whether the network layers have been initialized.

    Example:
        Basic usage with dynamic input size inference:
        >>> config = MLPConfig(
        ...     input_size=None,
        ...     hidden_sizes=(128, 64),
        ...     output_size=10
        ... )
        >>> model = MLP(config)
        >>> x = torch.randn(32, 784)
        >>> y = model(x)  # input_size is inferred as 784

        Training using a strategy:
        >>> train_strategy = model.get_training_strategy()()
        >>> history = train_strategy.train(...)

        Prediction using a strategy:
        >>> pred_strategy = model.get_prediction_strategy()()
        >>> y_pred = pred_strategy.predict(...)

        Saving and loading the model:
        >>> model.save("mlp_best.pth")
        >>> model.load("mlp_best.pth")
    """

    # Explicitly type the config attribute
    config: MLPConfig

    def __init__(self, config: MLPConfig) -> None:
        """Initialize the MLP model with the given configuration.

        If input_size is None in `config`, the model will build layers
        dynamically on the first forward pass.

        Args:
            config (MLPConfig): Configuration object specifying the model
                architecture and hyperparameters. `input_size` can be None.
        """
        # Initialize both parent classes
        super().__init__(config=config)

        # Explicitly assign config with correct type
        self.config = config

        self.activation_func = self._get_activation_function(config.activation_function)
        self._layers_built = False
        self.model: nn.Sequential | None = None

        # If input_size is provided, build layers immediately
        if config.input_size is not None:
            self._build_layers(config.input_size)
            self._layers_built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        If layers haven't been built yet, they will be built based on
        the input tensor's last dimension.

        Args:
            x (torch.Tensor): Input tensor where the last dimension represents the feature dimension (…, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # Ensure layers are built
        self._ensure_model_built(x)

        # Model is guaranteed to be non-None after _ensure_model_built
        assert self.model is not None, "Model should be built at this point"
        return self.model(x)

    def save(self, path: str | Path) -> None:
        """Save the MLP model weights to disk.

        This method delegates persistence logic to ModelRecorder and stores
        only the model's state_dict, following PyTorch best practices.

        Args:
            path (str | Path): Destination file path. Supported extensions:
                - .pt
                - .pth
        """
        ModelRecorder.save_best_model(self, path)

    def load(self, path: str | Path) -> "MLP":
        """Load model weights from disk.

        This method loads a saved state_dict into the current model instance.
        The model architecture must be compatible with the saved weights.

        Args:
            path (str | Path): Path to the saved model file (.pt or .pth).

        Returns:
            MLP: The current model instance with loaded weights.
        """
        ModelRecorder.load_model(path, model=self)
        return self

    def get_training_strategy(self) -> Type[TrainingStrategy]:
        """Return the training strategy class associated with this model.

        Returns:
            Type[TrainingStrategy]: Training strategy implementation.
        """
        return EpochTrainingStrategy

    def get_prediction_strategy(self) -> Type[PredictionStrategy]:
        """Return the prediction strategy class associated with this model.

        Returns:
            Type[PredictionStrategy]: Prediction strategy implementation.
        """
        return TorchPredictionStrategy

    def _build_layers(self, input_size: int) -> None:
        """Build the network layers with the given input size.

        Args:
            input_size (int): The size of the input features.
        """
        layers: list[nn.Module] = []

        # Create hidden layers with activation functions
        in_size = input_size
        for h in self.config.hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(self.activation_func)
            in_size = h

        # Add final output layer (no activation - handled by loss function)
        layers.append(nn.Linear(in_size, self.config.output_size))
        self.model = nn.Sequential(*layers)

        # Update config with the inferred input size using the safe method
        if self.config.is_input_size_dynamic():
            self.config.set_inferred_input_size(input_size)

    def _ensure_model_built(self, x: torch.Tensor) -> None:
        """Ensure the model layers are built, building them if necessary.

        Args:
            x (torch.Tensor): Input tensor to infer dimensions from if needed.
        """
        if not self._layers_built:
            input_size = x.shape[-1]

            self._build_layers(input_size)
            assert self.model is not None

            self.model = self.model.to(x.device)

            self._layers_built = True

    def _get_activation_function(self, activation: str) -> nn.Module:
        """Create and return the specified activation function module.

        Args:
            activation (str): Name of the activation function. Must be one of
                the supported activation functions from ActivationFunctionEnum.

        Returns:
            nn.Module: PyTorch activation function module.

        Raises:
            ValueError: If the activation function is not supported.

        Note:
            Supported activation functions:
            - "relu": Rectified Linear Unit (most common, good default)
            - "sigmoid": Sigmoid function (useful for binary outputs)
            - "tanh": Hyperbolic tangent (zero-centered output)
        """
        if activation == ActivationFunctionEnum.RELU.value:
            return nn.ReLU()
        elif activation == ActivationFunctionEnum.SIGMOID.value:
            return nn.Sigmoid()
        elif activation == ActivationFunctionEnum.TANH.value:
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def get_params(self) -> ParamsT:
        """Return model parameters for optimization.

        Provides access to all trainable parameters in the model for use
        with PyTorch optimizers. If the model hasn't been built yet,
        creates a dummy parameter to avoid empty parameter list errors.
        The real parameters will be added to the optimizer's param_groups
        once the model is built.

        Returns:
            ParamsT: Iterator over model parameters that can be passed
                directly to PyTorch optimizers.

        Example:
            >>> model = MLP(config)
            >>> optimizer = torch.optim.Adam(model.get_params(), lr=0.001)
        """
        if self.model is None:
            # Create a dummy parameter to avoid empty parameter list
            # This will be replaced with real parameters once the model is built
            if not hasattr(self, "_dummy_param"):
                self._dummy_param = nn.Parameter(torch.tensor(0.0, requires_grad=True))
            return [self._dummy_param]
        return self.model.parameters()
