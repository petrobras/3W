import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
from ..core.base_models import ModelsConfig, BaseModels
from ..core.enums import ModelTypeEnum, ActivationFunctionEnum
from typing import Iterable, Any, TypeAlias, Callable
from pydantic import Field, field_validator

# Type alias for PyTorch model parameters
ParamsT: TypeAlias = (
    Iterable[torch.Tensor]
    | Iterable[dict[str, Any]]
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
        activation_function (str, optional): Activation function to use between layers.
            Options: "relu", "sigmoid", "tanh". Defaults to "relu".
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

    @field_validator("input_size")
    @classmethod
    def check_input_size(cls, v):
        """Validate that `input_size` is positive if specified.

        Args:
            v (int | None): The input size to validate.

        Returns:
            int | None: The validated input size.

        Raises:
            ValueError: If `input_size` is specified but not positive.
        """
        if v is not None and v <= 0:
            raise ValueError("`input_size` must be > 0 when specified")
        return v

    @field_validator("activation_function")
    @classmethod
    def check_activation_function(cls, v):
        """Validate that the activation function is supported.

        Args:
            v (str): The activation function name to validate.

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
        if v not in valid:
            raise ValueError(f"activation_function must be one of {valid}")
        return v

    @field_validator("hidden_sizes")
    @classmethod
    def check_hidden_sizes(cls, v):
        """Validate that hidden sizes are positive integers.

        Args:
            v (tuple): The hidden layer sizes to validate.

        Returns:
            tuple: The validated hidden layer sizes.

        Raises:
            ValueError: If any hidden size is not a positive integer.
        """
        if not v or not all(isinstance(h, int) and h > 0 for h in v):
            raise ValueError("hidden_sizes must be a tuple of positive integers")
        return v

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
    """Multi-Layer Perceptron (MLP) implementation using PyTorch.

    A flexible MLP neural network that supports multiple hidden layers,
    various activation functions, and automatic task type detection based
    on output dimensions. The model automatically infers input size from
    the first batch of data during forward pass.

    This implementation focuses on training efficiency and integrates with
    the broader model ecosystem through the BaseModels interface while
    providing PyTorch nn.Module functionality.

    Args:
        config (MLPConfig): Configuration object containing model architecture
            and hyperparameter specifications. input_size can be None for
            automatic inference.

    Attributes:
        model (nn.Sequential | None): The sequential neural network layers.
        activation_func (nn.Module): The activation function module used
            between hidden layers.
        config (MLPConfig): The configuration used to build this model.
        _layers_built (bool): Flag indicating if layers have been built.

    Example:
        Basic usage with automatic input size:
        >>> config = MLPConfig(
        ...     input_size=None,  # Will be inferred automatically
        ...     hidden_sizes=(128, 64),
        ...     output_size=10
        ... )
        >>> model = MLP(config)
        >>> # Input size will be inferred on first forward pass
        >>> output = model(torch.randn(32, 784))  # input_size becomes 784

        Training example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> criterion = nn.CrossEntropyLoss()
        >>> history = model.fit(
        ...     train_loader=train_loader,
        ...     epochs=100,
        ...     optimizer=optimizer,
        ...     criterion=criterion,
        ...     val_loader=val_loader
        ... )
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
            self._layers_built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        If layers haven't been built yet, they will be built based on
        the input tensor's last dimension.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # Ensure layers are built
        self._ensure_model_built(x)

        # Model is guaranteed to be non-None after _ensure_model_built
        assert self.model is not None, "Model should be built at this point"
        return self.model(x)

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

    def _update_optimizer_params(self, optimizer: torch.optim.Optimizer) -> None:
        """Update optimizer with actual model parameters after model is built.

        This method should be called after the model is built to replace
        any dummy parameters in the optimizer with the actual model parameters.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to update.
        """
        if self.model is not None and hasattr(self, "_dummy_param"):
            # Clear existing parameter groups
            optimizer.param_groups.clear()

            # Add real model parameters
            optimizer.add_param_group({"params": list(self.model.parameters())})

            # Remove dummy parameter reference
            delattr(self, "_dummy_param")

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: Callable,
        optimizer: torch.optim.Optimizer,
        device: str,
    ) -> float:
        """Execute one complete training epoch.

        Performs forward pass, loss calculation, backward propagation,
        and parameter updates for all batches in the training data.

        Args:
            model (nn.Module): The neural network model to train.
            train_loader (DataLoader): DataLoader containing training batches.
            criterion (Callable): Loss function for computing training loss.
            optimizer (torch.optim.Optimizer): Optimizer for parameter updates.
            device (str): Device ('cpu' or 'cuda') for tensor operations.

        Returns:
            float: Average training loss across all batches in the epoch.

        Note:
            - Automatically detects task type based on output dimensions
            - Handles multiclass classification, binary classification, and regression
            - Sets model to training mode and handles gradient computation
        """
        model.train()  # Set model to training mode
        epoch_train_loss = 0.0

        for x_values, y_values in train_loader:
            # Move data to specified device
            x_values, y_values = x_values.to(device), y_values.to(device)

            # Clear gradients from previous iteration
            optimizer.zero_grad()

            # Forward pass
            outputs = model(x_values)

            # Automatic task type detection and loss computation
            if outputs.shape[1] > 1:  # Multiclass classification
                y_values = y_values.long()  # Convert to integer labels
                loss = criterion(outputs, y_values)
            else:  # Binary classification or regression
                if isinstance(criterion, nn.BCEWithLogitsLoss):
                    # Binary classification with logits
                    y_values = y_values.float()
                    loss = criterion(outputs.squeeze(1), y_values)
                else:
                    # Regression or other loss functions
                    y_values = y_values.float()
                    loss = criterion(outputs.squeeze(1), y_values)

            # Backward pass and parameter update
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        return epoch_train_loss / len(train_loader)

    def fit(
        self,
        train_loader: DataLoader,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        val_loader: DataLoader | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> dict[str, list[Any]]:
        """Train the MLP model with the provided data and configuration.

        Executes the complete training process including forward/backward passes,
        parameter updates, and optional validation loss tracking. Provides
        a progress bar for training monitoring.

        Args:
            train_loader (DataLoader): DataLoader containing training data batches.
                Each batch should contain (features, targets) tuples.
            epochs (int): Number of complete passes through the training data.
                Must be positive.
            optimizer (torch.optim.Optimizer): PyTorch optimizer for parameter
                updates (e.g., Adam, SGD, AdamW).
            criterion (Callable): Loss function for training. Should be appropriate
                for the task (e.g., CrossEntropyLoss for classification, MSELoss
                for regression).
            val_loader (DataLoader | None, optional): DataLoader for validation
                data. If provided, validation loss will be calculated and tracked.
                Defaults to None.
            device (str, optional): Computing device for training ('cpu' or 'cuda').
                Defaults to 'cuda' if available, otherwise 'cpu'.

        Returns:
            dict[str, list[Any]]: Training history dictionary containing:
                - 'train_loss': List of average training losses per epoch
                - 'val_loss': List of validation losses per epoch (if val_loader provided)

        Example:
            Basic training:
            >>> model = MLP(config)
            >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            >>> criterion = nn.CrossEntropyLoss()
            >>> history = model.fit(
            ...     train_loader=train_loader,
            ...     epochs=100,
            ...     optimizer=optimizer,
            ...     criterion=criterion
            ... )
            >>> print(f"Final training loss: {history['train_loss'][-1]:.4f}")

            Training with validation:
            >>> history = model.fit(
            ...     train_loader=train_loader,
            ...     epochs=100,
            ...     optimizer=optimizer,
            ...     criterion=criterion,
            ...     val_loader=val_loader,
            ...     device='cuda'
            ... )
            >>> print(f"Final validation loss: {history['val_loss'][-1]:.4f}")

        Note:
            - Model is automatically moved to the specified device
            - Training progress is displayed with a progress bar
            - Validation loss is computed without gradient tracking for efficiency
            - Loss values are averaged per epoch for consistent comparison
        """
        # Ensure model is built by doing a forward pass on the first batch
        if not self._layers_built:
            first_batch = next(iter(train_loader))
            x_first, _ = first_batch
            self._ensure_model_built(x_first)

            # Update optimizer with real parameters after model is built
            self._update_optimizer_params(optimizer)

        # Model is guaranteed to be non-None at this point
        assert self.model is not None, "Model should be built before training"

        # Move model to specified device
        self.model.to(device)

        # Initialize loss tracking dictionary
        loss_dict: dict[str, list[Any]] = {"train_loss": []}

        # Add validation loss tracking if validation data provided
        if val_loader is not None:
            loss_dict["val_loss"] = []

        pbar = tqdm(
            range(epochs), desc="[Pipeline] Training", unit="epoch", colour="#00b4d8"
        )

        for epoch_idx in pbar:
            # Execute one training epoch
            avg_epoch_train_loss = self._train_epoch(
                model=self.model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )
            loss_dict["train_loss"].append(avg_epoch_train_loss)

            # Calculate validation loss if validation data provided
            if val_loader is not None:
                val_loss = self._calculate_val_loss(val_loader, criterion, device)
                loss_dict["val_loss"].append(val_loss)

                pbar.set_description_str(
                    f"[Pipeline] Training | train_loss: {avg_epoch_train_loss:.4f}, val_loss: {val_loss:.4f}"
                )
            else:
                pbar.set_description_str(
                    f"[Pipeline] Training | train_loss: {avg_epoch_train_loss:.4f}"
                )

        return loss_dict

    def _calculate_val_loss(
        self, val_loader: DataLoader, criterion: Callable, device: str
    ) -> float:
        """Calculate validation loss without computing metrics.

        Computes the average loss on validation data without gradient tracking
        for memory efficiency. Uses the same task detection logic as training.

        Args:
            val_loader (DataLoader): DataLoader containing validation data.
            criterion (Callable): Loss function used during training.
            device (str): Device for tensor operations.

        Returns:
            float: Average validation loss across all batches.

        Note:
            - Model is set to evaluation mode during validation
            - No gradients are computed for efficiency
            - Uses same task detection as training for consistency
        """
        if self.model is None:
            raise ValueError("Model should be built before validation")

        self.model.eval()  # Set model to evaluation mode
        running_loss = 0.0

        # Disable gradient computation for efficiency
        with torch.no_grad():
            for x_values, y_values in val_loader:
                # Move data to device
                x_values, y_values = x_values.to(device), y_values.to(device)

                # Forward pass only
                outputs = self.model(x_values)

                # Task detection and loss computation (same logic as training)
                if outputs.shape[1] > 1:  # Multiclass classification
                    y_values = y_values.long()
                    loss = criterion(outputs, y_values)
                else:  # Binary classification or regression
                    if isinstance(criterion, nn.BCEWithLogitsLoss):
                        y_values = y_values.float()
                        loss = criterion(outputs.squeeze(1), y_values)
                    else:
                        y_values = y_values.float()
                        loss = criterion(outputs.squeeze(1), y_values)

                running_loss += loss.item()

        return running_loss / len(val_loader)

    def predict(
        self,
        loader: DataLoader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> np.ndarray:
        """Generate predictions for the given data.

        Performs inference on the provided data and returns predictions
        in the appropriate format based on the detected task type.

        Args:
            loader (DataLoader): DataLoader containing data for prediction.
                Should have the same structure as training data.
            device (str, optional): Computing device for inference.
                Defaults to 'cuda' if available, otherwise 'cpu'.

        Returns:
            np.ndarray: Array of predictions with shape (n_samples,).
                - For multiclass classification: predicted class indices
                - For binary classification: predicted class labels (0 or 1)
                - For regression: predicted continuous values

        Example:
            >>> test_predictions = model.predict(test_loader, device='cuda')
            >>> print(f"Predicted classes: {test_predictions}")

            >>> # For probability predictions in binary classification
            >>> # You would need to modify the method or use forward() directly
            >>> # with sigmoid activation

        Note:
            - Model is automatically set to evaluation mode
            - No gradients are computed during prediction for efficiency
            - Task type is automatically detected from output dimensions
            - Predictions are converted to numpy arrays for compatibility
        """
        if self.model is None:
            raise ValueError("Model should be built before prediction")

        self.model.eval()  # Set model to evaluation mode
        y_pred: list[Any] = []

        # Disable gradient computation for efficiency
        with torch.no_grad():
            for X_batch, _ in loader:  # Ignore labels in prediction
                X_batch = X_batch.to(device).float()

                # Forward pass to get raw outputs
                outputs = self.model.forward(X_batch)

                # Task detection and prediction conversion
                if outputs.shape[1] > 1:  # Multiclass classification
                    # Get class with highest probability
                    _, preds = torch.max(outputs, 1)
                elif outputs.shape[1] == 1:  # Binary classification
                    # Apply sigmoid and threshold at 0.5
                    preds = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)
                else:  # Regression (shouldn't happen with current architecture)
                    preds = outputs.squeeze(1)

                # Convert to CPU and add to prediction list
                y_pred.extend(preds.cpu().numpy())

        return np.array(y_pred)
