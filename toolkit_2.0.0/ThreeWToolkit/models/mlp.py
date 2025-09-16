import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from ..core.base_models import ModelsConfig, BaseModels
from ..core.enums import ModelTypeEnum, ActivationFunctionEnum
from typing import Iterable, Any, TypeAlias, Union, Callable
from tqdm import tqdm
from pydantic import Field, field_validator

# Type alias for PyTorch model parameters
ParamsT: TypeAlias = Union[
    Iterable[torch.Tensor], Iterable[dict[str, Any]], Iterable[tuple[str, torch.Tensor]]
]


class MLPConfig(ModelsConfig):
    """Configuration class for Multi-Layer Perceptron (MLP) model.

    This class defines all the architectural parameters and hyperparameters
    needed to configure an MLP neural network. It extends the base ModelsConfig
    with MLP-specific parameters and validation rules.

    Args:
        model_type (ModelTypeEnum, optional): Type of model, automatically set to MLP.
        input_size (int): Number of input features. Must be greater than 0.
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
        Basic MLP configuration:
        >>> config = MLPConfig(
        ...     input_size=784,
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
        - The model automatically adds activation functions between hidden layers
        - No activation is applied to the output layer (handled by loss function)
        - Input and output sizes must match your data dimensions
    """

    model_type: ModelTypeEnum = Field(
        default=ModelTypeEnum.MLP,
        description="Type of model (automatically set to MLP).",
    )
    input_size: int = Field(
        ..., gt=0, description="Number of input features (must be > 0)."
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


class MLP(BaseModels, nn.Module):
    """Multi-Layer Perceptron (MLP) implementation using PyTorch.

    A flexible MLP neural network that supports multiple hidden layers,
    various activation functions, and automatic task type detection based
    on output dimensions. The model is designed for both classification
    and regression tasks.

    This implementation focuses on training efficiency and integrates with
    the broader model ecosystem through the BaseModels interface while
    providing PyTorch nn.Module functionality.

    Args:
        config (MLPConfig): Configuration object containing model architecture
            and hyperparameter specifications.

    Attributes:
        model (nn.Sequential): The sequential neural network layers.
        activation_func (nn.Module): The activation function module used
            between hidden layers.
        config (MLPConfig): The configuration used to build this model.

    Example:
        Basic usage:
        >>> config = MLPConfig(
        ...     input_size=784,
        ...     hidden_sizes=(128, 64),
        ...     output_size=10
        ... )
        >>> model = MLP(config)
        >>> model.to('cuda')

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

        Prediction example:
        >>> predictions = model.predict(test_loader, device='cuda')

    Note:
        - The model automatically detects task type during training and prediction
        - Supports multiclass classification, binary classification, and regression
        - No activation function is applied to the final output layer
        - Integrates with both PyTorch optimizers and the broader training framework
    """

    def __init__(self, config: MLPConfig) -> None:
        """Initialize the MLP model with the given configuration.

        Builds the neural network architecture by creating a sequence of
        linear layers with activation functions between them.

        Args:
            config (MLPConfig): Configuration object specifying the model
                architecture and hyperparameters.
        """
        # Initialize both parent classes
        nn.Module.__init__(self)
        BaseModels.__init__(self, config)

        # Build the network layers
        layers: list[nn.Module] = []
        self.activation_func = self._get_activation_function(config.activation_function)

        # Create hidden layers with activation functions
        in_size = config.input_size
        for h in config.hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(self.activation_func)
            in_size = h

        # Add final output layer (no activation - handled by loss function)
        layers.append(nn.Linear(in_size, config.output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the neural network.

        Computes the forward propagation of input through all network layers.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, output_size).

        Note:
            This method is called automatically during training and prediction.
            For manual forward passes, you can also call model(x) directly.
        """
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
        with PyTorch optimizers.

        Returns:
            ParamsT: Iterator over model parameters that can be passed
                directly to PyTorch optimizers.

        Example:
            >>> model = MLP(config)
            >>> optimizer = torch.optim.Adam(model.get_params(), lr=0.001)
        """
        return self.model.parameters()

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
        # Move model to specified device
        self.model.to(device)

        # Initialize loss tracking dictionary
        loss_dict: dict[str, list[Any]] = {"train_loss": []}

        # Add validation loss tracking if validation data provided
        if val_loader is not None:
            loss_dict["val_loss"] = []

        # Training loop with progress bar
        with tqdm(
            range(epochs), desc="Training", unit="epoch", leave=False
        ) as progress_bar:
            for epoch_idx in progress_bar:
                progress_bar.set_description(f"Epoch {epoch_idx + 1}/{epochs}")

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

                    # Update progress bar with both losses
                    progress_bar.set_postfix(
                        {
                            "train_loss": f"{avg_epoch_train_loss:.4f}",
                            "val_loss": f"{val_loss:.4f}",
                        }
                    )
                else:
                    # Update progress bar with training loss only
                    progress_bar.set_postfix(
                        {"train_loss": f"{avg_epoch_train_loss:.4f}"}
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
