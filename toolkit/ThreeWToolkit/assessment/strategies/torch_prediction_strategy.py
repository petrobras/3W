import numpy as np
import torch
import torch.nn as nn

from typing import Any
from ...core.base_prediction_strategies import PredictionStrategy
from ...core.enums import TaskTypeEnum


class TorchPredictionStrategy(PredictionStrategy):
    """Prediction strategy for PyTorch-based models.

    This strategy supports both classification (binary and multiclass)
    and regression tasks, controlled explicitly via the `task` parameter.
    """

    def predict(
        self,
        model: nn.Module,
        task: TaskTypeEnum | None = TaskTypeEnum.CLASSIFICATION,
        **kwargs,
    ) -> np.ndarray:
        """Generate predictions using a PyTorch model.

        Args:
            model (nn.Module): Trained PyTorch model used for inference.
            task (TaskTypeEnum | None): Task type indicating how outputs should
                be interpreted. Defaults to TaskTypeEnum.CLASSIFICATION.
            **kwargs: Additional keyword arguments:
                loader (DataLoader): PyTorch DataLoader providing input batches.
                device (str): Device to run inference on (e.g., "cpu", "cuda").
                threshold (float): Threshold for binary classification.
                    Defaults to 0.5.

        Returns:
            np.ndarray: Array containing model predictions.

        Raises:
            AssertionError: If the model is not a valid torch.nn.Module.
            ValueError: If no DataLoader is provided or the task type is unknown.
        """
        assert model is not None and isinstance(
            model, nn.Module
        ), "Model must be a valid torch.nn.Module before prediction."

        loader = kwargs.get("loader")
        device = kwargs.get("device", "cpu")

        if loader is None:
            raise ValueError("A DataLoader must be provided via 'loader'.")

        model.eval()
        model.to(device)

        y_pred: list[Any] = []

        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(device).float()
                outputs = model(X_batch)

                if task == TaskTypeEnum.CLASSIFICATION:
                    threshold = kwargs.get("threshold", 0.5)
                    preds = self._predict_classification(outputs, threshold)

                elif task == TaskTypeEnum.REGRESSION:
                    preds = self._predict_regression(outputs)

                else:
                    raise ValueError(f"Unknown task type: {task}")

                y_pred.extend(preds.cpu().numpy())

        return np.asarray(y_pred)

    def _predict_classification(
        self, outputs: torch.Tensor, threshold: float
    ) -> torch.Tensor:
        """Convert raw model outputs into class predictions.

        Supports binary classification (N, 1) and multiclass classification (N, C).

        Args:
            outputs (torch.Tensor): Raw model outputs.
            threshold (float): Threshold used for binary classification.

        Returns:
            torch.Tensor: Predicted class labels.

        Raises:
            ValueError: If output tensor has an invalid shape.
        """
        if outputs.dim() != 2:
            raise ValueError("Classification outputs must be 2D (N, C).")

        if outputs.shape[1] == 1:
            probs = torch.sigmoid(outputs)
            return (probs > threshold).long().squeeze(1)

        return torch.argmax(outputs, dim=1)

    def _predict_regression(self, outputs: torch.Tensor) -> torch.Tensor:
        """Convert raw model outputs into regression predictions.

        Args:
            outputs (torch.Tensor): Raw model outputs with shape (N, 1) or (N,).

        Returns:
            torch.Tensor: Regression predictions.
        """
        if outputs.dim() == 2 and outputs.shape[1] == 1:
            return outputs.squeeze(1)

        return outputs.squeeze()

    def requires_dataloader(self) -> bool:
        """Check if the prediction strategy requires a DataLoader.

        Returns:
            bool: True because this strategy requires a DataLoader.
        """
        return True
