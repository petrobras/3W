import numpy as np
import torch.nn as nn
import torch
from ..core.base_models import ModelsConfig
from ..core.enums import (
    ModelTypeEnum,
    ActivationFunctionEnum,
)


class MLPConfig(ModelsConfig):
    model_type: ModelTypeEnum = ModelTypeEnum.MLP
    input_size: int
    hidden_sizes: tuple[int, ...]
    output_size: int
    activation_function: ActivationFunctionEnum
    regularization: float | None


class LabeledSubset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
    ):
        if len(samples) != len(labels):
            raise ValueError("Samples and labels must have the same length.")

        self.samples = samples
        self.labels = labels

    def __getitem__(self, idx):
        x_i = self.samples[idx]
        y_i = self.labels[idx]
        return x_i, y_i

    def __len__(self):
        return len(self.samples)


class MLP(nn.Module):
    def __init__(self, config: MLPConfig):
        super(MLP, self).__init__()
        layers = []
        self.activation_func = self._get_activation_function(config.activation_function)
        in_size = config.input_size
        for h in config.hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(self.activation_func)
            in_size = h
        layers.append(nn.Linear(in_size, config.output_size))
        self.model = nn.Sequential(*layers)

    def _get_activation_function(self, activation: ActivationFunctionEnum):
        if activation == ActivationFunctionEnum.RELU:
            return nn.ReLU()
        elif activation == ActivationFunctionEnum.SIGMOID:
            return nn.Sigmoid()
        elif activation == ActivationFunctionEnum.TANH:
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

    def _run_evaluation_epoch(self, loader: DataLoader) -> tuple[float, dict]:
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        total_eval_samples = 0
        with torch.no_grad():
            for x_values, y_values in loader:
                x_values, y_values = (
                    x_values.to(self.device).float(),
                    y_values.to(self.device).float(),
                )

                out = self.model(x_values)
                loss = self.criterion(out, y_values)
                running_loss += loss.item() * x_values.size(0)

                preds = out.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_values.cpu().numpy())
                total_eval_samples += x_values.size(0)

        avg_loss = running_loss / total_eval_samples
        metrics = self.evaluate(all_preds, all_labels, [explained_variance_score])
        return avg_loss, metrics

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: Callable,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        model.train()
        epoch_train_loss = 0.0
        total_train_samples = 0
        for x_values, y_values in train_loader:
            x_values, y_values = (
                x_values.to(self.device).float(),
                y_values.to(self.device).float(),
            )
            # Forward pass
            optimizer.zero_grad()
            outputs = model(x_values)

            # Compute the loss
            loss = criterion(outputs, y_values)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            # Compute the loss for the batch
            samples_in_batch = x_values.size(0)
            epoch_train_loss += loss.item() * samples_in_batch
            total_train_samples += samples_in_batch

        # Compute the average loss for the epoch
        avg_epoch_train_loss = epoch_train_loss / total_train_samples
        return avg_epoch_train_loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        **kwargs,
    ) -> None:
        """
        Train the model using provided dataloaders.
        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
        """

        best_model = {
            "epoch": -1,
            "model": None,
            "val_loss": float("inf"),
        }

        with tqdm(range(epochs), desc="Training", unit="epoch") as progress_bar:
            for epoch_idx in progress_bar:
                progress_bar.set_description(f"Epoch {epoch_idx + 1}/{self.epochs}")

                avg_epoch_train_loss = self._train_epoch(
                    model=self.model, train_loader, criterion=criterion, optimizer=optimizer
                )

                if val_loader is not None:
                    avg_val_loss, metrics = self._run_evaluation_epoch(val_loader)
                    # val_acc = metrics["explained_variance_score"]
                else:
                    avg_val_loss = float("nan")

                # Store results
                self.history["train_loss"].append(avg_epoch_train_loss)
                self.history["val_loss"].append(avg_val_loss)

                # Show metrics in tqdm bar
                progress_bar.set_postfix(
                    {
                        "train_loss": f"{avg_epoch_train_loss:.4f}",
                        "val_loss": f"{avg_val_loss:.4f}",
                    }
                )

                # Update the best model if the current epoch has the best validation loss
                if avg_val_loss < best_model["val_loss"]:
                    best_model["epoch"] = epoch_idx
                    best_model["model"] = self.model.eval()
                    best_model["val_loss"] = avg_val_loss

    def evaluate(
        self, x: list | torch.Tensor, y: list | torch.Tensor, metrics: list[Callable]
    ) -> dict:
        return {metric.__name__: metric(y, x) for metric in metrics}

    def test(self, test_loader: DataLoader) -> tuple[float, dict]:
        return self._run_evaluation_epoch(test_loader)

    def predict(self, loader: DataLoader) -> np.ndarray:
        self.model.eval()
        y_pred = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device).float()
                outputs = self.model.forward(X_batch)
                _, preds = torch.max(outputs, 1)
                y_pred.extend(preds.cpu().numpy())
        return np.array(y_pred)
