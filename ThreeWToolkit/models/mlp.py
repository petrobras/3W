from collections.abc import Sized
from typing import Any, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from enum import Enum
from tqdm import tqdm
from typing import Tuple, Optional
from torch.utils.data import DataLoader, Dataset
from ThreeWToolkit.core.enums import ModelTypeEnum
from ThreeWToolkit.metrics import accuracy_score
from ..core.base_models import BaseModels, ModelsConfig


class ActivationFunction(Enum):
    LINEAR = nn.Identity()
    RELU = nn.ReLU()
    SIGMOID = nn.Sigmoid()
    TANH = nn.Tanh()


class MLPConfig(ModelsConfig):
    model_type: ModelTypeEnum = ModelTypeEnum.MLP
    random_seed: int = 42
    input_size: int
    hidden_sizes: Tuple[int, ...]
    output_size: int
    activation_function: ActivationFunction


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
        return len(self.samples)  # equal to len(self.labels)


class MLP(nn.Module):
    def __init__(self, config: MLPConfig):
        super(MLP, self).__init__()
        layers = []
        in_size = config.input_size
        for h in config.hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(config.activation_function.value)
            in_size = h
        layers.append(nn.Linear(in_size, config.output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class MLPTrainer(BaseModels):
    def __init__(
        self,
        config: MLPConfig,
        batch_size: int = 32,
        lr: float = 1e-4,
        nfolds: int = 5,
        seed: int = 42,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.batch_size = batch_size
        self.config = config
        self.lr = lr
        self.nfolds = nfolds
        self.seed = seed
        self.class_weights = (
            class_weights.to(self.device) if class_weights is not None else None
        )

        self.fold_val_accuracies: list = []
        self.best_model: MLP | None = None

        # Dictionary to store the losses along training
        self.history: dict = {}

    def _get_model(self):
        return MLP(self.config).to(self.device)

    def _get_optimizer(self, model: MLP):
        return optim.Adam(model.parameters(), lr=self.lr)

    def _get_fn_cost(self):
        return nn.CrossEntropyLoss(weight=self.class_weights)

    def get_params(self):
        """
        Retrieve the current configuration parameters.
        """
        return self.config.__dict__

    def set_params(self, **params):
        """
        Update the configuration parameters.
        Only valid parameters present in the config are updated.
        """
        for key, value in params.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")

    def create_dataloader(self, x, y, shuffle: bool):
        dataset = LabeledSubset(x, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def evaluate(self, x: list, y: list, metrics: list[Callable]):
        return {metric.__name__: metric(y, x) for metric in metrics}

    def run_evaluation_epoch(self, model, loader, criterion=None):
        if criterion is None:
            criterion = self._get_fn_cost()

        model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        total_eval_samples = 0
        with torch.no_grad():
            for x_values, y_values in loader:
                x_values, y_values = (
                    x_values.to(self.device).float(),
                    y_values.to(self.device).long(),
                )

                out = model(x_values)
                loss = criterion(out, y_values)
                running_loss += loss.item() * x_values.size(0)

                preds = out.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_values.cpu().numpy())
                total_eval_samples += x_values.size(0)

        avg_loss = running_loss / total_eval_samples
        metrics = self.evaluate(all_preds, all_labels, [accuracy_score])
        return avg_loss, metrics

    def train_epoch(self, model, train_loader, criterion, optimizer):
        model.train()
        epoch_train_loss = 0.0
        total_train_samples = 0
        for x_values, y_values in train_loader:
            x_values, y_values = (
                x_values.to(self.device).float(),
                y_values.to(self.device).long(),
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
        x: Dataset[Any],
        y: Any = None,
        epochs: int = 10,
        split_train: float = 0.8,
        **kwargs,
    ) -> None:
        # TODO: This training does not consider nfolds!

        # Required by linting to guarantee that the datasets are Sized and we can use len()
        if not isinstance(x, Sized):
            raise TypeError("Expected Sized Dataset.")

        # Clean history dictionary
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

        # Shuffle the dataset
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        train_idx = idx[: int(len(idx) * split_train)]
        val_idx = idx[len(train_idx) :]

        # TODO: Change this part to use our data loader
        # Create the train dataset
        x_train_samples = x[train_idx]
        y_train_labels = y[train_idx]
        # Create the validation dataset
        x_val_samples = x[val_idx]
        y_val_labels = y[val_idx]
        # Create dataloaders
        train_loader = self.create_dataloader(
            x_train_samples, y_train_labels, shuffle=True
        )
        val_loader = self.create_dataloader(x_val_samples, y_val_labels, shuffle=False)

        print(f"\t Training samples: {len(train_idx)}")
        print(f"\t Validation samples: {len(val_idx)}")

        model = self._get_model()
        criterion = self._get_fn_cost()
        optimizer = self._get_optimizer(model)

        # Store the best model
        best_model = {
            "epoch": -1,
            "model": None,
            "val_loss": float("inf"),
            "val_acc": -1,
        }

        # Train for epochs
        progress_bar = tqdm(range(epochs))
        for epoch_idx in progress_bar:
            progress_bar.set_description(f"Training Epoch [{epoch_idx + 1}/{epochs}]")

            # Train for this epoch
            avg_epoch_train_loss = self.train_epoch(
                model, train_loader, criterion, optimizer
            )
            # Run a single evaluation epoch with the validation set
            avg_val_loss, metrics = self.run_evaluation_epoch(
                model, val_loader, criterion
            )
            val_acc = metrics["accuracy_score"]

            # Print the results for this epoch
            print(f"Epoch {epoch_idx + 1}/{epochs}")
            print(f"\tTrain loss: {avg_epoch_train_loss:.4f}")
            print(f"\tVal loss: {avg_val_loss:.4f}")
            print(f"\tVal acc: {val_acc:.4f}")

            # After training epochs, store the results
            self.history["train_loss"].append(avg_epoch_train_loss)
            self.history["val_loss"].append(avg_val_loss)
            self.history["val_acc"].append(val_acc)

            # Update the best model if the current epoch has the best validation loss
            if avg_val_loss < best_model["val_loss"]:
                best_model["epoch"] = epoch_idx
                best_model["model"] = model.eval()
                best_model["val_loss"] = avg_val_loss
                best_model["val_acc"] = val_acc
                print(f"New best model at epoch {epoch_idx + 1}")

        # Store the best model
        if isinstance(best_model["model"], (MLP, type(None))):
            self.best_model = best_model["model"]
        else:
            raise TypeError(
                f"best_model['model'] is not an MLP or None, got {type(best_model['model'])}"
            )

    def predict(self, model, loader):
        model.eval()
        y_pred = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device).float()
                outputs = model.forward(X_batch)
                _, preds = torch.max(outputs, 1)
                y_pred.extend(preds.cpu().numpy())
        return np.array(y_pred)
