from collections.abc import Sized
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from enum import Enum
from tqdm import tqdm
from typing import Tuple, Optional
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from ..core.base_models import BaseModels, ModelsConfig


class ActivationFunction(Enum):
    # LINEAR = nn.Linear()
    RELU = nn.ReLU()
    SIGMOID = nn.Sigmoid()
    TANH = nn.Tanh()


class MLPConfig(ModelsConfig):
    # learning_rate: float = Field(..., lt = 0.0, description="Learning rate must be < 0.")
    input_size: int
    hidden_sizes: Tuple[int, ...]
    output_size: int
    activation_function: ActivationFunction


class MLP(nn.Module, BaseModels):
    def __init__(self, config: MLPConfig):
        super(MLP, self).__init__(config)
        layers = []
        in_size = config.input_size
        for h in config.hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(config.activation_function.value())
            in_size = h
        layers.append(nn.Linear(in_size, config.output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class MLPTrainer:
    def __init__(
        self,
        train_dataset: Dataset[Any],
        test_dataset: Dataset[Any],
        config: MLPConfig,
        batch_size: int = 32,
        lr: float = 1e-4,
        nfolds: int = 5,
        seed: int = 42,
        class_weights: Optional[torch.Tensor] = None,
    ):
        # Required by linting to guarantee that the datasets are Sized and we can use len()
        if not isinstance(train_dataset, Sized):
            raise TypeError("Expected Sized Dataset.")
        if not isinstance(test_dataset, Sized):
            raise TypeError("Expected Sized Dataset.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.config = config
        self.lr = lr
        self.nfolds = nfolds
        self.seed = seed
        self.class_weights = (
            class_weights.to(self.device) if class_weights is not None else None
        )

        self.models: list = []
        self.fold_val_accuracies: list = []
        # Dicionário para armazenar o histórico de médias
        self.history: dict = {"train_loss": [], "val_loss": [], "val_acc": []}

    def _get_model(self):
        return MLP(self.config).to(self.device)

    def _get_optimizer(self, model: MLP):
        return optim.Adam(model.parameters(), lr=self.lr)

    def _get_fn_cost(self):
        return nn.CrossEntropyLoss(weight=self.class_weights)

    def create_dataloader(self, dataset, shuffle: bool):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    # Método para validar a cada época, retornando loss e acurácia
    def validate_epoch(self, model, loader, criterion):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xvalues, yvalues in loader:
                xvalues, yvalues = (
                    xvalues.to(self.device).float(),
                    yvalues.to(self.device).long(),
                )

                out = model(xvalues)
                loss = criterion(out, yvalues)
                running_loss += loss.item() * xvalues.size(0)

                preds = out.argmax(dim=1)
                correct += (preds == yvalues).sum().item()
                total += yvalues.size(0)

        avg_loss = running_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def train(self, epochs: int = 10):
        skf = StratifiedKFold(
            n_splits=self.nfolds, shuffle=True, random_state=self.seed
        )
        y_train_values = np.array(
            [self.train_dataset[i][1] for i in range(len(self.train_dataset))]
        )

        # Listas para armazenar o histórico de cada fold
        all_folds_train_loss = []
        all_folds_val_loss = []
        all_folds_val_acc = []

        for idx_fold, (train_idx, val_idx) in enumerate(
            skf.split(range(len(self.train_dataset)), y_train_values), 1
        ):  # type: ignore
            print(f"\n### Fold {idx_fold}/{self.nfolds} ###")

            train_subset = Subset(self.train_dataset, train_idx)
            val_subset = Subset(self.train_dataset, val_idx)

            train_loader = self.create_dataloader(train_subset, True)
            val_loader = self.create_dataloader(val_subset, False)

            model = self._get_model()
            criterion = self._get_fn_cost()
            optimizer = self._get_optimizer(model)

            # Histórico para o fold atual
            fold_train_loss_hist = []
            fold_val_loss_hist = []
            fold_val_acc_hist = []

            for epoch in tqdm(range(epochs), desc=f"Fold {idx_fold} Training"):
                model.train()
                epoch_train_loss = 0.0
                for xvalues, yvalues in train_loader:
                    xvalues, yvalues = (
                        xvalues.to(self.device).float(),
                        yvalues.to(self.device).long(),
                    )

                    optimizer.zero_grad()
                    outputs = model(xvalues)
                    loss = criterion(outputs, yvalues)
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item() * xvalues.size(0)

                # Calcular e armazenar as perdas da época
                avg_epoch_train_loss = epoch_train_loss / len(train_loader.dataset)
                val_loss, val_acc = self.validate_epoch(model, val_loader, criterion)

                fold_train_loss_hist.append(avg_epoch_train_loss)
                fold_val_loss_hist.append(val_loss)
                fold_val_acc_hist.append(val_acc)

            # Armazena o histórico do fold
            all_folds_train_loss.append(fold_train_loss_hist)
            all_folds_val_loss.append(fold_val_loss_hist)
            all_folds_val_acc.append(fold_val_acc_hist)

            # A acurácia final do fold é a da última época
            final_fold_acc = fold_val_acc_hist[-1]
            self.fold_val_accuracies.append(final_fold_acc)
            print(
                f"Fold {idx_fold} - Acurácia de Validação Final: {final_fold_acc:.4f}"
            )

            self.models.append(model)

        # Calcular a média das curvas de loss/acc entre os folds
        self.history["train_loss"] = np.mean(all_folds_train_loss, axis=0)
        self.history["val_loss"] = np.mean(all_folds_val_loss, axis=0)
        self.history["val_acc"] = np.mean(all_folds_val_acc, axis=0)

        mean_val_acc = np.mean(self.fold_val_accuracies)
        print("\nTreinamento concluído.")
        print(
            f"Média da Acurácia de Validação nos {self.nfolds} folds: {mean_val_acc:.4f}"
        )

    def evaluate(self, model, loader):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for xvalues, yvalues in loader:
                xvalues, yvalues = (
                    xvalues.to(self.device).float(),
                    yvalues.to(self.device).long(),
                )
                out = model(xvalues)
                preds = out.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yvalues.cpu().numpy())
        accuracy = accuracy_score(all_labels, all_preds)
        return np.array(all_preds), accuracy

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
