"""
LSTM Autoencoder Models for Anomaly Detection

This module provides a stable LSTM autoencoder (PyTorch, the same framework used
by the 3W Toolkit) optimized for numerical stability and time series anomaly
detection. The stability decisions are discussed in the companion book,
Appendix E (see the README of this resource).
"""

import copy

import numpy as np
import torch
from torch import nn

SEED = 42


class _LSTMAutoencoderNet(nn.Module):
    """LSTM encoder -> latent bottleneck -> LSTM decoder."""

    def __init__(self, time_steps, n_features, latent_dim, lstm_units):
        super().__init__()
        self.time_steps = time_steps
        self.encoder_input_dropout = nn.Dropout(0.1)
        self.encoder_lstm = nn.LSTM(n_features, lstm_units, batch_first=True)
        self.encoder_dropout = nn.Dropout(0.2)
        self.latent = nn.Linear(lstm_units, latent_dim)
        self.decoder_input_dropout = nn.Dropout(0.1)
        self.decoder_lstm = nn.LSTM(latent_dim, lstm_units, batch_first=True)
        self.decoder_dropout = nn.Dropout(0.2)
        self.output = nn.Linear(lstm_units, n_features)
        self._init_weights()

    def _init_weights(self):
        """Glorot input weights, orthogonal recurrent weights, unit forget bias."""
        for lstm in (self.encoder_lstm, self.decoder_lstm):
            nn.init.xavier_uniform_(lstm.weight_ih_l0)
            nn.init.orthogonal_(lstm.weight_hh_l0)
            nn.init.zeros_(lstm.bias_ih_l0)
            nn.init.zeros_(lstm.bias_hh_l0)
            hidden = lstm.hidden_size
            lstm.bias_ih_l0.data[hidden : 2 * hidden] = 1.0  # forget gate
        for dense in (self.latent, self.output):
            nn.init.xavier_uniform_(dense.weight)
            nn.init.zeros_(dense.bias)

    def encode(self, x):
        _, (hidden, _) = self.encoder_lstm(self.encoder_input_dropout(x))
        return torch.tanh(self.latent(self.encoder_dropout(hidden[-1])))

    def forward(self, x):
        latent = self.encode(x)
        repeated = latent.unsqueeze(1).expand(-1, self.time_steps, -1)
        decoded, _ = self.decoder_lstm(self.decoder_input_dropout(repeated))
        return torch.sigmoid(self.output(self.decoder_dropout(decoded)))


class StableLSTMAutoencoder:
    """Stable LSTM Autoencoder for time series anomaly detection."""

    def __init__(self, time_steps, n_features, latent_dim=16, lstm_units=64):
        """
        Initialize the LSTM Autoencoder.

        Args:
            time_steps (int): Number of time steps in input sequences
            n_features (int): Number of features per time step
            latent_dim (int): Dimension of the latent representation
            lstm_units (int): Number of LSTM units in encoder/decoder
        """
        self.time_steps = time_steps
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.model = None
        self.history = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_model(self):
        """Build the stable LSTM autoencoder architecture."""
        print("🏗️ Building Stable LSTM Autoencoder:")
        print(f"   • Input shape: ({self.time_steps}, {self.n_features})")
        print(f"   • Encoder LSTM units: {self.lstm_units}")
        print(f"   • Latent dimension: {self.latent_dim}")
        print(f"   • Decoder LSTM units: {self.lstm_units}")

        # Fixed seed: same initial weights (and batch order) on every run
        torch.manual_seed(SEED)

        self.model = _LSTMAutoencoderNet(
            self.time_steps, self.n_features, self.latent_dim, self.lstm_units
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        print("✅ Stable LSTM Autoencoder model created")
        print(f"   • Total parameters: {n_params:,}")
        print("   • Gradient clipping enabled (clipnorm=1.0)")
        print("   • Conservative learning rate (0.0005)")

        return self.model

    def _to_tensor(self, data):
        return torch.as_tensor(np.asarray(data, dtype=np.float32), device=self.device)

    def _epoch_loss(self, data, batch_size=256):
        """Mean squared error of the model (inference mode) over `data`."""
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(data), batch_size):
                batch = data[start : start + batch_size]
                total += torch.sum((self.model(batch) - batch) ** 2).item()
        return total / data.numel()

    def train(self, train_data, val_data=None, epochs=30, batch_size=32, verbose=1):
        """
        Train the autoencoder model.

        Args:
            train_data (np.array): Training data
            val_data (np.array, optional): Validation data. If None, no validation is used.
            epochs (int): Maximum number of epochs
            batch_size (int): Batch size for training
            verbose (int): Verbosity level

        Returns:
            bool: True if the training finished without NaN values
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        print("🚂 Training LSTM Autoencoder:")
        print(f"   • Training samples: {len(train_data)}")
        if val_data is not None:
            print(f"   • Validation samples: {len(val_data)}")
        else:
            print("   • No validation data - using all data for training")
        print(f"   • Max epochs: {epochs}")
        print(f"   • Batch size: {batch_size}")

        x_train = self._to_tensor(train_data)
        x_val = self._to_tensor(val_data) if val_data is not None else None

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        # Same schedule as before: halve the learning rate on plateaus
        monitored = "val_loss" if x_val is not None else "loss"
        plateau_patience = 5 if x_val is not None else 8
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=plateau_patience - 1,
            min_lr=1e-6,
            threshold=1e-4,
            threshold_mode="abs",
        )

        generator = torch.Generator().manual_seed(SEED)
        history = {"loss": [], "val_loss": []}
        best_val, best_state, wait = np.inf, None, 0

        for epoch in range(epochs):
            self.model.train()
            order = torch.randperm(len(x_train), generator=generator)
            for start in range(0, len(x_train), batch_size):
                batch = x_train[order[start : start + batch_size].to(self.device)]
                optimizer.zero_grad()
                loss = torch.mean((self.model(batch) - batch) ** 2)
                loss.backward()
                # Per-tensor clipping, as Keras' `clipnorm` does
                for parameter in self.model.parameters():
                    torch.nn.utils.clip_grad_norm_([parameter], max_norm=1.0)
                optimizer.step()

            history["loss"].append(self._epoch_loss(x_train))
            if x_val is not None:
                history["val_loss"].append(self._epoch_loss(x_val))
            scheduler.step(history[monitored][-1])

            if verbose:
                shown = ", ".join(f"{k}: {v[-1]:.6f}" for k, v in history.items() if v)
                print(f"   Epoch {epoch + 1}/{epochs} - {shown}")

            if not np.isfinite(history["loss"][-1]):
                break

            if x_val is not None:
                if history["val_loss"][-1] < best_val - 1e-6:
                    best_val = history["val_loss"][-1]
                    best_state = copy.deepcopy(self.model.state_dict())
                    wait = 0
                else:
                    wait += 1
                    if wait >= 10:
                        print(f"   Early stopping at epoch {epoch + 1}")
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.history = history

        # Check if training was successful
        final_loss = history["loss"][-1]

        if val_data is not None:
            final_val_loss = history["val_loss"][-1]

            if np.isfinite(final_loss) and np.isfinite(final_val_loss):
                print("✅ Training successful - no NaN values")
                print(f"   • Final training loss: {final_loss:.6f}")
                print(f"   • Final validation loss: {final_val_loss:.6f}")
                return True
            else:
                print("❌ Training failed - NaN values detected")
                print(f"   • Final training loss: {final_loss}")
                print(f"   • Final validation loss: {final_val_loss}")
                return False
        else:
            if np.isfinite(final_loss):
                print("✅ Training successful - no NaN values")
                print(f"   • Final training loss: {final_loss:.6f}")
                return True
            else:
                print("❌ Training failed - NaN values detected")
                print(f"   • Final training loss: {final_loss}")
                return False

    def _forward_in_batches(self, data, fn, batch_size=256):
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        tensor = self._to_tensor(data)
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(tensor), batch_size):
                outputs.append(fn(tensor[start : start + batch_size]).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def predict(self, data, verbose=0):
        """Predict/reconstruct data using the trained model."""
        return self._forward_in_batches(data, self.model)

    def get_reconstruction_errors(self, data, verbose=0):
        """
        Compute reconstruction errors for input data.

        Args:
            data (np.array): Input data to compute errors for
            verbose (int): Verbosity level

        Returns:
            np.array: Reconstruction errors (MSE per sample)
        """
        reconstructed = self.predict(data, verbose=verbose)
        # Compute MSE per sample
        mse_per_sample = np.mean(np.square(data - reconstructed), axis=(1, 2))
        return mse_per_sample

    def encode(self, data, verbose=0):
        """
        Get latent representations from the encoder.

        Args:
            data (np.array): Input data to encode
            verbose (int): Verbosity level

        Returns:
            np.array: Latent representations
        """
        return self._forward_in_batches(data, self.model.encode)

    def save(self, path):
        """Save the trained weights (PyTorch `state_dict`) to `path` (e.g. `.pt`)."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """Build the model and load weights previously saved with `save`."""
        self.build_model()
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        return self.model
