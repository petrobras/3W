"""
LSTM Autoencoder Models for Anomaly Detection

This module provides stable LSTM autoencoder architectures optimized for
numerical stability and time series anomaly detection.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    RepeatVector,
    TimeDistributed,
    Dropout,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, Orthogonal
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np


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

    def build_model(self):
        """Build the stable LSTM autoencoder architecture."""
        print(f"🏗️ Building Stable LSTM Autoencoder:")
        print(f"   • Input shape: ({self.time_steps}, {self.n_features})")
        print(f"   • Encoder LSTM units: {self.lstm_units}")
        print(f"   • Latent dimension: {self.latent_dim}")
        print(f"   • Decoder LSTM units: {self.lstm_units}")

        # Clear any previous models to avoid memory issues
        tf.keras.backend.clear_session()

        # Input layer
        input_layer = Input(
            shape=(self.time_steps, self.n_features), name="input", dtype="float32"
        )

        # Encoder with careful initialization and dropout
        encoded = LSTM(
            self.lstm_units,
            activation="tanh",  # More stable than relu
            recurrent_activation="sigmoid",
            kernel_initializer=GlorotUniform(seed=42),
            recurrent_initializer=Orthogonal(seed=42),
            dropout=0.1,
            recurrent_dropout=0.1,
            name="encoder_lstm",
        )(input_layer)
        encoded = Dropout(0.2, name="encoder_dropout")(encoded)

        # Latent representation (bottleneck)
        latent = Dense(
            self.latent_dim,
            activation="tanh",  # More stable than relu
            kernel_initializer=GlorotUniform(seed=42),
            name="latent",
        )(encoded)

        # Decoder
        decoded = RepeatVector(self.time_steps, name="repeat_vector")(latent)
        decoded = LSTM(
            self.lstm_units,
            activation="tanh",  # More stable than relu
            recurrent_activation="sigmoid",
            kernel_initializer=GlorotUniform(seed=42),
            recurrent_initializer=Orthogonal(seed=42),
            dropout=0.1,
            recurrent_dropout=0.1,
            return_sequences=True,
            name="decoder_lstm",
        )(decoded)
        decoded = Dropout(0.2, name="decoder_dropout")(decoded)
        decoded = TimeDistributed(
            Dense(
                self.n_features,
                kernel_initializer=GlorotUniform(seed=42),
                activation="sigmoid",
            ),  # Sigmoid to match data range
            name="output",
        )(decoded)

        # Create autoencoder model
        self.model = Model(input_layer, decoded, name="stable_lstm_autoencoder")

        # Compile model with stable settings and gradient clipping
        self.model.compile(
            optimizer=Adam(
                learning_rate=0.0005, clipnorm=1.0  # Conservative learning rate
            ),  # Gradient clipping
            loss="mse",
            metrics=["mae"],
        )

        print("✅ Stable LSTM Autoencoder model created")
        print(f"   • Total parameters: {self.model.count_params():,}")
        print(f"   • Gradient clipping enabled (clipnorm=1.0)")
        print(f"   • Conservative learning rate (0.0005)")

        return self.model

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
            dict: Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        print(f"🚂 Training LSTM Autoencoder:")
        print(f"   • Training samples: {len(train_data)}")
        if val_data is not None:
            print(f"   • Validation samples: {len(val_data)}")
        else:
            print(f"   • No validation data - using all data for training")
        print(f"   • Max epochs: {epochs}")
        print(f"   • Batch size: {batch_size}")

        # Setup callbacks based on whether we have validation data
        callbacks = []

        if val_data is not None:
            # Conservative callbacks for stable training with validation
            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=10,  # More patience for stability
                restore_best_weights=True,
                verbose=1,
                min_delta=1e-6,
            )

            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
            )

            callbacks = [early_stopping, reduce_lr]
            validation_data = (val_data, val_data)
        else:
            # Without validation data, use simpler learning rate reduction
            reduce_lr = ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1
            )

            callbacks = [reduce_lr]
            validation_data = None

        # Train autoencoder
        self.history = self.model.fit(
            train_data,
            train_data,  # Autoencoder: input = target
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=True,
        )

        # Check if training was successful
        final_loss = self.history.history["loss"][-1]

        if val_data is not None:
            final_val_loss = self.history.history["val_loss"][-1]

            if np.isfinite(final_loss) and np.isfinite(final_val_loss):
                print(f"✅ Training successful - no NaN values")
                print(f"   • Final training loss: {final_loss:.6f}")
                print(f"   • Final validation loss: {final_val_loss:.6f}")
                return True
            else:
                print(f"❌ Training failed - NaN values detected")
                print(f"   • Final training loss: {final_loss}")
                print(f"   • Final validation loss: {final_val_loss}")
                return False
        else:
            if np.isfinite(final_loss):
                print(f"✅ Training successful - no NaN values")
                print(f"   • Final training loss: {final_loss:.6f}")
                return True
            else:
                print(f"❌ Training failed - NaN values detected")
                print(f"   • Final training loss: {final_loss}")
                return False

    def predict(self, data, verbose=0):
        """Predict/reconstruct data using the trained model."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        return self.model.predict(data, verbose=verbose)

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
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Create encoder model if it doesn't exist
        if not hasattr(self, '_encoder_model') or self._encoder_model is None:
            self._encoder_model = Model(
                inputs=self.model.input,
                outputs=self.model.get_layer('latent').output,
                name='encoder'
            )
        
        return self._encoder_model.predict(data, verbose=verbose)
