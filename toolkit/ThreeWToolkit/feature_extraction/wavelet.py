import numpy as np
import pandas as pd
from pydantic import Field, field_validator
import pywt

from ..core.base_feature_extractor import BaseFeatureExtractor, BaseFeatureExtractorConfig
from ..core.dataset_outputs import DatasetOutputs


class WaveletConfig(BaseFeatureExtractorConfig):
    """Configuration for the Wavelet feature extractor."""

    wavelet: str = Field(default="haar", description="Name of the wavelet to use for decomposition. Must be a valid wavelet name supported by PyWavelets.")

    level: int = Field(default=7, gt=1, description="Number of decomposition levels for wavelet transform. Must be a positive integer (>= 1).")

    full: bool = Field(default=False, description="Whether to return both approximation and detail coefficients or only detail coefficients (False).")

    target_: type = Field(default_factory=lambda: WaveletFeatures)

    @field_validator("wavelet")
    @classmethod
    def check_wavelet_name(cls, v):
        """Validates that the wavelet name is supported."""
        if v not in pywt.wavelist(): # type: ignore
            raise ValueError(f"Unknown wavelet: '{v}'.")
        return v


class WaveletFeatures(BaseFeatureExtractor):
    """
    Extracts wavelet features from windowed time series data using Stationary Wavelet Transform (SWT).

    Supports both univariate and multivariate analysis using PyWavelets library.

    IMPORTANT: Data must be already windowed. Input data should be a multi-index dataframe where each row corresponds to
    a window of data for a specific variable. The window size should match 2**level for proper wavelet decomposition.


    Input format: DataFrame with windowed data where each row is a window
    Output format:
    - Univariate: [var1_feature1, var1_feature2, ..., label]
    - Multivariate: [var1_feature1, var2_feature1, ..., var1_feature2, var2_feature2, ..., label]
    """

    def __init__(self, config: WaveletConfig):
        """
        Initialize the wavelet feature extractor.

        Args:
            config: Configuration object with wavelet parameters
        """
        self.config: WaveletConfig = config

        # Initialize wavelet filter matrix
        self._initialize_wavelet_filters()

    def _initialize_wavelet_filters(self):
        """
        Initialize the wavelet filter matrix using SWT decomposition.
        Creates filter matrix for efficient batch processing.
        """
        # Create impulse response for filter matrix generation
        filter_size = 2 ** self.config.level
        impulse = np.zeros(filter_size)
        impulse[-1] = 1

        # Perform SWT decomposition on impulse to get coefficients
        swt_coefficients = pywt.swt(
                impulse,
                self.config.wavelet,
                level=self.config.level,
                trim_approx=not self.config.full)

        if self.config.full:
            # Each level has both approximation and detail coefficients
            self.features = [f"{f}{level}" for level in range(self.config.level, 0, -1) for f in ["A", "D"]] + ["A0"]
            responses = [wave for level in swt_coefficients for wave in level] + [impulse]
        else: # Approximation just for the last level, details for all levels
            self.features = [f"A{self.config.level}"] + [f"D{level}" for level in range(self.config.level, 0, -1)] + ["A0"]
            responses = swt_coefficients + [impulse]

        self.waves = np.array(responses) # Shape: (num_features, filter_size)


    def transform(self, data: DatasetOutputs) -> DatasetOutputs:
        """
        Apply wavelet transform to the input data. Input data should be a multi-index dataframe where each row
        corresponds to a window of data for a specific variable. The window size should match 2**level for proper
        wavelet decomposition.

        Args:
            data: DatasetOutputs with signal and label data

        Returns:
            DatasetOutputs with wavelet features in the signal and original labels
        """
        signal = data.signal.values # Shape: ((num_windows * num_features), window_size)

        if signal.shape[1] != 2**self.config.level:
            raise ValueError(f"Input window size must be 2**level ({2**self.config.level}), but got {signal.shape[1]}.")

        # Multiply each window by each wavelet filter using matrix multiplication to get the coefficients.
        feats = np.einsum("ik,lk->il", signal, self.waves) # (num_windows * num_features, num_wavelet_features)

        signal = pd.DataFrame(feats, index=data.signal.index, columns=self.features) # assemble multiindex DataFrame with features as cols
        signal = signal.unstack("variable") # unstack variable to get per-variable features in columns
        signal.columns = ['_'.join(col).strip() for col in signal.columns] # flatten multiindex columns

        return DatasetOutputs(signal=signal, label=data.label, metadata=data.metadata.copy()) # type: ignore
