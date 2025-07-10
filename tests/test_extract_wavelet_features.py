import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

from ThreeWToolkit.feature_extraction.extract_wavelet_features import ExtractWaveletFeatures, WaveletConfig

# --- Início da suíte de testes ---
class TestExtractWaveletFeatures:
    """
    Unit tests for the ExtractWaveletFeatures class.
    """

    def test_basic_extraction(self):
        """
        Tests the Wavelet feature extraction for a basic case with a single column,
        level=1, and stride=1.
        """
        data = pd.DataFrame(
            {"signal": [1.0, 2.0, 3.0, 4.0]},
            index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"])
        )
        data.index.name = "time"
        
        config = WaveletConfig(level=1, stride=1, offset=0)
        extractor = ExtractWaveletFeatures(config)
        
        result = extractor(data)

        expected_index = data.index[1:]
        expected_data = {
            "signal_A1": [2.12132034, 3.53553391, 4.94974747],
            "signal_D1": [0.70710678, 0.70710678, 0.70710678],
            "signal_A0": [2.0, 3.0, 4.0]
        }
        expected_result = pd.DataFrame(expected_data, index=expected_index)
        expected_result.index.name = "time"

        assert_frame_equal(result, expected_result, atol=1e-6)


    def test_stride_parameter(self):
        """
        Tests if the 'stride' parameter correctly reduces the number of output windows.
        """
        data = pd.DataFrame({"signal": np.arange(1.0, 9.0)})
        
        config = WaveletConfig(level=2, stride=4, offset=0)
        extractor = ExtractWaveletFeatures(config)
        
        result = extractor(data)
        
        assert len(result) == 2
        assert result.index[0] == 3 
        assert result.index[1] == 7

    def test_offset_parameter(self):
        """
        Tests if the 'offset' parameter correctly ignores the initial data points.
        """
        data = pd.DataFrame({"signal": [10, 20, 1, 2, 3, 4]})
        data.index.name = "idx"

        config = WaveletConfig(level=1, stride=1, offset=2)
        extractor = ExtractWaveletFeatures(config)

        result = extractor(data)
 
        assert result["signal_A0"].iloc[0] == 2.0
        assert result.index[0] == 3
        assert len(result) == 3
        
    def test_multiple_columns(self):
        """
        Tests if the extractor works correctly for a DataFrame with multiple columns.
        """
        data = pd.DataFrame({
            "s1": [1.0, 2.0, 3.0, 4.0],
            "s2": [5.0, 6.0, 7.0, 8.0]
        })
        
        config = WaveletConfig(level=1, stride=1, offset=0)
        extractor = ExtractWaveletFeatures(config)

        result = extractor(data)

        assert len(result.columns) == 6
        assert 's1_A1' in result.columns
        assert 's2_A1' in result.columns
        assert 's1_A0' in result.columns
        assert 's2_D1' in result.columns

        assert result['s2_A0'].iloc[0] == 6.0

    def test_insufficient_data(self):
        """
        Tests if the function returns an empty DataFrame if the data is not sufficient
        for a single window.
        """
        data = pd.DataFrame({'signal': [1.0, 2.0, 3.0]})
        data.index.name = "id"
        
        config = WaveletConfig(level=2, stride=1, offset=0)
        extractor = ExtractWaveletFeatures(config)

        result = extractor(data)
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert result.index.name == "id"
        assert 'signal_A2' in result.columns
        assert 'signal_A0' in result.columns
        
    def test_exact_window_size_data(self):
        """
        Tests the case where the data size is exactly equal to one window size.
        """
        data = pd.DataFrame({'signal': [1.0, 2.0, 3.0, 4.0]})
        
        config = WaveletConfig(level=2, stride=1, offset=0)
        extractor = ExtractWaveletFeatures(config)
        
        result = extractor(data)
        
        assert len(result) == 1
        assert result.index[0] == 3
