import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

from ThreeWToolkit.feature_extraction.extract_wavelet_features import ExtractWaveletFeatures, WaveletConfig


class TestExtractWaveletFeatures:
    """Unit tests for the refactored ExtractWaveletFeatures class."""

    def test_basic_extraction(self):
        """Tests basic wavelet extraction with an overlap config."""
        data = pd.DataFrame({"signal": [1.0, 2.0, 3.0, 4.0]})
        data.index.name = "time"
        
        config = WaveletConfig(level=1, overlap=0.5, offset=0)
        extractor = ExtractWaveletFeatures(config)
        result = extractor(data)

        # The pywt library's convention results in positive Detail coefficients
        # for an increasing signal.
        expected_data = {
            "signal_A1": [2.121320, 3.535534, 4.949747],
            "signal_D1": [0.707107, 0.707107, 0.707107],
            "signal_A0": [2.0, 3.0, 4.0]
        }
        
        expected_result = pd.DataFrame(expected_data, index=data.index[1:])
        expected_result.index.name = "time"

        assert_frame_equal(result, expected_result, atol=1e-6)

    def test_overlap_parameter(self):
        """Tests if the 'overlap' parameter produces the correct number of windows."""
        data = pd.DataFrame({'signal': np.arange(1.0, 9.0)}) # 8 data points

        config = WaveletConfig(level=2, overlap=0.0)
        extractor = ExtractWaveletFeatures(config)
        result = extractor(data)
        
        # With overlap=0.0 (stride=4), expect 2 windows: [1,2,3,4] and [5,6,7,8]
        assert len(result) == 2
        assert result.index[0] == 3 
        assert result.index[1] == 7

    def test_offset_parameter(self):
        """Tests if the 'offset' parameter correctly ignores initial data points."""
        data = pd.DataFrame({'signal': [10, 20, 1, 2, 3, 4]})
        
        config = WaveletConfig(level=1, overlap=0.5, offset=2)
        extractor = ExtractWaveletFeatures(config)
        result = extractor(data)
        
        # First window processed is [1, 2]. Its A0 value (last element) should be 2.0
        assert result['signal_A0'].iloc[0] == 2.0
        assert len(result) == 3
        
    def test_invalid_overlap_raises_error(self):
        """
        Tests that the Pydantic validator raises a ValueError for invalid overlap values.
        """
        # Test case where overlap is exactly 1.0 (should fail)
        with pytest.raises(ValueError, match="Overlap must be in the range"):
            WaveletConfig(level=1, overlap=1.0)

        # Test case where overlap is negative (should fail)
        with pytest.raises(ValueError, match="Overlap must be in the range"):
            WaveletConfig(level=1, overlap=-0.1)

    def test_insufficient_data(self):
        """Tests that an empty DataFrame is returned for insufficient data."""
        data = pd.DataFrame({'signal': [1.0, 2.0, 3.0]})
        config = WaveletConfig(level=2, overlap=0.0)
        extractor = ExtractWaveletFeatures(config)
        result = extractor(data)
        assert result.empty

    def test_handles_mixed_success_from_windowing(self, monkeypatch):
        """Tests the mixed scenario where one column succeeds and one fails."""
        def mock_windowing_mixed(X: pd.Series, *args, **kwargs):
            if X.name == "s1":
                return pd.DataFrame(np.random.rand(5, kwargs.get("window_size", 4)))
            else:
                return pd.DataFrame()
        
        monkeypatch.setattr("ThreeWToolkit.feature_extraction.extract_wavelet_features.windowing",
                            mock_windowing_mixed)
        
        data = pd.DataFrame({"s1": np.arange(20), "s2": np.arange(20, 40)})
        config = WaveletConfig(level=2, overlap=0.5)
        extractor = ExtractWaveletFeatures(config)
        result = extractor(data)
        
        assert not result.empty
        assert "s1_A1" in result.columns
        assert "s2_A1" not in result.columns
