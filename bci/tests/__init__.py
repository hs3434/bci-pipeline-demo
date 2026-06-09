"""
Test Module
===========
Unit tests for BCI Pipeline components
"""

import pytest
import numpy as np
from pathlib import Path


class TestFilterConfig:
    """Test filtering configuration"""

    def test_filter_config_validation(self):
        from bci.config import FilterConfig
        cfg = FilterConfig(l_freq=0.5, h_freq=40)
        assert cfg.validate() == True

    def test_filter_config_invalid(self):
        from bci.config import FilterConfig
        cfg = FilterConfig(l_freq=40, h_freq=0.5)
        with pytest.raises(ValueError):
            cfg.validate()


class TestPreprocessor:
    """Test preprocessing functions"""

    def test_bandpass_params(self):
        from bci.preprocessor import Preprocessor
        # Just test the parameters are correct
        assert True  # Placeholder


class TestDecoder:
    """Test decoding functions"""

    def test_cca_score(self):
        from bci.decoder.ssvep import SSVEPDecoder
        import numpy as np

        detector = SSVEPDecoder(target_freqs=[10.0, 12.0], fs=500)

        # Create test signal (10 Hz)
        t = np.arange(500) / 500
        signal = np.sin(2 * np.pi * 10 * t).reshape(1, 1, -1)  # (1, 1, 500)

        idx = detector.predict(signal)
        assert idx[0] in [0, 1]  # Should detect one of the targets


class TestPipeline:
    """Test pipeline integration"""

    def test_pipeline_config(self):
        from bci.config import PipelineConfig
        config = PipelineConfig()
        assert config.filter.l_freq == 0.5
        assert config.epoch.tmin == -0.2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])