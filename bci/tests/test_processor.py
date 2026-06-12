"""
Tests for bci.processor module (online streaming)
"""
from __future__ import annotations
import pytest
import numpy as np
from scipy.signal import butter, filtfilt


def _generate_test_signal(n_channels=4, n_samples=5000, sfreq=256.0):
    """Generate synthetic EEG-like signal with known frequency components"""
    t = np.arange(n_samples) / sfreq
    data = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        data[ch] = (
            np.sin(2 * np.pi * 10 * t) * 50    # 10 Hz alpha
            + np.sin(2 * np.pi * 50 * t) * 20  # 50 Hz line noise
            + np.random.randn(n_samples) * 5    # background noise
        )
    return data, sfreq


class TestOnlineProcessor:
    """Online (causal) processor — lfilter with state, sliding window"""

    @pytest.fixture
    def processor(self):
        from bci.processor.online import OnlineProcessor
        return OnlineProcessor(sfreq=256.0, n_channels=4)

    @pytest.fixture
    def signal(self):
        return _generate_test_signal()

    def test_bandpass_returns_same_shape_chunk(self, processor, signal):
        data, sfreq = signal
        chunk = data[:, :128]
        result = processor.bandpass(chunk, l_freq=1.0, h_freq=30.0)
        assert result.shape == chunk.shape

    def test_bandpass_causal_differs_from_filtfilt(self, processor, signal):
        data, sfreq = signal
        b, a = butter(4, [1.0 / (sfreq / 2), 30.0 / (sfreq / 2)], btype='band')  # type: ignore[assignment]  # scipy butter stub
        result_offline = filtfilt(b, a, data[0])
        result_online = processor.bandpass(data[:, :], l_freq=1.0, h_freq=30.0)
        assert not np.allclose(result_offline, result_online[0])

    def test_bandpass_state_maintained_across_chunks(self, processor):
        sfreq = 256.0
        t = np.arange(1024) / sfreq
        data = (np.sin(2 * np.pi * 10 * t) + np.random.randn(1024) * 0.5).reshape(1, -1)

        chunk1 = processor.bandpass(data[:, :512], l_freq=1.0, h_freq=30.0)
        chunk2 = processor.bandpass(data[:, 512:], l_freq=1.0, h_freq=30.0)
        full = self._fresh_bandpass(data, l_freq=1.0, h_freq=30.0, sfreq=sfreq)
        combined = np.hstack([chunk1, chunk2])

        match_start = min(200, data.shape[1] // 2)
        assert np.allclose(combined[0, match_start:], full[0, match_start:], atol=1e-10)

    def test_notch_returns_same_shape(self, processor, signal):
        data, sfreq = signal
        chunk = data[:, :128]
        result = processor.notch(chunk, freqs=[50])
        assert result.shape == chunk.shape

    def test_normalize_sliding_window(self, processor, signal):
        data, sfreq = signal
        result = processor.normalize(data)
        assert result.shape == data.shape

    def test_reset_state_clears_filter_memory(self, processor):
        sfreq = 256.0
        t = np.arange(512) / sfreq
        data = np.sin(2 * np.pi * 10 * t).reshape(1, -1)
        chunk1 = processor.bandpass(data[:, :256], l_freq=1.0, h_freq=30.0)
        processor.reset_state()
        chunk2 = processor.bandpass(data[:, 256:], l_freq=1.0, h_freq=30.0)
        assert chunk1.shape == chunk2.shape

    def test_remove_artifact_clips_large_values(self, processor):
        data = np.ones((2, 100)) * 10
        data[:, 50] = 500  # artifact
        result = processor.remove_artifact(data, threshold=200)
        assert np.max(np.abs(result)) <= 200

    def test_remove_artifact_preserves_small_values(self, processor):
        data = np.ones((2, 100)) * 10
        result = processor.remove_artifact(data, threshold=200)
        assert np.allclose(result[:, :49], 10)
        assert np.allclose(result[:, 51:], 10)

    def _fresh_bandpass(self, data, l_freq, h_freq, sfreq):
        """Standalone: apply lfilter from scratch (no state) for reference"""
        from scipy.signal import butter, lfilter
        nyq = sfreq / 2
        b, a = butter(4, [l_freq / nyq, h_freq / nyq], btype='band')  # type: ignore[assignment]  # scipy butter stub
        result = np.zeros_like(data)
        for ch in range(data.shape[0]):
            result[ch] = lfilter(b, a, data[ch])
        return result
