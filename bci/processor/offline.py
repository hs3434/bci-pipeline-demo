"""
Offline Processor
=================
Batch signal processing: filtfilt, full-dataset normalization, ICA.
Pursues maximum accuracy with access to the entire signal.
"""
from __future__ import annotations
from typing import List
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


class OfflineProcessor:
    """Batch (offline) signal processor.

    Uses zero-phase filtering (filtfilt) and full-dataset operations.
    Not suitable for real-time use — use OnlineProcessor for streaming.
    """

    def bandpass(self, data: np.ndarray, sfreq: float,
                 l_freq: float = 0.5, h_freq: float = 40.0,
                 order: int = 4) -> np.ndarray:
        """Apply zero-phase bandpass filter."""
        nyq = sfreq / 2
        b, a = butter(order, [l_freq / nyq, h_freq / nyq], btype='band')
        result = np.zeros_like(data)
        for ch in range(data.shape[0]):
            result[ch] = filtfilt(b, a, data[ch])
        return result

    def notch(self, data: np.ndarray, sfreq: float,
              freqs: List[int] | None = None,
              q: int = 30) -> np.ndarray:
        """Apply zero-phase notch filters for powerline noise."""
        if freqs is None:
            freqs = [50, 100]
        result = data.copy()
        nyq = sfreq / 2
        for freq in freqs:
            if freq < nyq:
                b, a = iirnotch(freq, q, sfreq)
                for ch in range(result.shape[0]):
                    result[ch] = filtfilt(b, a, result[ch])
        return result

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Zero-mean unit-variance normalization over the entire dataset."""
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-12:
            return data - mean
        return (data - mean) / std
