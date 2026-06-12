"""
Online Processor
================
Causal, real-time signal processing: lfilter with state, sliding-window
normalization, threshold-based artifact removal.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
from scipy.signal import butter, lfilter, iirnotch


class OnlineProcessor:
    """Real-time (streaming) signal processor.

    Uses causal filtering (lfilter) with maintained state between chunks.
    Suitable for real-time BCI data streams.
    """

    def __init__(self, sfreq: float, n_channels: int):
        self.sfreq = sfreq
        self.n_channels = n_channels
        self._bandpass_zi = None
        self._bandpass_state = None
        self._notch_zi = None
        self._notch_freqs = None
        self._running_mean = 0.0
        self._running_var = 1.0
        self._alpha = 0.01

    def bandpass(self, data: np.ndarray,
                 l_freq: float = 0.5, h_freq: float = 40.0,
                 order: int = 4) -> np.ndarray:
        """Apply causal bandpass filter with maintained state."""
        nyq = self.sfreq / 2
        b, a = butter(order, [l_freq / nyq, h_freq / nyq], btype='band')  # type: ignore[assignment]  # scipy butter stub returns complex union

        key = (l_freq, h_freq, order)
        if self._bandpass_state is None or self._bandpass_state != key:
            n_zi = max(len(a), len(b)) - 1
            self._bandpass_zi = np.zeros((n_zi, data.shape[0]))
            self._bandpass_state = key

        result = np.zeros_like(data)
        zi = self._bandpass_zi
        new_zi = np.zeros_like(zi) if zi is not None else np.zeros((0, 0))
        for ch in range(data.shape[0]):
            zi_ch = zi[:, ch] if zi is not None else np.array([])
            y, zf = lfilter(b, a, data[ch], zi=zi_ch)  # type: ignore[assignment]  # scipy lfilter stub
            result[ch] = y
            new_zi[:, ch] = zf
        self._bandpass_zi = new_zi
        return result

    def notch(self, data: np.ndarray,
              freqs: Sequence[int] | None = None,
              q: int = 30) -> np.ndarray:
        """Apply causal notch filter(s) with maintained state."""
        if freqs is None:
            freqs = [50, 100]
        nyq = self.sfreq / 2
        result = data.copy()

        if self._notch_zi is None or self._notch_freqs != tuple(freqs):
            n_ch = data.shape[0]
            self._notch_zi = {}
            for freq in freqs:
                if freq < nyq:
                    b, a = iirnotch(freq, q, self.sfreq)
                    self._notch_zi[freq] = np.zeros((max(len(a), len(b)) - 1, n_ch))
            self._notch_freqs = tuple(freqs)

        for freq in freqs:
            if freq >= nyq:
                continue
            b, a = iirnotch(freq, q, self.sfreq)
            zi = self._notch_zi.get(freq)
            if zi is None:
                continue
            new_zi = np.zeros_like(zi)
            for ch in range(result.shape[0]):
                y, zf = lfilter(b, a, result[ch], zi=zi[:, ch])
                result[ch] = y
                new_zi[:, ch] = zf
            self._notch_zi[freq] = new_zi
        return result

    def remove_artifact(self, data: np.ndarray,
                        threshold: float = 200e-6) -> np.ndarray:
        """Clip or zero out samples exceeding amplitude threshold."""
        result = data.copy()
        result = np.clip(result, -threshold, threshold)
        return result

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Exponential moving average normalization."""
        flat = data.ravel()
        batch_mean = np.mean(flat)
        batch_var = np.var(flat)
        alpha = self._alpha
        self._running_mean = (1 - alpha) * self._running_mean + alpha * batch_mean
        self._running_var = (1 - alpha) * self._running_var + alpha * batch_var
        if self._running_var < 1e-12:
            return data - self._running_mean
        return (data - self._running_mean) / np.sqrt(self._running_var)

    def reset_state(self) -> None:
        """Clear all internal filter states."""
        self._bandpass_zi = None
        self._bandpass_state = None
        self._notch_zi = None
        self._notch_freqs = None
        self._running_mean = 0.0
        self._running_var = 1.0
