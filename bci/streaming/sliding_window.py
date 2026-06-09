"""
SlidingWindow — rolling sample buffer with decision-interval triggering.

Application-layer wrapper around any decoder's predict_proba.
Decoder-agnostic: works with LDA / CNN / SSVEP / FBCCA / Transformer.
"""
from __future__ import annotations
import numpy as np


class SlidingWindow:
    """Rolling buffer + ready/consume semantics.

    Usage:
        sw = SlidingWindow(n_channels=64, window_size=1000, decision_interval=25)
        for chunk in eeg_stream:
            sw.push(chunk)
            if sw.ready():
                window = sw.get_window()                 # (n_ch, window_size)
                probs = decoder.predict_proba(window[None])[0]  # (n_classes,)
                act(probs)
                sw.consume()
    """

    def __init__(self, n_channels: int, window_size: int, decision_interval: int):
        if window_size <= 0:
            raise ValueError(f"window_size={window_size} must be positive")
        if decision_interval <= 0:
            raise ValueError(f"decision_interval={decision_interval} must be positive")
        if decision_interval > window_size:
            raise ValueError(
                f"decision_interval={decision_interval} cannot exceed "
                f"window_size={window_size}"
            )
        self.n_channels = n_channels
        self.window_size = window_size
        self.decision_interval = decision_interval
        self._buf = np.zeros((n_channels, window_size), dtype=np.float32)
        self._n_filled = 0
        self._write_pos = 0
        self._since_last = 0

    def push(self, chunk: np.ndarray) -> None:
        """Append chunk. chunk: (n_channels, n_new_samples) or (n_channels,)."""
        if chunk.ndim == 1:
            chunk = chunk[:, None]
        if chunk.shape[0] != self.n_channels:
            raise ValueError(
                f"chunk.shape[0]={chunk.shape[0]} != n_channels={self.n_channels}"
            )
        n_new = chunk.shape[1]
        for i in range(n_new):
            self._buf[:, self._write_pos] = chunk[:, i]
            self._write_pos = (self._write_pos + 1) % self.window_size
        self._n_filled = min(self._n_filled + n_new, self.window_size)
        self._since_last += n_new

    def ready(self) -> bool:
        """True when buffer is full AND decision_interval samples accumulated."""
        return (
            self._n_filled >= self.window_size
            and self._since_last >= self.decision_interval
        )

    def get_window(self) -> np.ndarray:
        """Return (n_channels, window_size) in chronological order."""
        if self._n_filled < self.window_size:
            return self._buf[:, :self._n_filled].copy()
        return np.concatenate(
            [self._buf[:, self._write_pos:], self._buf[:, :self._write_pos]],
            axis=-1,
        ).copy()

    def consume(self) -> None:
        """Reset since_last counter (call after get_window)."""
        self._since_last = 0

    def reset(self) -> None:
        """Clear buffer (new trial / session)."""
        self._buf[:] = 0
        self._n_filled = 0
        self._write_pos = 0
        self._since_last = 0
