"""
StreamSource — Simulated Real-Time Data Feed
============================================
Reads a file in chunks, simulating a live EEG acquisition device.
Supports speed control, loop mode, seek, and progress tracking.
"""
from __future__ import annotations
from typing import Optional
from pathlib import Path
import numpy as np

from .base import DataSource


class StreamSource(DataSource):
    """Streaming data source that reads an EEG file chunk-by-chunk.

    Simulates real-time acquisition by feeding data in chunks.
    Supports configurable playback speed, loop mode, and seeking.

    For Qt-based GUI, use the chunk_ready signal to drive updates.
    For CLI testing, use read_chunk() directly.
    """

    def __init__(self, filepath: str | Path,
                 chunk_duration: float = 0.1):
        self.filepath = Path(filepath)
        self.chunk_duration = chunk_duration
        self._raw = None
        self._data = None
        self._position = 0
        self._speed = 1.0
        self._loop = False
        self._closed = False

    def open(self) -> None:
        import mne
        self._raw = mne.io.read_raw(self.filepath, preload=True)
        self._data = self._raw.get_data()
        self._position = 0
        self._closed = False

    def read_chunk(self, n_samples: int) -> Optional[np.ndarray]:
        if self._closed or self._data is None:
            return None
        total = self._data.shape[1]
        if self._position >= total:
            if self._loop:
                self._position = 0
            else:
                return None
        end = min(self._position + n_samples, total)
        chunk = self._data[:, self._position:end]
        self._position = end
        return chunk

    def seek(self, sample_idx: int) -> None:
        if self._data is not None:
            total = self._data.shape[1]
            self._position = max(0, min(sample_idx, total))

    def close(self) -> None:
        self._raw = None
        self._data = None
        self._closed = True

    def reset(self) -> None:
        self._position = 0

    def set_loop(self, enabled: bool) -> None:
        self._loop = enabled

    def set_speed(self, speed: float) -> None:
        self._speed = max(0.01, speed)

    @property
    def sfreq(self) -> float:
        return float(self._raw.info['sfreq']) if self._raw else 0.0

    @property
    def n_channels(self) -> int:
        return int(self._data.shape[0]) if self._data is not None else 0

    @property
    def total_samples(self) -> Optional[int]:
        return int(self._data.shape[1]) if self._data is not None else None

    @property
    def is_stream(self) -> bool:
        return True

    @property
    def position(self) -> int:
        return self._position

    @property
    def progress(self) -> int:
        if self._data is None or self._data.shape[1] == 0:
            return 0
        return min(100, int(self._position / self._data.shape[1] * 100))
