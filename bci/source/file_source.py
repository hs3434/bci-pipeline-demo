"""
FileSource — Full-File Batch Data Access
========================================
Loads an entire EEG file into memory for offline analysis.
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from pathlib import Path

from .base import DataSource


class FileSource(DataSource):
    """Batch data source wrapping an EEG file loaded via MNE.

    Provides random access to the full dataset for offline analysis.
    """

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self._raw = None
        self._data = None
        self._times = None

    def open(self) -> None:
        import mne
        self._raw = mne.io.read_raw(self.filepath, preload=True)
        self._data, self._times = self._raw[:, :]

    def read_chunk(self, n_samples: int) -> Optional[np.ndarray]:
        raise NotImplementedError("FileSource uses get_data(), not read_chunk()")

    def seek(self, sample_idx: int) -> None:
        pass

    def close(self) -> None:
        self._raw = None
        self._data = np.empty((0, 0))
        self._times = np.empty(0)

    @property
    def sfreq(self) -> float:
        return float(self._raw.info['sfreq'])

    @property
    def n_channels(self) -> int:
        return int(self._data.shape[0])

    @property
    def total_samples(self) -> int:
        return int(self._data.shape[1])

    def get_data(self, start: Optional[int] = None,
                 stop: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        if self._data is None or self._times is None or self._data.size == 0:
            return np.array([]), np.array([])
        if start is None:
            start = 0
        if stop is None:
            stop = self._data.shape[1]
        return self._data[:, start:stop], self._times[start:stop]
