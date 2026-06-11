"""StreamSource — chunk-by-chunk streaming over an MNE Raw object.

Wraps an MNE Raw object and feeds it chunk-by-chunk, simulating
real-time EEG acquisition for online/streaming display.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class StreamSource:
    """Streaming wrapper over a pre-loaded MNE Raw object.

    Does NOT read files — receives a ready-to-use Raw from
    FileSource.load() or any other producer.

    Example:
        >>> raw = FileSource.load('data.edf')
        >>> stream = StreamSource(raw, chunk_duration=0.1)
        >>> while (chunk := stream.read_chunk()) is not None:
        ...     process(chunk)
    """

    def __init__(self, raw, chunk_duration: float = 0.1,
                 source_path: str | None = None):
        self._raw = raw
        self._data = raw.get_data()
        self.sfreq = raw.info['sfreq']
        self.n_channels = raw.info['nchan']
        self.chunk_duration = chunk_duration
        self.source_path = source_path

        self._position = 0
        self._speed = 1.0
        self._loop = False
        self._closed = False

    @property
    def ch_names(self) -> list[str]:
        return self._raw.ch_names

    @property
    def chunk_samples(self) -> int:
        return max(1, int(self.sfreq * self.chunk_duration))

    def read_chunk(self, n_samples: int | None = None) -> Optional[np.ndarray]:
        """Read next chunk.

        Args:
            n_samples: Samples to read (defaults to chunk_samples).

        Returns:
            (n_channels, n_read) array, or None at EOF / after close.
        """
        if n_samples is None:
            n_samples = self.chunk_samples
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
        total = self._data.shape[1]
        self._position = max(0, min(sample_idx, total))

    def close(self) -> None:
        self._data = np.empty((0, 0))
        self._closed = True

    def reset(self) -> None:
        self._position = 0

    def set_loop(self, enabled: bool) -> None:
        self._loop = enabled

    def set_speed(self, speed: float) -> None:
        self._speed = max(0.01, speed)

    @property
    def total_samples(self) -> int:
        return self._data.shape[1]

    @property
    def position(self) -> int:
        return self._position

    @property
    def progress(self) -> int:
        if self._data.shape[1] == 0:
            return 0
        return min(100, int(self._position / self._data.shape[1] * 100))

    @property
    def is_stream(self) -> bool:
        return True
