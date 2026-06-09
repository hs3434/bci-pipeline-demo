"""
DataSource Abstract Base Class
==============================
Unified interface for file-based and streaming data sources.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class DataSource(ABC):
    """Abstract data source for EEG signals.

    Subclasses implement either batch (FileSource) or streaming
    (StreamSource) access patterns. Consumers only depend on this
    interface, not on the concrete source type.
    """

    @abstractmethod
    def open(self) -> None:
        """Open the data source and prepare for reading."""
        ...

    @abstractmethod
    def read_chunk(self, n_samples: int) -> Optional[np.ndarray]:
        """Read up to n_samples from current position.

        Returns:
            (n_channels, n_read) array, or None at EOF / after close.
        """
        ...

    @abstractmethod
    def seek(self, sample_idx: int) -> None:
        """Move read position to sample_idx."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources."""
        ...

    @property
    @abstractmethod
    def sfreq(self) -> float:
        """Sampling frequency in Hz."""
        ...

    @property
    @abstractmethod
    def n_channels(self) -> int:
        """Number of EEG channels."""
        ...

    @property
    def total_samples(self) -> Optional[int]:
        """Total samples available, None if unknown (live stream)."""
        return None

    @property
    def is_stream(self) -> bool:
        """True for streaming sources, False for batch."""
        return False

    def get_data(self, start: Optional[int] = None,
                 stop: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Convenience: get data range with times. Default: all data."""
        raise NotImplementedError
