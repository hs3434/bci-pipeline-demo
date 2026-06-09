"""
EEGData Container + EEGReader ABC + Reader Registry
====================================================
Data container, reader abstraction, and pluggable reader registry
for file-format-agnostic EEG loading.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Type

import numpy as np


@dataclass
class EEGData:
    """Portable EEG data container — format-agnostic.

    Attributes:
        data: Raw signal of shape (n_channels, n_samples).
        sfreq: Sampling frequency in Hz.
        ch_names: List of channel names.
        montage: Montage name, or None if unknown.
        source_path: Original file path, or None.
    """
    data: np.ndarray
    sfreq: float
    ch_names: list[str]
    montage: str | None = None
    source_path: str | None = None

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    @property
    def total_samples(self) -> int:
        return self.n_samples

    @property
    def duration(self) -> float:
        return self.n_samples / self.sfreq if self.sfreq > 0 else 0.0


class EEGReader(ABC):
    """Abstract reader for a specific EEG file format.

    Subclasses register themselves via ``@register_reader(suffix)``,
    then implement ``read()`` to extract an EEGData container and
    optionally ``read_raw()`` to return a MNE Raw object.
    """

    @abstractmethod
    def read(self, filepath: Path) -> EEGData:
        """Load file and return a portable EEGData container."""
        ...

    def read_raw(self, filepath: Path):
        """Load file and return an MNE Raw object (optional, GUI / pipeline use)."""
        return _eegdata_to_raw(self.read(filepath))


# ---------------------------------------------------------------------------
# Reader registry
# ---------------------------------------------------------------------------

_reader_registry: Dict[str, EEGReader] = {}


def _eegdata_to_raw(eeg: EEGData):
    """Convert EEGData to an MNE RawArray (lazy-import MNE)."""
    import mne
    info = mne.create_info(eeg.ch_names, eeg.sfreq, ch_types='eeg')
    return mne.io.RawArray(eeg.data, info)


def register_reader(*suffixes: str):
    """Decorator: register an EEGReader subclass under one or more suffixes."""
    def wrapper(cls: type[EEGReader]) -> type[EEGReader]:
        instance = cls()
        for suffix in suffixes:
            _reader_registry[suffix.lower()] = instance
        return cls
    return wrapper


def get_reader(filepath: Path) -> EEGReader:
    """Resolve the appropriate reader for a file path."""
    suffix = filepath.suffix.lower()
    reader = _reader_registry.get(suffix)
    if reader is None:
        supported = sorted(_reader_registry.keys())
        raise ValueError(
            f"Unsupported file format '{suffix}'. "
            f"Supported: {', '.join(supported)}"
        )
    return reader
