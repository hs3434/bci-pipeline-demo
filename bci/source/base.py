"""
EEGReader ABC + Reader Registry
================================
Reader abstraction and pluggable reader registry
for file-format-agnostic EEG loading.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Type


class EEGReader(ABC):
    """Abstract reader for a specific EEG file format.

    Subclasses register themselves via ``@register_reader(suffix)``,
    then implement ``read()`` to return an MNE Raw object.
    """

    @abstractmethod
    def read(self, filepath: Path) -> 'mne.io.Raw':
        """Load file and return an MNE Raw object."""
        ...


_reader_registry: Dict[str, EEGReader] = {}


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
