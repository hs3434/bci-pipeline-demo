"""bci.source — EEG data loading abstractions.

Provides reader abstractions (EEGReader, register_reader)
and concrete sources (FileSource, StreamSource).
"""
from .base import EEGReader, register_reader, get_reader
from .file_source import FileSource
from .stream_source import StreamSource

# Trigger reader registration
from . import readers  # noqa: F401

__all__ = [
    'EEGReader',
    'FileSource',
    'StreamSource',
    'register_reader',
    'get_reader',
]
