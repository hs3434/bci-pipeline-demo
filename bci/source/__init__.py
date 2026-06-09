"""
Data Source Module
==================
EEG data loading (FileSource), streaming (StreamSource), and reader
abstractions (EEGData, EEGReader, register_reader).
"""
from .base import EEGData, EEGReader, register_reader, get_reader
from .file_source import FileSource
from .stream_source import StreamSource
from . import readers  # trigger built-in reader registrations

__all__ = [
    'EEGData',
    'EEGReader',
    'register_reader',
    'get_reader',
    'FileSource',
    'StreamSource',
]
