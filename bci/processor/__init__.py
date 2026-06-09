"""
Processor Module
================
Offline (batch) and online (streaming) signal processors.
"""
from .offline import OfflineProcessor
from .online import OnlineProcessor

__all__ = ['OfflineProcessor', 'OnlineProcessor']
