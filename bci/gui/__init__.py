"""
GUI Module
==========
Qt-based BCI Signal Viewer with dual-mode analysis.
"""
from __future__ import annotations

from .main_window import BCIMainWindow, main
from .batch_tab import BatchTab
from .stream_tab import StreamTab
from .worker import BatchWorker, StreamWorker

__all__ = [
    'BCIMainWindow',
    'main',
    'BatchTab',
    'StreamTab',
    'BatchWorker',
    'StreamWorker',
]
