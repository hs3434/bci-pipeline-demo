"""Shared type definitions for EEG data sources."""
from __future__ import annotations
from typing import Protocol, List, Optional, runtime_checkable

import numpy as np


@runtime_checkable
class EEGSource(Protocol):
    """Structural interface satisfied by mne.io.Raw and StreamSource.

    Both types expose the attributes needed by GUI consumers
    (info panel, waveform plot, channel-name display). Runtime check
    is opt-in via isinstance(); static type checkers use it
    structurally.
    """
    @property
    def ch_names(self) -> List[str]: ...
    @property
    def sfreq(self) -> float: ...
    @property
    def n_channels(self) -> int: ...
    @property
    def n_times(self) -> int: ...
    @property
    def filepath(self) -> Optional[str]: ...
    def get_data(self) -> np.ndarray: ...
