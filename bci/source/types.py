"""Shared type definitions for EEG data sources."""
from __future__ import annotations
from typing import Protocol, List, Optional

import numpy as np


class EEGSource(Protocol):
    """Structural interface satisfied by mne.io.Raw and StreamSource.

    Both types expose the attributes/info needed by GUI consumers
    (info panel, waveform plot, channel-name display). Used for static
    type checking only — the actual attributes live in different places
    on each concrete type (mne.io.Raw uses .info / .filenames, StreamSource
    exposes them as direct attrs), so runtime isinstance() is not
    meaningful here. Consumers use getattr() for defensive attribute
    access.
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
