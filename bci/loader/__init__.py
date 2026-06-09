"""
Data Loader Module
==================
EEG Data Loading - MNE/EEGLAB/BrainVision support
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

import logging
from dataclasses import dataclass

if TYPE_CHECKING:
    import mne
    import numpy as np
    from bci.config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class LoaderResult:
    """Result of data loading"""
    filepath: Path
    format: str
    n_channels: int
    duration: float
    sfreq: float


class DataLoader:
    """EEG Data Loader - supports multiple formats

    Supported formats: EDF (.edf), FIF (.fif), EEGLAB (.set), BrainVision (.vhdr)

    Examples:
        >>> from bci.loader import DataLoader
        >>> loader = DataLoader()
        >>> raw = loader.load('data.edf')
        >>> print(f"Loaded {len(raw.ch_names)} channels")
        Loaded 64 channels

        >>> info = loader.get_info()
        >>> print(f"Duration: {info['duration']:.1f}s, SFreq: {info['sfreq']}Hz")
        Duration: 120.5s, SFreq: 256.0Hz

        >>> data, times = loader.get_data(start=0, stop=1000)
        >>> print(f"Data shape: {data.shape}")
        Data shape: (64, 1000)
    """

    SUPPORTED_FORMATS = {
        '.edf': 'edf',
        '.fif': 'fif',
        '.set': 'eeglab',
        '.vhdr': 'brainvision',
        '.fdt': 'eeglab',
    }

    def __init__(self, config: Optional['PipelineConfig'] = None):
        self.config = config
        self.raw: Optional['mne.io.Raw'] = None

    def load(self, filepath: Path | str, preload: bool = True) -> 'mne.io.Raw':
        import mne

        filepath = Path(filepath)
        suffix = filepath.suffix.lower()

        logger.info(f"Loading {suffix} file: {filepath}")

        raw: mne.io.Raw
        if suffix == '.edf' or suffix == '.EDF':
            raw = mne.io.read_raw_edf(filepath, preload=preload)  # type: ignore
        elif suffix == '.fif':
            raw = mne.io.read_raw_fif(filepath, preload=preload)  # type: ignore
        elif suffix == '.set':
            raw = mne.io.read_raw_eeglab(filepath, preload=preload)  # type: ignore
        elif suffix == '.vhdr':
            raw = mne.io.read_raw_brainvision(filepath, preload=preload)  # type: ignore
        else:
            raise ValueError(f"Unsupported format: {suffix}")

        self.raw = raw
        logger.info(f"Loaded {len(self.raw.ch_names)} channels, "
                   f"{self.raw.n_times/self.raw.info['sfreq']:.1f}s, "
                   f"{self.raw.info['sfreq']} Hz")

        return self.raw

    def get_info(self) -> dict:
        """Get raw info as dict"""
        if self.raw is None:
            raise RuntimeError("No data loaded, call load() first")
        return {
            'n_channels': len(self.raw.ch_names),
            'sfreq': self.raw.info['sfreq'],
            'duration': self.raw.n_times / self.raw.info['sfreq'],
            'channels': self.raw.ch_names,
            'lowpass': self.raw.info.get('lowpass'),
            'highpass': self.raw.info.get('highpass'),
        }

    def get_data(self, start: Optional[int] = None, stop: Optional[int] = None,
                 picks: str = 'eeg') -> Tuple[np.ndarray, np.ndarray]:
        """Get data and times

        Args:
            start: Start sample (None = 0)
            stop: Stop sample
            picks: Channel selection

        Returns:
            (data, times) tuple
        """
        if self.raw is None:
            raise RuntimeError("No data loaded")
        start_val: int = start if start is not None else 0
        data = self.raw.get_data(picks=picks, start=start_val, stop=stop)
        times = self.raw.times[start_val:stop]
        return data, times  # type: ignore


def load_raw(filepath: Path | str, **kwargs) -> 'mne.io.Raw':
    """Convenience function to load EEG data"""
    loader = DataLoader()
    return loader.load(filepath, **kwargs)