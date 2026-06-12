"""MNE-backed EEG readers with format registry."""
from __future__ import annotations

from pathlib import Path

import mne

from bci.source.base import EEGReader, register_reader


class _MNEReader(EEGReader):
    """Base for MNE-supported formats."""

    _mne_reader: str = ''

    def read(self, filepath: Path) -> mne.io.Raw:
        reader = getattr(mne.io, self._mne_reader)
        raw = reader(str(filepath), preload=True, verbose=False)
        return raw


@register_reader('.edf', '.bdf')
class EDFReader(_MNEReader):
    _mne_reader = 'read_raw_edf'


@register_reader('.fif', '.fif.gz')
class FIFFReader(_MNEReader):
    _mne_reader = 'read_raw_fif'


@register_reader('.set')
class EEGLABReader(_MNEReader):
    _mne_reader = 'read_raw_eeglab'


@register_reader('.vhdr')
class BrainVisionReader(_MNEReader):
    _mne_reader = 'read_raw_brainvision'
