"""
Built-in EEG File Format Readers
=================================
EDF, FIF, EEGLAB (.set), and BrainVision (.vhdr) support via MNE.
"""
from __future__ import annotations
from pathlib import Path

from bci.source.base import EEGData, EEGReader, register_reader


class _MNEReader(EEGReader):
    """Shared MNE-backed reader — subclasses set :attr:`_mne_reader`."""

    _mne_reader: str = ''

    def read(self, filepath: Path) -> EEGData:
        import mne
        raw = getattr(mne.io, self._mne_reader)(
            filepath, preload=True, verbose=False)
        data = raw.get_data()
        ch_names = [str(c) for c in raw.ch_names]
        sfreq = float(raw.info['sfreq'])
        return EEGData(data=data, sfreq=sfreq, ch_names=ch_names)

    def read_raw(self, filepath: Path):
        import mne
        return getattr(mne.io, self._mne_reader)(
            filepath, preload=True, verbose=False)


@register_reader('.edf')
class EDFReader(_MNEReader):
    _mne_reader = 'read_raw_edf'


@register_reader('.fif')
class FIFFReader(_MNEReader):
    _mne_reader = 'read_raw_fif'


@register_reader('.set')
class EEGLABReader(_MNEReader):
    _mne_reader = 'read_raw_eeglab'


@register_reader('.vhdr')
class BrainVisionReader(_MNEReader):
    _mne_reader = 'read_raw_brainvision'
