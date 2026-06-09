"""
FileSource — File Loading Facade
=================================
Format-agnostic EEG file loading with optional session concatenation.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import logging
import re
import glob as glob_lib

import numpy as np

from bci.source.base import EEGData, get_reader

logger = logging.getLogger(__name__)


class FileSource:
    """File-loading facade for EEG data.

    Delegates to registered EEGReader implementations based on file
    suffix. Supports single-file loading and session-level multi-run
    concatenation.

    Examples:
        # Single file
        >>> eeg = FileSource.load('data.edf')
        >>> print(eeg.n_channels, eeg.n_samples)

        # Multi-run session
        >>> eeg = FileSource.load('S001R04.edf', session=True)
    """

    @staticmethod
    def load(filepath: Path | str | List[str],
             session: bool = False) -> EEGData:
        if isinstance(filepath, list):
            paths = [Path(p) for p in filepath]
        else:
            filepath = Path(filepath)
            paths = find_session_runs(filepath) if session else [filepath]

        if not paths:
            raise FileNotFoundError(f"No files found for: {filepath}")

        eegs: List[EEGData] = []
        for p in paths:
            logger.info(f"Loading: {p}")
            reader = get_reader(p)
            eegs.append(reader.read(p))

        if len(eegs) == 1:
            eegs[0].source_path = str(paths[0])
            return eegs[0]

        result = _concat_eegs(eegs)
        result.source_path = str(paths[0])
        return result

    @staticmethod
    def load_raw(filepath: Path | str):
        """Load file and return an MNE Raw object.

        Useful when downstream code needs MNE-specific features
        (montage metadata, plot_topomap, etc.).
        """
        filepath = Path(filepath)
        reader = get_reader(filepath)
        return reader.read_raw(filepath)


def find_session_runs(filepath: Path) -> List[Path]:
    """Discover all runs belonging to the same subject session.

    Given ``S001R04.edf``, glob for ``S001R*.edf`` in the same
    directory and sort by ascending run number.
    """
    stem = filepath.stem
    match = re.match(r'^(.*R)0?(\d+)$', stem)
    if match is None:
        return [filepath]

    base = match.group(1)
    ext = filepath.suffix
    pattern = f"{filepath.parent}/{base}*{ext}"
    runs = sorted(
        glob_lib.glob(pattern),
        key=lambda p: int(re.search(r'R(\d+)', p).group(1)),
    )
    return [Path(p) for p in runs]


def _concat_eegs(eegs: List[EEGData]) -> EEGData:
    """Concatenate multiple EEGData objects along the time axis."""
    data = np.concatenate([e.data for e in eegs], axis=1)
    sfreq = eegs[0].sfreq
    ch_names = eegs[0].ch_names
    return EEGData(data=data, sfreq=sfreq, ch_names=ch_names)
