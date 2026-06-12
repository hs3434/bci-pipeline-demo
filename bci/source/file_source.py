"""FileSource — static facade for loading EEG files."""
from __future__ import annotations

from pathlib import Path
from typing import List

import mne

from bci.source.base import get_reader


class FileSource:
    """Static facade for loading EEG files.

    Provides ``load()`` which returns an MNE Raw object.
    """

    @staticmethod
    def load(filepath: Path | str | List[Path | str]) -> mne.io.Raw:
        """Load EEG file(s) and return an MNE Raw object.

        Parameters
        ----------
        filepath : Path | str | list[Path | str]
            A single file path or a list of paths.
            When a list is given the files are concatenated
            along the time axis.
        """
        if isinstance(filepath, list):
            paths = [Path(p) for p in filepath]
        else:
            paths = [Path(filepath)]

        raws = []
        for p in paths:
            reader = get_reader(p)
            raws.append(reader.read(p))

        if len(raws) == 1:
            return raws[0]

        result = mne.concatenate_raws(raws)
        return result
