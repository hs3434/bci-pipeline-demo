"""FileSource — static facade for loading EEG files."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

import mne

from bci.source.base import get_reader


class FileSource:
    """Static facade for loading EEG files.

    Provides ``load()`` which returns an MNE Raw object.
    """

    @staticmethod
    def load(filepath: Path | str | List[str]):
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
        result._source_path = str(paths[0])
        return result


def find_session_runs(filepath: Path) -> List[Path]:
    """Discover all runs belonging to the same subject session.

    Given ``S001R04.edf``, glob for ``S001R*.edf`` in the same
    directory and sort by ascending run number.
    """
    import glob as glob_lib

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
