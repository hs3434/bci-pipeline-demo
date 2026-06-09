"""
Session Source — Multi-Run Concatenation
======================================
Loads a session directory containing multiple runs from the same subject
(e.g. S001R04.edf, S001R06.edf, S001R08.edf, S001R10.edf) and
concatenates them into a single stream for batch or streaming use.
"""
from __future__ import annotations
from typing import Optional, List, Callable
from pathlib import Path
import numpy as np
import re
import glob as glob_lib


def find_session_runs(filepath: str | Path) -> List[Path]:
    """Find all runs for the same subject.

    Given a path like /data/bci/S001R04.edf, glob to find all matching
    runs (S001R04.edf, S001R06.edf, S001R08.edf, S001R10.edf) sorted
    by run number.
    """
    filepath = Path(filepath)
    parent = filepath.parent
    stem = filepath.stem  # e.g. "S001R04"
    match = re.match(r'^(.*R)0?(\d+)$', stem)
    if match is None:
        return [filepath]

    base = match.group(1)  # e.g. "S001R"
    ext = filepath.suffix  # e.g. ".edf"
    pattern = f"{parent}/{base}*{ext}"
    runs = sorted(glob_lib.glob(pattern), key=lambda p: int(re.search(r'R(\d+)', p).group(1)))
    return [Path(p) for p in runs]


class SessionSource:
    """Concatenates multiple EDF runs from the same subject.

    Implements the DataSource-like interface so it can be used as a
    drop-in for FileSource or StreamSource in batch/stream modes.
    """

    def __init__(self, filepath: str | Path | List[str] | List[Path], chunk_duration: float = 0.1):
        if isinstance(filepath, (list, tuple)):
            self._runs: List[Path] = [Path(p) for p in filepath]
            self.filepath = self._runs[0] if self._runs else Path(".")
        else:
            self._runs: List[Path] = []
            self.filepath = Path(filepath)
        self.chunk_duration = chunk_duration
        self._raws: List = []  # mne.io.Raw objects
        self._data_list: List[np.ndarray] = []
        self._position = 0
        self._total_samples = 0
        self._loop = False
        self._speed = 1.0
        self._closed = False

    def open(self, progress_callback: Optional[Callable[[int, int], None]] = None) -> None:
        import mne

        if not self._runs:
            self._runs = find_session_runs(self.filepath)
        total_runs = len(self._runs)
        self._raws = []
        self._data_list = []
        self._total_samples = 0

        for i, run_path in enumerate(self._runs):
            raw = mne.io.read_raw(run_path, preload=False, verbose=False)
            self._raws.append(raw)
            data = raw.get_data()
            self._data_list.append(data)
            self._total_samples += data.shape[1]
            if progress_callback is not None:
                progress_callback(i + 1, total_runs)

        self._position = 0
        self._closed = False

    def read_chunk(self, n_samples: int) -> Optional[np.ndarray]:
        if self._closed or not self._data_list:
            return None

        if self._position >= self._total_samples:
            if self._loop:
                self._position = 0
            else:
                return None

        result_ch = self._data_list[0].shape[0]
        result = np.zeros((result_ch, 0))
        remaining = n_samples

        while remaining > 0:
            run_idx = 0
            offset = 0
            for i, data in enumerate(self._data_list):
                if self._position < offset + data.shape[1]:
                    run_idx = i
                    break
                offset += data.shape[1]
            else:
                if self._loop:
                    self._position = 0
                    continue
                else:
                    return result if result.shape[1] > 0 else None

            data = self._data_list[run_idx]
            local_pos = self._position - offset
            chunk_len = min(remaining, data.shape[1] - local_pos)
            chunk = data[:, local_pos:local_pos + chunk_len]
            result = np.concatenate([result, chunk], axis=1)
            self._position += chunk_len
            remaining -= chunk_len

            if self._position >= self._total_samples:
                if self._loop:
                    self._position = 0
                else:
                    self._data_list = []
                    break

        return result

    def seek(self, sample_idx: int) -> None:
        self._position = max(0, min(sample_idx, self._total_samples))

    def close(self) -> None:
        self._raws = []
        self._data_list = []
        self._closed = True

    def reset(self) -> None:
        self._position = 0

    def set_loop(self, enabled: bool) -> None:
        self._loop = enabled

    def set_speed(self, speed: float) -> None:
        self._speed = max(0.01, speed)

    @property
    def sfreq(self) -> float:
        if self._raws:
            return float(self._raws[0].info['sfreq'])
        return 256.0

    @property
    def n_channels(self) -> int:
        if self._data_list:
            return self._data_list[0].shape[0]
        return 0

    @property
    def total_samples(self) -> int:
        return self._total_samples

    @property
    def is_stream(self) -> bool:
        return True

    @property
    def position(self) -> int:
        return self._position

    @property
    def progress(self) -> int:
        if self._total_samples == 0:
            return 0
        return min(100, int(self._position / self._total_samples * 100))

    @property
    def run_count(self) -> int:
        return len(self._runs)