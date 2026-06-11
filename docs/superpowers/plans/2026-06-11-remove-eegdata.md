# Remove EEGData — Replace with MNE Raw

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the `EEGData` dataclass and use `mne.io.Raw` as the universal data container throughout the codebase.

**Architecture:** All readers return `mne.io.Raw` directly (they already create one internally). `FileSource.load()` returns Raw; `load_raw()` is removed as redundant. `_concat_eegs()` is replaced by `mne.concatenate_raws()`. `StreamSource` accepts Raw and extracts what it needs. `source_path` is stored as a custom attribute on Raw objects (`raw._source_path`). `EEGReader.read()` returns Raw; `read_raw()` is removed.

**Tech Stack:** MNE-Python, PyQt6, pytest

---

## Property Mapping Reference

| EEGData | MNE Raw |
|---------|---------|
| `.data` | `.get_data()` |
| `.sfreq` | `.info['sfreq']` |
| `.ch_names` | `.ch_names` |
| `.n_channels` | `.info['nchan']` or `len(raw.ch_names)` |
| `.n_samples` / `.total_samples` | `.n_times` |
| `.duration` | `.n_times / .info['sfreq']` |
| `.source_path` | `._source_path` (custom attr) |
| `.montage` | `.get_montage()` |

---

### Task 1: Refactor `base.py` — Remove EEGData, simplify EEGReader

**Files:**
- Modify: `bci/source/base.py`

- [ ] **Step 1: Rewrite base.py**

Replace the entire file content:

```python
"""
EEGReader ABC + Reader Registry
================================
Reader abstraction and pluggable reader registry
for file-format-agnostic EEG loading.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Type


class EEGReader(ABC):
    """Abstract reader for a specific EEG file format.

    Subclasses register themselves via ``@register_reader(suffix)``,
    then implement ``read()`` to return an MNE Raw object.
    """

    @abstractmethod
    def read(self, filepath: Path):
        """Load file and return an MNE Raw object."""
        ...


_reader_registry: Dict[str, EEGReader] = {}


def register_reader(*suffixes: str):
    """Decorator: register an EEGReader subclass under one or more suffixes."""
    def wrapper(cls: type[EEGReader]) -> type[EEGReader]:
        instance = cls()
        for suffix in suffixes:
            _reader_registry[suffix.lower()] = instance
        return cls
    return wrapper


def get_reader(filepath: Path) -> EEGReader:
    """Resolve the appropriate reader for a file path."""
    suffix = filepath.suffix.lower()
    reader = _reader_registry.get(suffix)
    if reader is None:
        supported = sorted(_reader_registry.keys())
        raise ValueError(
            f"Unsupported file format '{suffix}'. "
            f"Supported: {', '.join(supported)}"
        )
    return reader
```

- [ ] **Step 2: Verify no import errors**

Run: `uv run python -c "from bci.source.base import EEGReader, register_reader, get_reader; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add bci/source/base.py
git commit -m "refactor: remove EEGData from base.py, simplify EEGReader"
```

---

### Task 2: Refactor `readers.py` — Return Raw directly

**Files:**
- Modify: `bci/source/readers.py`

- [ ] **Step 1: Rewrite readers.py**

Replace the entire file content:

```python
"""MNE-backed EEG readers with format registry."""
from __future__ import annotations

from pathlib import Path

import mne

from bci.source.base import EEGReader, register_reader


class _MNEReader(EEGReader):
    """Base for MNE-supported formats."""

    _mne_reader: str = ''

    def read(self, filepath: Path):
        reader = getattr(mne.io, self._mne_reader)
        raw = reader(str(filepath), preload=True, verbose=False)
        raw._source_path = str(filepath)
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
```

- [ ] **Step 2: Verify no import errors**

Run: `uv run python -c "from bci.source.readers import EDFReader; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add bci/source/readers.py
git commit -m "refactor: readers return Raw directly, remove read_raw override"
```

---

### Task 3: Refactor `file_source.py` — load() returns Raw, remove load_raw and _concat_eegs

**Files:**
- Modify: `bci/source/file_source.py`

- [ ] **Step 1: Rewrite file_source.py**

Replace the entire file content:

```python
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
    Supports single-file and multi-file (session) loading.
    """

    @staticmethod
    def load(filepath: Path | str | List[str]):
        """Load EEG file(s) and return an MNE Raw object.

        Parameters
        ----------
        filepath : Path | str | list[Path | str]
            A single file path or a list of paths belonging to
            the same session.  When a list is given the files
            are concatenated along the time axis.
        """
        if isinstance(filepath, list):
            paths = [Path(p) for p in filepath]
        else:
            p = Path(filepath)
            runs = find_session_runs(p)
            paths = runs if len(runs) > 1 else [p]

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
    stem = filepath.stem
    match = re.match(r'^(.*R)0?(\d+)$', stem)
    if match is None:
        return [filepath]

    base = match.group(1)
    ext = filepath.suffix
    parent = filepath.parent
    runs = sorted(parent.glob(f'{base}*{ext}'))
    return runs if runs else [filepath]
```

- [ ] **Step 2: Verify no import errors**

Run: `uv run python -c "from bci.source import FileSource; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add bci/source/file_source.py
git commit -m "refactor: FileSource.load() returns Raw, remove load_raw and _concat_eegs"
```

---

### Task 4: Update `source/__init__.py` — Remove EEGData export

**Files:**
- Modify: `bci/source/__init__.py`

- [ ] **Step 1: Rewrite __init__.py**

Replace the entire file content:

```python
"""bci.source — EEG data loading abstractions.

Provides reader abstractions (EEGReader, register_reader)
and concrete sources (FileSource, StreamSource).
"""
from .base import EEGReader, register_reader, get_reader
from .file_source import FileSource
from .stream_source import StreamSource

# Trigger reader registration
from . import readers  # noqa: F401

__all__ = [
    'EEGReader',
    'FileSource',
    'StreamSource',
    'register_reader',
    'get_reader',
]
```

- [ ] **Step 2: Verify no import errors**

Run: `uv run python -c "from bci.source import FileSource, StreamSource; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add bci/source/__init__.py
git commit -m "refactor: remove EEGData from source package exports"
```

---

### Task 5: Refactor `stream_source.py` — Accept Raw instead of EEGData

**Files:**
- Modify: `bci/source/stream_source.py`

- [ ] **Step 1: Rewrite stream_source.py**

Replace the entire file content:

```python
"""StreamSource — chunk-by-chunk streaming over an MNE Raw object.

Wraps an MNE Raw object and feeds it chunk-by-chunk, simulating
real-time EEG acquisition for online/streaming display.
"""
from __future__ import annotations

import numpy as np


class StreamSource:
    """Streaming wrapper over a pre-loaded MNE Raw object.

    Does NOT read files — receives a ready-to-use Raw from
    FileSource.load() or similar.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data to stream.
    chunk_duration : float
        Duration of each chunk in seconds (default 0.1 s).
    """

    def __init__(self, raw, chunk_duration: float = 0.1):
        self._raw = raw
        self._data = raw.get_data()
        self.sfreq = raw.info['sfreq']
        self.n_channels = raw.info['nchan']
        self.chunk_duration = chunk_duration
        self._chunk_samples = int(self.sfreq * chunk_duration)
        self._position = 0

    @property
    def ch_names(self):
        return self._raw.ch_names

    @property
    def source_path(self):
        return getattr(self._raw, '_source_path', None)

    @property
    def position(self) -> int:
        return self._position

    @property
    def total_samples(self) -> int:
        return self._raw.n_times

    @property
    def duration(self) -> float:
        return self._raw.n_times / self.sfreq if self.sfreq > 0 else 0.0

    def read_chunk(self) -> np.ndarray | None:
        """Read the next chunk of data.

        Returns
        -------
        np.ndarray or None
            Shape (n_channels, chunk_samples), or None if at end.
        """
        if self._position >= self._data.shape[1]:
            return None

        end = min(self._position + self._chunk_samples, self._data.shape[1])
        chunk = self._data[:, self._position:end]
        self._position = end
        return chunk

    def reset(self):
        """Reset the read position to the start."""
        self._position = 0
```

- [ ] **Step 2: Verify no import errors**

Run: `uv run python -c "from bci.source import StreamSource; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add bci/source/stream_source.py
git commit -m "refactor: StreamSource accepts Raw instead of EEGData"
```

---

### Task 6: Update `pipeline/__init__.py` — Use FileSource.load() instead of load_raw()

**Files:**
- Modify: `bci/pipeline/__init__.py:103`

- [ ] **Step 1: Replace load_raw with load**

In `BCIPipeline.load()`, change line 103:

```python
raw_data = FileSource.load(filepath)
```

(Pipeline already stores Raw; `FileSource.load()` now returns Raw directly.)

- [ ] **Step 2: Run pipeline smoke test**

Run: `uv run python -c "from bci.pipeline import BCIPipeline; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add bci/pipeline/__init__.py
git commit -m "refactor: pipeline uses FileSource.load() (now returns Raw)"
```

---

### Task 7: Update `worker.py` — Remove EEGData references

**Files:**
- Modify: `bci/gui/worker.py`

- [ ] **Step 1: Update imports**

Remove `EEGData` from the import on line 21:

```python
from bci.source import FileSource, StreamSource
```

- [ ] **Step 2: Update LoadWorker docstring**

Change line 65 from `"emits the loaded EEGData object"` to `"emits the loaded MNE Raw object"`.

- [ ] **Step 3: Update StreamWorker docstring**

Change line 130 from `"Wraps an EEGData in a StreamSource"` to `"Wraps an MNE Raw in a StreamSource"`.

- [ ] **Step 4: Update StreamWorker isinstance check**

Replace lines 148-153:

```python
            if isinstance(source, StreamSource):
                self.source = source
            else:
                self.source = StreamSource(source, chunk_duration)
```

(Previously checked for `EEGData`; now accepts any object StreamSource can wrap, i.e. Raw.)

- [ ] **Step 5: Verify no import errors**

Run: `uv run python -c "from bci.gui.worker import LoadWorker, BatchWorker, StreamWorker; print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add bci/gui/worker.py
git commit -m "refactor: remove EEGData from worker.py"
```

---

### Task 8: Update `batch_tab.py` — Access Raw properties

**Files:**
- Modify: `bci/gui/batch_tab.py`

- [ ] **Step 1: Update `_on_load_finished` property accesses**

Find the method `_on_load_finished` and update property accesses using the mapping:

- `source.n_channels` → `source.info['nchan']`
- `source.total_samples` → `source.n_times`
- `source.sfreq` → `source.info['sfreq']`
- `source.data` → `source.get_data()`
- `source.ch_names` → `source.ch_names` (unchanged)

Read the file first to see exact lines, then make the edits.

- [ ] **Step 2: Verify no import errors**

Run: `uv run python -c "from bci.gui.batch_tab import BatchTab; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add bci/gui/batch_tab.py
git commit -m "refactor: batch_tab uses Raw property accessors"
```

---

### Task 9: Update `stream_tab.py` — Wrap Raw in StreamSource

**Files:**
- Modify: `bci/gui/stream_tab.py`

- [ ] **Step 1: Update `_on_load_finished`**

Find the method `_on_load_finished`. The `StreamSource(eeg)` call already works since StreamSource now accepts Raw. Update any direct property accesses on `self._stream_source` (these go through StreamSource which already exposes the right properties, so may need no changes). Check for any `eeg.sfreq`, `eeg.n_channels` etc. that bypass StreamSource.

Read the file first to see exact lines, then make the edits.

- [ ] **Step 2: Verify no import errors**

Run: `uv run python -c "from bci.gui.stream_tab import StreamTab; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add bci/gui/stream_tab.py
git commit -m "refactor: stream_tab uses Raw via StreamSource"
```

---

### Task 10: Update `info_panel.py` — Access source_path from Raw

**Files:**
- Modify: `bci/gui/widgets/info_panel.py`

- [ ] **Step 1: Update `_display_name` and property accesses**

The `source_path` is now on `raw._source_path` (accessed via `StreamSource.source_path` which already delegates to it). Check `_display_name` helper on line ~169 and any property accesses like `.n_channels`, `.total_samples`, `.sfreq` on source objects.

Read the file first to see exact lines, then make the edits.

- [ ] **Step 2: Verify no import errors**

Run: `uv run python -c "from bci.gui.widgets.info_panel import EEGInfoPanel; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add bci/gui/widgets/info_panel.py
git commit -m "refactor: info_panel uses Raw property accessors"
```

---

### Task 11: Rewrite `test_source.py` — Remove EEGData tests

**Files:**
- Modify: `bci/tests/test_source.py`

- [ ] **Step 1: Rewrite test_source.py**

Remove `TestEEGData` class entirely. Update all other tests:

- `TestEEGReaderABC`: `read()` returns Raw, remove `read_raw()` tests
- `TestReaderRegistry`: `DummyReader.read()` returns Raw
- `TestFileSource`: `FileSource.load()` returns Raw; check `.info['sfreq']`, `.ch_names`, `.n_times`, `._source_path`
- `TestStreamSource`: construct with `mne.io.RawArray` instead of `EEGData`; check `.source_path` via `._source_path`
- `TestFindSessionRuns`: unchanged

Read the file first, then make the edits.

- [ ] **Step 2: Run tests**

Run: `uv run pytest bci/tests/test_source.py -v`
Expected: all pass

- [ ] **Step 3: Commit**

```bash
git add bci/tests/test_source.py
git commit -m "refactor: rewrite test_source.py for Raw-based API"
```

---

### Task 12: Update `test_worker.py` — Replace EEGData with Raw

**Files:**
- Modify: `bci/tests/test_worker.py`

- [ ] **Step 1: Update test_worker.py**

- Remove `from bci.source import EEGData` import
- Change `isinstance(results[0], EEGData)` to `isinstance(results[0], mne.io.RawArray)`
- Update property accesses: `.n_channels` → `.info['nchan']`, `.total_samples` → `.n_times`
- `FileSource.load()` now returns Raw, so `StreamWorker(eeg)` already accepts it

Read the file first, then make the edits.

- [ ] **Step 2: Run tests**

Run: `uv run pytest bci/tests/test_worker.py -v`
Expected: all pass

- [ ] **Step 3: Commit**

```bash
git add bci/tests/test_worker.py
git commit -m "refactor: update test_worker.py for Raw-based API"
```

---

### Task 13: Update `test_stream_worker_sw.py` — Replace EEGData with Raw

**Files:**
- Modify: `bci/tests/test_stream_worker_sw.py`

- [ ] **Step 1: Replace _mock_eeg()**

Replace the helper to create an MNE RawArray:

```python
def _mock_eeg(sfreq=256.0, n_channels=3, duration=1.0):
    import mne
    import numpy as np
    n_samples = int(sfreq * duration)
    data = np.zeros((n_channels, n_samples))
    ch_names = [f'EEG{i}' for i in range(n_channels)]
    info = mne.create_info(ch_names, sfreq, ch_types='eeg')
    return mne.io.RawArray(data, info, verbose=False)
```

Remove the `from bci.source.base import EEGData` import.

- [ ] **Step 2: Run tests**

Run: `uv run pytest bci/tests/test_stream_worker_sw.py -v`
Expected: all pass

- [ ] **Step 3: Commit**

```bash
git add bci/tests/test_stream_worker_sw.py
git commit -m "refactor: update test_stream_worker_sw.py for Raw-based API"
```

---

### Task 14: Update `test_session.py` — Replace EEGData property access

**Files:**
- Modify: `bci/tests/test_session.py`

- [ ] **Step 1: Update property accesses**

Replace throughout:
- `eeg.total_samples` → `eeg.n_times`
- `eeg.n_channels` → `eeg.info['nchan']`
- `eeg.sfreq` → `eeg.info['sfreq']`

`FileSource.load()` already returns Raw, so no type change needed.

Read the file first, then make the edits.

- [ ] **Step 2: Run tests**

Run: `uv run pytest bci/tests/test_session.py -v`
Expected: all pass

- [ ] **Step 3: Commit**

```bash
git add bci/tests/test_session.py
git commit -m "refactor: update test_session.py for Raw property accessors"
```

---

### Task 15: Update `test_tabs.py` — Check for EEGData references

**Files:**
- Modify: `bci/tests/test_tabs.py` (if needed)

- [ ] **Step 1: Search for EEGData references**

Run: `grep -n "EEGData\|\.n_channels\|\.total_samples\|\.sfreq" bci/tests/test_tabs.py`

If found, update property accesses using the mapping. If not found, skip.

- [ ] **Step 2: Run tests**

Run: `uv run pytest bci/tests/test_tabs.py -v`
Expected: all pass (some may be skipped due to CI Abort issue)

- [ ] **Step 3: Commit** (if changed)

```bash
git add bci/tests/test_tabs.py
git commit -m "refactor: update test_tabs.py for Raw property accessors"
```

---

### Task 16: Final verification — Full test suite

**Files:** None

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest bci/tests/ -v`
Expected: all pass (except known CI-skip marks)

- [ ] **Step 2: Verify no remaining EEGData references**

Run: `grep -rn "EEGData" bci/`
Expected: no matches (except possibly in comments/docstrings if missed)

- [ ] **Step 3: Final commit if any cleanup needed**

```bash
git add -A
git commit -m "refactor: final cleanup — remove all EEGData references"
```
