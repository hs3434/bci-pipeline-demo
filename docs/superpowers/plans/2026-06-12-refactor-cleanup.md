# Refactor Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair all type, naming, and lifecycle inconsistencies left by the previous `EEGData → mne.io.Raw` refactor across the GUI, source, decoder, and epocher modules.

**Architecture:** Pure refactor — no behavior change. Tighten `Optional[object]` / untyped annotations to `mne.io.Raw` / `StreamSource` / `BCIPipeline`; introduce an `EEGSource` Protocol for the GUI/info-panel boundary; unify naming (`source_path` → `filepath`, `refresh_chart(data=...)`); source the decoder method dropdown from `bci.decoder.list_methods()`; let `BaseWorker` own its QThread lifecycle via a new `cleanup()` method.

**Tech Stack:** Python 3.x, PyQt6, mne, pytest

**Spec:** `docs/superpowers/specs/2026-06-12-refactor-cleanup-design.md`

**Working directory:** `/work/bci-pipeline-demo`

---

## File Structure

### New files
- `bci/source/types.py` — `EEGSource` Protocol definition

### Modified files
- `bci/source/stream_source.py` — rename `source_path` → `filepath`, add `raw: 'mne.io.Raw'` type
- `bci/source/base.py` — add `-> 'mne.io.Raw'` return type
- `bci/source/readers.py` — add `-> 'mne.io.Raw'` return type
- `bci/gui/worker.py` — `BaseWorker.stop()` + `cleanup()`, type `source`, fix `__import__`
- `bci/gui/batch_tab.py` — type annotations, simplify `_stop_workers()`
- `bci/gui/stream_tab.py` — type annotations, simplify `_stop_workers()`, rename `eeg` → `source`
- `bci/gui/session_loader.py` — replace `__import__('re')` with `re`
- `bci/gui/widgets/info_panel.py` — type annotations, simplify `_display_name` filepath lookup
- `bci/gui/widgets/main_page.py` — type annotations, `List[str]`
- `bci/gui/widgets/preprocess_page.py` — rename `source` → `data` in `refresh_chart`
- `bci/gui/widgets/epoch_page.py` — rename `pipeline` → `data` in `refresh_chart`
- `bci/gui/widgets/decode_page.py` — dynamic decoder list from `list_methods()`
- `bci/epocher/__init__.py` — remove dead `RuntimeError` check
- `bci/config/__init__.py` — remove empty `TYPE_CHECKING: pass`
- `bci/decoder/deep.py` — rename `_EEGCNN.__init__` param `n_times` → `n_samples`, propagate
- `README.md` — fix `BatchWorker (QThread)` description
- `bci/tests/test_worker.py` — verify existing tests still pass (no changes expected)

---

## Task 1: Add `EEGSource` Protocol

**Files:**
- Create: `bci/source/types.py`

- [ ] **Step 1: Create the protocol file**

Write to `bci/source/types.py`:

```python
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
```

- [ ] **Step 2: Verify file imports cleanly**

Run: `python -c "from bci.source.types import EEGSource; print(EEGSource)"`
Expected: prints `<class 'bci.source.types.EEGSource'>` (or similar), no errors.

- [ ] **Step 3: Commit**

```bash
git add bci/source/types.py
git commit -m "feat: add EEGSource Protocol for GUI/source boundary"
```

---

## Task 2: Source-layer type and naming fixes

**Files:**
- Modify: `bci/source/stream_source.py:26-33,42,107`
- Modify: `bci/source/base.py:21`
- Modify: `bci/source/readers.py:16`

- [ ] **Step 1: Fix `stream_source.py` — add type, rename `source_path` → `filepath`**

Edit `bci/source/stream_source.py`:
- Line 26: change `def __init__(self, raw, chunk_duration: float = 0.1,` to `def __init__(self, raw: 'mne.io.Raw', chunk_duration: float = 0.1,`
- Line 27: change `source_path: str | None = None):` to `filepath: str | None = None):`
- Line 28: change `self._raw = raw` to `self._raw: 'mne.io.Raw' = raw`
- Line 33: change `self.source_path = source_path` to `self.filepath = filepath`
- Line 42: change `return self._raw.ch_names` (no change; property already exists)
- Line 107: add `filepath` property alias? — NO. Keep `filepath` as plain attribute (set in __init__). The `EEGSource` Protocol says `filepath` as a property, but StreamSource defines it as a plain attribute. **However**, the Protocol's `runtime_checkable` will fail `isinstance()` if `filepath` is an attribute, not a property. Solution: convert to a property that reads the same attribute.

Change line 33 + add property. After `self.filepath = filepath`, also add a property block. The full replacement block at lines 28-33:

```python
        self._raw: 'mne.io.Raw' = raw
        self._data = raw.get_data()
        self.sfreq = raw.info['sfreq']
        self.n_channels = raw.info['nchan']
        self.chunk_duration = chunk_duration
        self._filepath: Optional[str] = filepath

        self._position = 0
        self._speed = 1.0
        self._loop = False
        self._closed = False

    @property
    def filepath(self) -> Optional[str]:
        return self._filepath
```

Then delete the `source_path` line (line 33 in the original).

- [ ] **Step 2: Fix `base.py` — add return type to `read`**

Edit `bci/source/base.py` line 21:
- Change `def read(self, filepath: Path):` to `def read(self, filepath: Path) -> 'mne.io.Raw':`

- [ ] **Step 3: Fix `readers.py` — add return type to `read`**

Edit `bci/source/readers.py` line 16:
- Change `def read(self, filepath: Path):` to `def read(self, filepath: Path) -> 'mne.io.Raw':`

- [ ] **Step 4: Run source tests to verify no regression**

Run: `python -m pytest bci/tests/test_source.py bci/tests/test_stream_worker_sw.py -v --no-header`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add bci/source/stream_source.py bci/source/base.py bci/source/readers.py
git commit -m "refactor: source layer — add mne.io.Raw types, rename source_path to filepath"
```

---

## Task 3: Decoder — rename `n_times` → `n_samples` in `_EEGCNN`

**Files:**
- Modify: `bci/decoder/deep.py:18,28,61,66`

- [ ] **Step 1: Read the current state**

Read `bci/decoder/deep.py` lines 15-30 and 58-68 to confirm current state.

- [ ] **Step 2: Rename `_EEGCNN.__init__` parameter**

Edit `bci/decoder/deep.py`:
- Line 18: change `def __init__(self, n_channels: int, n_times: int, n_classes: int,` to `def __init__(self, n_channels: int, n_samples: int, n_classes: int,`
- Line 28: change `dummy = torch.zeros(1, 1, n_channels, n_times)` to `dummy = torch.zeros(1, 1, n_channels, n_samples)`

- [ ] **Step 3: Propagate to call sites (CNNDecoder.fit)**

Edit `bci/decoder/deep.py`:
- Line 61: change `self._input_shape = (n_channels, n_times)` to `self._input_shape = (n_channels, n_samples)`
- Line 66: change `self.model = _EEGCNN(n_channels, n_times, self._n_classes,` to `self.model = _EEGCNN(n_channels, n_samples, self._n_classes,`

Note: The local unpacking at line 60 (`n_epochs, n_channels, n_times = X.shape`) is **NOT** changed — `n_times` is the correct mne convention name for the time dimension in `(n_epochs, n_channels, n_times)`. We add a new local variable to bridge:

```python
n_epochs, n_channels, n_times = X.shape
self._input_shape = (n_channels, n_times)
...
self.model = _EEGCNN(n_channels, n_times, ...)
```

Becomes:

```python
n_epochs, n_channels, n_times = X.shape
self._input_shape = (n_channels, n_times)
...
self.model = _EEGCNN(n_channels, n_times, ...)
```

(unchanged — `n_times` is the local variable name we pass to `_EEGCNN`, which is fine since `_EEGCNN`'s parameter is now also called `n_samples` but they're just two names for the same value).

- [ ] **Step 4: Verify all `_EEGCNN(...)` call sites**

Run: `grep -n "_EEGCNN" bci/decoder/deep.py`
Expected: only one call site at line 66 (now passes `n_times` which equals the new `n_samples` param).

- [ ] **Step 5: Run decoder tests**

Run: `python -m pytest bci/tests/test_decoder.py bci/tests/test_transformer.py bci/tests/test_transformer_bert.py -v --no-header`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add bci/decoder/deep.py
git commit -m "refactor: rename _EEGCNN n_times to n_samples for mne convention"
```

---

## Task 4: DecodePage — dynamic decoder method list

**Files:**
- Modify: `bci/gui/widgets/decode_page.py:1-50`

- [ ] **Step 1: Read the current state**

Read `bci/gui/widgets/decode_page.py` lines 1-50 to understand imports and current method list setup.

- [ ] **Step 2: Add import and replace hardcoded list**

In `bci/gui/widgets/decode_page.py`:
- Add to the top-level imports: `from bci.decoder import list_methods`
- Find the line `self._method.addItems([...])` (around line 31) and replace the hardcoded list with:
  ```python
  self._method.addItems(list_methods())
  ```

- [ ] **Step 3: Verify the file still imports**

Run: `python -c "from bci.gui.widgets.decode_page import DecodePage; print('ok')"`
Expected: prints `ok` (or similar) with no ImportError.

- [ ] **Step 4: Run widget tests**

Run: `python -m pytest bci/tests/test_widgets.py -v --no-header`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add bci/gui/widgets/decode_page.py
git commit -m "fix: DecodePage method dropdown sourced from bci.decoder.list_methods()"
```

---

## Task 5: `BaseWorker.stop()` + `cleanup()` redesign

**Files:**
- Modify: `bci/gui/worker.py:24-57`

- [ ] **Step 1: Read the current state**

Read `bci/gui/worker.py` lines 1-60 to confirm current `BaseWorker` shape.

- [ ] **Step 2: Modify `BaseWorker`**

Edit `bci/gui/worker.py`:
- In `BaseWorker.__init__`, add `self._thread: Optional[QThread] = None` (after `super().__init__()`).
- In `start_in_thread`, after `self.finished.connect(self._thread.quit)`, add a new line: `self.finished.connect(self._thread.wait)`.
- After `start_in_thread`, add two new methods:
  ```python
  def stop(self) -> None:
      """Default no-op. Subclasses override to halt periodic work
      (e.g. StreamWorker stops its QTimer)."""
      pass

  def cleanup(self) -> None:
      """Stop the worker, quit its thread, and wait for it to finish.
      Safe to call multiple times. Safe to call even if never started.
      """
      self.stop()
      if self._thread is not None and self._thread.isRunning():
          self._thread.quit()
          self._thread.wait()
      self._thread = None
  ```

The full new `BaseWorker` block (lines 24-57 in the original) becomes:

```python
class BaseWorker(QObject):
    """Abstract base for all background workers.

    Defines the common interface:
    - ``finished = pyqtSignal(object)``  — work done, carries result or None
    - ``error    = pyqtSignal(str)``     — work failed, carries message
    - ``run()``                         — abstract entry point (same name as QThread.run)
    - ``start_in_thread()``              — moveToThread + start, returns QThread
    - ``stop()``                         — halt periodic work; default no-op
    - ``cleanup()``                      — stop + quit + wait; safe to call multiple times
    """
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._thread: Optional[QThread] = None

    @abstractmethod
    def run(self):
        ...

    def start_in_thread(self, slot=None, parent=None) -> QThread:
        """Move this worker into a new QThread, connect signals, and start.

        Args:
            slot: The method to run when the thread starts.
                  Defaults to ``run`` if not specified.
                  Pass a different method to leverage moveToThread's
                  multi-entry-point advantage (e.g. StreamWorker.start).
            parent: Optional parent for the QThread.

        Returns the QThread so the caller can track / quit it.
        """
        self._thread = QThread(parent)
        self.moveToThread(self._thread)
        self._thread.started.connect(slot or self.run)
        self.finished.connect(self._thread.quit)
        self.finished.connect(self._thread.wait)
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        """Default no-op. Subclasses override to halt periodic work
        (e.g. StreamWorker stops its QTimer)."""
        pass

    def cleanup(self) -> None:
        """Stop the worker, quit its thread, and wait for it to finish.
        Safe to call multiple times. Safe to call even if never started.
        """
        self.stop()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
        self._thread = None
```

- [ ] **Step 3: Verify worker tests still pass**

Run: `python -m pytest bci/tests/test_worker.py -v --no-header -k "not moveToThread"`
Expected: tests in `TestBatchWorker`, `TestLoadWorker`, `TestStreamWorker` pass. The 3 `moveToThread CI abort` skipped tests stay skipped (don't run them).

- [ ] **Step 4: Commit**

```bash
git add bci/gui/worker.py
git commit -m "refactor: BaseWorker.stop() + cleanup() — worker owns its thread lifecycle"
```

---

## Task 6: BatchTab — type annotations + simplified `_stop_workers`

**Files:**
- Modify: `bci/gui/batch_tab.py:1-50,99-107,145`

- [ ] **Step 1: Read the current state**

Read `bci/gui/batch_tab.py` lines 1-50 and 99-110 to confirm the import block, member declarations, and current `_stop_workers`.

- [ ] **Step 2: Update member type annotations**

Edit `bci/gui/batch_tab.py`:
- Line 28: `self._source: Optional[object] = None` → `self._source: Optional['mne.io.Raw'] = None`
- Line 31: `self._worker_thread = None` → `self._worker_thread: Optional[QThread] = None`
- Line 33: `self._load_thread = None` → `self._load_thread: Optional[QThread] = None`
- Line 34: `self._pipeline: object = None` → `self._pipeline: Optional['BCIPipeline'] = None`
- Line 145: `def _on_load_finished(self, source):` → `def _on_load_finished(self, source: 'mne.io.Raw'):`

- [ ] **Step 3: Replace `_stop_workers` with shared pattern**

Edit `bci/gui/batch_tab.py` lines 99-107:

Replace the existing `_stop_workers` body with:

```python
    def _stop_workers(self):
        for w in (self._worker, self._load_worker):
            if w is not None:
                w.cleanup()
        self._worker = None
        self._load_worker = None
        self._worker_thread = None
        self._load_thread = None
```

- [ ] **Step 4: Verify the file imports cleanly**

Run: `python -c "from bci.gui.batch_tab import BatchTab; print('ok')"`
Expected: prints `ok`, no errors.

- [ ] **Step 5: Run tab tests**

Run: `python -m pytest bci/tests/test_tabs.py -v --no-header -k "not moveToThread"`
Expected: non-skipped tests pass.

- [ ] **Step 6: Commit**

```bash
git add bci/gui/batch_tab.py
git commit -m "refactor: BatchTab — tighten types, simplify _stop_workers via BaseWorker.cleanup()"
```

---

## Task 7: StreamTab — type annotations + simplified `_stop_workers`

**Files:**
- Modify: `bci/gui/stream_tab.py:1-50,195-208,245`

- [ ] **Step 1: Read the current state**

Read `bci/gui/stream_tab.py` lines 1-50 and 195-215 to confirm the import block, member declarations, and current `_stop_workers`.

- [ ] **Step 2: Update member type annotations**

Edit `bci/gui/stream_tab.py`:
- Line 33: `self._source: Optional[object] = None` → `self._source: Optional['mne.io.Raw'] = None`
- Line 34: `self._stream_source: Optional[object] = None` → `self._stream_source: Optional['StreamSource'] = None`
- Line 36: `self._worker_thread = None` → `self._worker_thread: Optional[QThread] = None`
- Line 38: `self._load_thread = None` → `self._load_thread: Optional[QThread] = None`
- Line 245: `def _on_load_finished(self, eeg):` → `def _on_load_finished(self, source: 'mne.io.Raw'):`

- [ ] **Step 3: Replace `_stop_workers` with shared pattern**

Edit `bci/gui/stream_tab.py` lines 195-208:

Replace the existing `_stop_workers` body with:

```python
    def _stop_workers(self):
        for w in (self._worker, self._load_worker):
            if w is not None:
                w.cleanup()
        self._worker = None
        self._load_worker = None
        self._worker_thread = None
        self._load_thread = None
        self.info_panel.clear()
```

(Note: keep the `info_panel.clear()` call — it's stream-specific.)

- [ ] **Step 4: Verify the file imports cleanly**

Run: `python -c "from bci.gui.stream_tab import StreamTab; print('ok')"`
Expected: prints `ok`, no errors.

- [ ] **Step 5: Run tab tests**

Run: `python -m pytest bci/tests/test_tabs.py -v --no-header -k "not moveToThread"`
Expected: non-skipped tests pass.

- [ ] **Step 6: Commit**

```bash
git add bci/gui/stream_tab.py
git commit -m "refactor: StreamTab — tighten types, rename eeg→source, simplify _stop_workers"
```

---

## Task 8: Info panel — type annotations + filepath lookup simplification

**Files:**
- Modify: `bci/gui/widgets/info_panel.py:1-15,77,105,127,166-182`

- [ ] **Step 1: Read the current state**

Read `bci/gui/widgets/info_panel.py` lines 1-15 (imports), 77, 105, 127, 166-182.

- [ ] **Step 2: Add Protocol/StreamSource import**

Edit `bci/gui/widgets/info_panel.py` imports (top of file):
- Add: `from bci.source.types import EEGSource`
- Add: `from bci.source.stream_source import StreamSource`

- [ ] **Step 3: Add type annotations to public methods**

Edit `bci/gui/widgets/info_panel.py`:
- Line 77: `def show_batch(self, source) -> None:` → `def show_batch(self, source: EEGSource) -> None:`
- Line 105: `def show_stream(self, source) -> None:` → `def show_stream(self, source: StreamSource) -> None:`
- Line 127: `def update_elapsed(self, source) -> None:` → `def update_elapsed(self, source: StreamSource) -> None:`

- [ ] **Step 4: Add types + simplify `_display_name` and `_channel_label`**

Edit `bci/gui/widgets/info_panel.py` lines 166-182:

```python
def _display_name(source: EEGSource) -> str:
    import re
    from pathlib import Path
    path = getattr(source, 'filepath', None)
    if path:
        stem = Path(str(path)).stem
        m = re.match(r'^(.*)R\d+$', stem)
        return m.group(1) if m else stem
    return "EEG Data"


def _channel_label(source: EEGSource) -> str:
    if hasattr(source, 'ch_names'):
        names = source.ch_names[:6]
        suffix = f" +{len(source.ch_names) - 6}" if len(source.ch_names) > 6 else ""
        return ", ".join(names) + suffix
    return ""
```

(Removed the `or getattr(source, 'filepath', None)` fallback — now all sources have `filepath`.)

- [ ] **Step 5: Verify file imports cleanly**

Run: `python -c "from bci.gui.widgets.info_panel import EEGInfoPanel, _display_name; print('ok')"`
Expected: prints `ok`, no errors.

- [ ] **Step 6: Run widget tests**

Run: `python -m pytest bci/tests/test_widgets.py -v --no-header`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add bci/gui/widgets/info_panel.py
git commit -m "refactor: info_panel — type annotations, simplify filepath lookup"
```

---

## Task 9: Main page + refresh_chart rename

**Files:**
- Modify: `bci/gui/widgets/main_page.py:1-50,86,89,30`
- Modify: `bci/gui/widgets/preprocess_page.py:69`
- Modify: `bci/gui/widgets/epoch_page.py:128`

- [ ] **Step 1: Read the current state of all three files**

Read `bci/gui/widgets/main_page.py` lines 1-50, 86, 89.
Read `bci/gui/widgets/preprocess_page.py` line 69 and surrounding.
Read `bci/gui/widgets/epoch_page.py` line 128 and surrounding.

- [ ] **Step 2: Main page type fixes**

Edit `bci/gui/widgets/main_page.py`:
- Line 30: `self._ch_names: list = []` → `self._ch_names: List[str] = []`
- Line 86: `def show_batch_info(self, source):` → `def show_batch_info(self, source: 'mne.io.Raw'):`
- Line 89: `def show_stream_info(self, source):` → `def show_stream_info(self, source: 'StreamSource'):`

- [ ] **Step 3: PreprocessPage — rename `source` → `data`**

Edit `bci/gui/widgets/preprocess_page.py` line 69:
- `def refresh_chart(self, source: Optional[object] = None):` → `def refresh_chart(self, data: Optional['mne.io.Raw'] = None):`

Inside the method body, replace all uses of `source` with `data`:
- Line 76: `if source is None:` → `if data is None:`
- Line 77: `ax.text(0.5, 0.5, "No data loaded", ...)` (no change — string literal)
- Line 80: `sfreq = source.info['sfreq']` → `sfreq = data.info['sfreq']`
- Line 81: `n_fft = min(2048, source.n_times)` → `n_fft = min(2048, data.n_times)`
- Line 82-84: `source.compute_psd(fmax=min(source.info['sfreq'] / 2, 80), n_fft=n_fft, ...)` → `data.compute_psd(fmax=min(data.info['sfreq'] / 2, 80), n_fft=n_fft, ...)`

- [ ] **Step 4: EpochPage — rename `pipeline` → `data`**

Edit `bci/gui/widgets/epoch_page.py` line 128:
- `def refresh_chart(self, pipeline: Optional[object] = None):` → `def refresh_chart(self, data: Optional['BCIPipeline'] = None):`

Inside the method body, replace all uses of `pipeline` with `data` (read the body first to confirm extent, but typical pattern is `pipeline.epochs if pipeline is not None else None`).

- [ ] **Step 5: Update callers in batch_tab.py**

Edit `bci/gui/batch_tab.py`:
- Line 88: `self._preprocess_page.refresh_chart(self._source)` (no change — keyword optional)
- Line 163: `self._preprocess_page.refresh_chart(self._source)` (no change)
- Line 238: `self._epoch_page.refresh_chart(pipeline)` → `self._epoch_page.refresh_chart(self._pipeline)` (use `self._pipeline` instead of bare `pipeline`)

Read line 238 first to confirm.

- [ ] **Step 6: Verify all files import cleanly**

Run:
```bash
python -c "from bci.gui.widgets.main_page import MainPage; print('ok')"
python -c "from bci.gui.widgets.preprocess_page import PreprocessPage; print('ok')"
python -c "from bci.gui.widgets.epoch_page import EpochPage; print('ok')"
python -c "from bci.gui.batch_tab import BatchTab; print('ok')"
```
Expected: all print `ok`.

- [ ] **Step 7: Run widget + tab tests**

Run: `python -m pytest bci/tests/test_widgets.py bci/tests/test_tabs.py -v --no-header -k "not moveToThread"`
Expected: non-skipped tests pass.

- [ ] **Step 8: Commit**

```bash
git add bci/gui/widgets/main_page.py bci/gui/widgets/preprocess_page.py bci/gui/widgets/epoch_page.py bci/gui/batch_tab.py
git commit -m "refactor: refresh_chart(data=...) uniform param, tighten types in main_page"
```

---

## Task 10: Worker source param types + `__import__` cleanup

**Files:**
- Modify: `bci/gui/worker.py:96,99,146,172`

- [ ] **Step 1: Read the current state**

Read `bci/gui/worker.py` lines 90-180 to see BatchWorker, StreamWorker, and the `__import__` block.

- [ ] **Step 2: Type BatchWorker source**

Edit `bci/gui/worker.py`:
- Line 96: `def __init__(self, source, config: PipelineConfig, pipeline: Optional['BCIPipeline'] = None):` → `def __init__(self, source: 'mne.io.Raw', config: PipelineConfig, pipeline: Optional['BCIPipeline'] = None):`
- Line 99: `self.source = source` (no change)

- [ ] **Step 3: Type StreamWorker source + fix `__import__`**

Edit `bci/gui/worker.py`:
- Line 146: `def __init__(self, source, chunk_duration: float = 0.1):` → `def __init__(self, source: Union['StreamSource', 'mne.io.Raw'], chunk_duration: float = 0.1):`
- Add import at top of file (or use `Union` if already imported). Check imports: read top of `bci/gui/worker.py` to see if `Union` is already imported; if not, add `from typing import Union`.
- Line 172 area: replace `__import__('bci.processor.online', fromlist=['OnlineProcessor']).OnlineProcessor(...)` with a normal import inside the method. Read lines 165-180 first to see the full block. The replacement pattern:
  ```python
  from bci.processor.online import OnlineProcessor
  self._online_proc = OnlineProcessor(
      sfreq=self.source.sfreq, n_channels=self.source.n_channels
  )
  ```
  (Replace the entire `__import__` expression.)

- [ ] **Step 4: Verify the file imports cleanly**

Run: `python -c "from bci.gui.worker import BaseWorker, LoadWorker, BatchWorker, StreamWorker; print('ok')"`
Expected: prints `ok`, no errors.

- [ ] **Step 5: Run worker + processor tests**

Run: `python -m pytest bci/tests/test_worker.py bci/tests/test_processor.py -v --no-header -k "not moveToThread"`
Expected: non-skipped tests pass.

- [ ] **Step 6: Commit**

```bash
git add bci/gui/worker.py
git commit -m "refactor: worker.py — type source params, replace __import__ with normal import"
```

---

## Task 11: Session loader, epocher, config — small cleanups

**Files:**
- Modify: `bci/gui/session_loader.py:35`
- Modify: `bci/epocher/__init__.py:68-69`
- Modify: `bci/config/__init__.py:15-16`

- [ ] **Step 1: Fix `session_loader.py` `__import__('re')`**

Read `bci/gui/session_loader.py` lines 30-40 to see context.

Edit `bci/gui/session_loader.py` line 35:
- `match = __import__('re').match(r'^(.*)R\d+$', session_name)` → `match = re.match(r'^(.*)R\d+$', session_name)`

(Verify `re` is already imported at top of file — read line 17 to confirm.)

- [ ] **Step 2: Remove epocher dead error check**

Read `bci/epocher/__init__.py` lines 65-75 to see the dead check and surrounding code.

Edit `bci/epocher/__init__.py`:
- Delete lines 68-69:
  ```python
  if self.raw is None:
      raise RuntimeError("No raw data loaded, call load() first")
  ```

- [ ] **Step 3: Remove empty `TYPE_CHECKING` guard from `config/__init__.py`**

Read `bci/config/__init__.py` lines 10-20 to see the empty guard.

Edit `bci/config/__init__.py`:
- Delete lines 15-16:
  ```python
  if TYPE_CHECKING:
      pass
  ```
  (Also remove the now-unused `TYPE_CHECKING` import at top of file if any. Read line 5-15 to confirm.)

- [ ] **Step 4: Verify all three files import cleanly**

Run:
```bash
python -c "from bci.gui.session_loader import open_session_files, SessionDialog; print('ok')"
python -c "from bci.epocher import Epocher; print('ok')"
python -c "from bci.config import create_default_config, PipelineConfig; print('ok')"
```
Expected: all print `ok`.

- [ ] **Step 5: Run related tests**

Run: `python -m pytest bci/tests/test_session.py bci/tests/test_session_loader.py -v --no-header -k "not moveToThread"`
Expected: non-skipped tests pass.

- [ ] **Step 6: Commit**

```bash
git add bci/gui/session_loader.py bci/epocher/__init__.py bci/config/__init__.py
git commit -m "refactor: clean up __import__('re'), dead epocher error check, empty TYPE_CHECKING"
```

---

## Task 12: README + final verification

**Files:**
- Modify: `README.md:50,101`

- [ ] **Step 1: Fix README description of BatchWorker**

Read `README.md` lines 45-55 and 95-105 to see current text.

Edit `README.md`:
- Line 50 area: change `BatchWorker (QThread)` to `BatchWorker (QObject)`. Update the matching description for StreamWorker to `StreamWorker (QObject+QTimer)` if needed for symmetry.
- Line 101 area: change "QThread background execution" to "background execution via `BaseWorker.moveToThread` pattern".

- [ ] **Step 2: Run full non-skipped test suite**

Run: `python -m pytest bci/tests/ -v --no-header -k "not moveToThread"`
Expected: all non-skipped tests pass. The 3 `moveToThread` skipped tests stay skipped (they're the known CI issue).

- [ ] **Step 3: Manual smoke test (optional but recommended)**

If a Qt display is available, run:
```bash
python -m bci.main
```

Load an EEG file, walk through Preprocess → Epoch → Decode steps. Verify:
- PreprocessPage shows PSD
- EpochPage shows butterfly
- DecodePage shows all methods including `csp` and `transformer_bert`
- Info panel shows File/Channels/Rate/Duration

If no display: skip this step, the test suite covers logic.

- [ ] **Step 4: Final commit**

```bash
git add README.md
git commit -m "docs: fix README — BatchWorker is QObject, not QThread"
```

---

## Self-Review

### Spec coverage
- 4.1 `EEGSource` Protocol — Task 1 ✓
- 4.2 `source_path` → `filepath` — Tasks 2, 8 ✓
- 4.2 `refresh_chart(data=...)` — Task 9 ✓
- 4.2 `eeg` → `source` — Task 7 ✓
- 4.3 Type annotation repairs (24 entries) — Tasks 2, 5, 6, 7, 8, 9, 10 ✓
- 4.4 Decoder dropdown dynamic — Task 4 ✓
- 4.5 `BaseWorker.stop()` + `cleanup()` — Tasks 5, 6, 7 ✓
- 4.6 Refactor leftovers (epocher, `__import__`, TYPE_CHECKING) — Tasks 10, 11 ✓
- 4.7 `n_times` → `n_samples` in decoders — Task 3 ✓
- 4.8 README updates — Task 12 ✓

### Placeholder scan
- No "TBD", "TODO", "implement later" anywhere
- All code blocks contain actual code
- All commands have expected output
- No "similar to Task N" — each step is self-contained

### Type consistency
- `BaseWorker._thread: Optional[QThread]` defined in Task 5 — referenced in cleanup() same task
- `EEGSource` Protocol defined in Task 1 — used in Tasks 8 (info_panel) ✓
- `filepath` rename: Tasks 2 (StreamSource.filepath property), 8 (info_panel uses `getattr(source, 'filepath', None)`) — consistent
- `refresh_chart(data=...)` rename: Tasks 9 (definition) + 9 (callers in batch_tab) — consistent

### Gaps
- None detected.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-12-refactor-cleanup.md`.

Two execution options:

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
