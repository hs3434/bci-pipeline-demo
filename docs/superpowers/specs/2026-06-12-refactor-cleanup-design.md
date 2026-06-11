# Design: Refactor Cleanup — Type Annotations, API Consistency, Thread Lifecycle

**Date:** 2026-06-12
**Status:** Approved (pending user review of written spec)
**Scope:** GUI layer + supporting modules (bci/gui, bci/source, bci/decoder, bci/epocher, bci/config)

## 1. Background

A previous refactor (commit `27cbfef` "refactor: consumer layer uses Raw, remove EEGData references") replaced the project-wide `EEGData` domain type with `mne.io.Raw`. The core pipeline, preprocessor, epocher, and decoder modules were updated to use the new type, but several consumer modules — primarily the GUI layer — were left with stale type annotations, dead code paths, and inconsistent naming.

This design addresses the inconsistencies found during a manual audit of the codebase on 2026-06-12. The findings are summarized inline below.

## 2. Goals

1. Type annotations on `source`/`raw`/`pipeline` reflect the actual runtime type (`mne.io.Raw`, `StreamSource`, `BCIPipeline`) — no `object` placeholders.
2. Naming is consistent: `filepath` (not `source_path`); `refresh_chart(data=...)` (not `source`/`pipeline`); `n_samples` (not `n_times`) for the time dimension in decoder docstrings and code.
3. Decoder method dropdown in `DecodePage` is sourced from `bci.decoder.list_methods()` — no more manual drift.
4. `BaseWorker` owns its thread lifecycle — `BatchTab._stop_workers()` and `StreamTab._stop_workers()` deduplicate into a shared pattern.
5. Dead code paths and import hacks (`__import__`, empty `TYPE_CHECKING`) are removed.

## 3. Non-Goals

- Skipped tests for `moveToThread` CI aborts are **not** fixed in this work — investigation deferred.
- Decoder `n_times`/`n_samples` in `transformer_bert.py:216` where the variable holds a tuple unpack — only docstrings and constructor parameters are unified, not internal array unpacks that happen to use these names incidentally.
- README architecture diagram updates are in-scope (line 50, 101) because they directly describe the redesigned `BaseWorker` pattern.
- GUI visual design, layout, and existing widget behavior are unchanged.

## 4. Architecture

### 4.1 New: `EEGSource` Protocol

A structural type covering the union `mne.io.Raw | StreamSource` that appears at the GUI/info-panel boundary.

```python
# bci/source/types.py
from typing import Protocol, List, Optional, runtime_checkable

@runtime_checkable
class EEGSource(Protocol):
    """Structural interface satisfied by mne.io.Raw and StreamSource.

    Both types expose the attributes/info needed by GUI consumers
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

**Why:** Replaces `Union['mne.io.Raw', 'StreamSource']` at the GUI/info-panel boundary. The protocol names the concept ("a thing that is a source of EEG data") and avoids repeating `Union[...]` at every consumer. The class is `runtime_checkable` so that `isinstance(x, EEGSource)` works for defensive checks if desired.

**File:** new `bci/source/types.py` (single module, ~15 lines).

### 4.2 Naming Unification

| Current | New | Files affected |
|---|---|---|
| `StreamSource.source_path` | `StreamSource.filepath` | `bci/source/stream_source.py`, `bci/gui/widgets/info_panel.py` |
| `refresh_chart(source=...)` | `refresh_chart(data=...)` | `bci/gui/widgets/preprocess_page.py` |
| `refresh_chart(pipeline=...)` | `refresh_chart(data=...)` | `bci/gui/widgets/epoch_page.py` |
| `n_times` (decoder constructor param) | `n_samples` | `bci/decoder/deep.py` |
| `n_times` in decoder docstrings | `n_samples` | `bci/decoder/deep.py`, `bci/decoder/csp.py`, `bci/decoder/base.py` |
| `eeg` parameter name (stream_tab callback) | `source` | `bci/gui/stream_tab.py` |

**StreamSource.filepath** replaces `source_path` to match `FileSource.load(filepath=...)`, `EEGReader.read(filepath=...)`, CLI `args.filepath`. The `_display_name` helper in `info_panel.py:166-174` simplifies to a single `getattr(source, 'filepath', None)`.

**refresh_chart(data=...)** is uniform across `PreprocessPage` (which displays PSD from `mne.io.Raw`) and `EpochPage` (which displays ERP/butterfly from `BCIPipeline.epochs`). The `data` parameter is typed as `Optional[EEGSource]` and `Optional[BCIPipeline]` respectively, since they consume different things.

### 4.3 Type Annotation Repairs

All `Optional[object]` / untyped parameters and members are tightened. Forward-reference strings (`'mne.io.Raw'`) are used in the GUI modules. **Rationale:** PyQt6 startup is on the main thread; deferring the mne import to the worker thread (where it already runs) keeps GUI launch responsive. The source-layer modules (`bci/source/*.py`) and the epocher/pipeline modules use direct `import mne` (already in use).

| File | Line(s) | Before | After |
|---|---|---|---|
| `gui/batch_tab.py` | 28 | `self._source: Optional[object] = None` | `self._source: Optional['mne.io.Raw'] = None` |
| `gui/batch_tab.py` | 31, 33 | `self._worker_thread = None` | `self._worker_thread: Optional[QThread] = None` |
| `gui/batch_tab.py` | 34 | `self._pipeline: object = None` | `self._pipeline: Optional['BCIPipeline'] = None` |
| `gui/batch_tab.py` | 145 | `def _on_load_finished(self, source):` | `def _on_load_finished(self, source: 'mne.io.Raw'):` |
| `gui/stream_tab.py` | 33 | `self._source: Optional[object] = None` | `self._source: Optional['mne.io.Raw'] = None` |
| `gui/stream_tab.py` | 34 | `self._stream_source: Optional[object] = None` | `self._stream_source: Optional[StreamSource] = None` |
| `gui/stream_tab.py` | 36, 38 | `self._worker_thread = None` | `self._worker_thread: Optional[QThread] = None` |
| `gui/stream_tab.py` | 245 | `def _on_load_finished(self, eeg):` | `def _on_load_finished(self, source: 'mne.io.Raw'):` |
| `gui/widgets/preprocess_page.py` | 69 | `def refresh_chart(self, source: Optional[object] = None):` | `def refresh_chart(self, data: Optional['mne.io.Raw'] = None):` |
| `gui/widgets/epoch_page.py` | 128 | `def refresh_chart(self, pipeline: Optional[object] = None):` | `def refresh_chart(self, data: Optional['BCIPipeline'] = None):` |
| `gui/widgets/info_panel.py` | 77 | `def show_batch(self, source) -> None:` | `def show_batch(self, source: EEGSource) -> None:` |
| `gui/widgets/info_panel.py` | 105 | `def show_stream(self, source) -> None:` | `def show_stream(self, source: StreamSource) -> None:` |
| `gui/widgets/info_panel.py` | 127 | `def update_elapsed(self, source) -> None:` | `def update_elapsed(self, source: StreamSource) -> None:` |
| `gui/widgets/info_panel.py` | 166 | `def _display_name(source) -> str:` | `def _display_name(source: EEGSource) -> str:` |
| `gui/widgets/info_panel.py` | 177 | `def _channel_label(source) -> str:` | `def _channel_label(source: EEGSource) -> str:` |
| `gui/widgets/main_page.py` | 30 | `self._ch_names: list = []` | `self._ch_names: List[str] = []` |
| `gui/widgets/main_page.py` | 86, 89 | `def show_batch_info(self, source):` | `def show_batch_info(self, source: 'mne.io.Raw'):` |
| `gui/widgets/main_page.py` | 89 | `def show_stream_info(self, source):` | `def show_stream_info(self, source: StreamSource):` |
| `gui/worker.py` | 96 | `def __init__(self, source, ...):` | `def __init__(self, source: 'mne.io.Raw', ...):` |
| `gui/worker.py` | 146 | `def __init__(self, source, chunk_duration):` | `def __init__(self, source: Union[StreamSource, 'mne.io.Raw'], chunk_duration):` |
| `source/stream_source.py` | 26 | `def __init__(self, raw, ...):` | `def __init__(self, raw: 'mne.io.Raw', ...):` |
| `source/stream_source.py` | 28 | `self._raw = raw` | `self._raw: 'mne.io.Raw' = raw` |
| `source/base.py` | 21 | `def read(self, filepath: Path):` | `def read(self, filepath: Path) -> 'mne.io.Raw':` |
| `source/readers.py` | 16 | `def read(self, filepath: Path):` | `def read(self, filepath: Path) -> 'mne.io.Raw':` |

### 4.4 Decoder Dropdown — Dynamic Source

`DecodePage` (`gui/widgets/decode_page.py:31`) currently has hardcoded:
```python
self._method.addItems(['lda', 'ssvep', 'fbcca', 'cnn', 'transformer'])
```

Misses `csp` and `transformer_bert` registered in `bci/decoder/__init__.py`.

**New behavior:** import `list_methods()` from `bci.decoder` and pass the result:
```python
from bci.decoder import list_methods
self._method.addItems(list_methods())
```

This guarantees GUI ↔ registry never drift. `list_methods()` already exists in `bci/decoder/__init__.py:62` — no decoder-side changes needed.

### 4.5 `BaseWorker.stop()` + Thread Lifecycle

**Current state:** `BaseWorker.start_in_thread()` creates a `QThread`, moves the worker into it, and starts. `BatchTab._stop_workers()` and `StreamTab._stop_workers()` each manually call `worker.stop()` (only StreamWorker has one), `thread.quit()`, `thread.wait()`, and null out references. The logic is duplicated and inconsistent.

**New design:**

```python
# bci/gui/worker.py
class BaseWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._thread: Optional[QThread] = None

    @abstractmethod
    def run(self): ...

    def start_in_thread(self, slot=None, parent=None) -> QThread:
        """Move this worker into a new QThread, connect signals, and start.
        Stores the QThread internally for later cleanup.
        """
        self._thread = QThread(parent)
        self.moveToThread(self._thread)
        self._thread.started.connect(slot or self.run)
        self.finished.connect(self._thread.quit)
        self.finished.connect(self._thread.wait)
        self._thread.start()
        return self._thread

    def stop(self):
        """Default no-op. Subclasses override to halt periodic work
        (e.g. StreamWorker stops its QTimer)."""
        pass

    def cleanup(self):
        """Stop the worker, quit its thread, and wait for it to finish.
        Safe to call multiple times. Safe to call even if never started.
        """
        self.stop()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
        self._thread = None
```

**Consumer cleanup becomes:**
```python
# BatchTab and StreamTab (identical)
def _stop_workers(self):
    for w in (self._worker, self._load_worker):
        if w is not None:
            w.cleanup()
    self._worker = None
    self._load_worker = None
    self._worker_thread = None
    self._load_thread = None
```

`StreamWorker` already implements `stop()` — keep it. `LoadWorker` and `BatchWorker` inherit the no-op default.

**Thread tracking:** previously the caller stored the QThread reference (`self._worker_thread`). Now `BaseWorker` stores it internally (`self._thread`). Callers no longer need their own `_worker_thread` field. The returned `QThread` is still useful for external observation (e.g. `isRunning()` checks during shutdown), so `start_in_thread()` keeps returning it, and callers keep a reference for safety, but the cleanup logic uses `BaseWorker._thread` only.

**`finished → _thread.wait` connection:** added so the thread fully drains before the next `start_in_thread()` is called — prevents race where `cleanup()` runs while `run()` is still mid-emit. The existing `finished → _thread.quit` connection stays.

### 4.6 Refactor Leftover Cleanup

| File | Line | Cleanup |
|---|---|---|
| `bci/epocher/__init__.py` | 68-69 | Delete the `if self.raw is None: raise RuntimeError("No raw data loaded, call load() first")` block. `raw` is a required constructor argument; the check is dead. |
| `bci/gui/session_loader.py` | 35 | Replace `__import__('re').match(...)` with `re.match(...)` (top-level `import re` already at line 17). |
| `bci/gui/worker.py` | 172 | Replace `__import__('bci.processor.online', fromlist=['OnlineProcessor']).OnlineProcessor(...)` with `from bci.processor.online import OnlineProcessor` inside the method. |
| `bci/gui/session_loader.py` | 18 | Leave `import glob as glob_lib` — alias is intentional, no action. |
| `bci/config/__init__.py` | 15-16 | Delete empty `if TYPE_CHECKING: pass` block. If forward references become needed later, re-add with real content. |
| `bci/gui/batch_tab.py` & `stream_tab.py` | (none) | No change. Keep `'mne.io.Raw'` as forward-reference strings. `import mne` happens only in source/worker thread context. |

> **Note on mne imports in GUI:** The GUI files use string forward references (`'mne.io.Raw'`) for type annotations only. No runtime `import mne` is added to GUI files. This avoids forcing the heavy mne import on the main thread during PyQt6 startup.

### 4.7 `n_times` → `n_samples` in Decoders

`n_samples` is the time-dimension name used by `mne.Epochs.get_data()` shape `(n_epochs, n_channels, n_samples)`. The project should follow this convention.

**Findings from re-read of `bci/decoder/deep.py`:**

- Line 18: `_EEGCNN.__init__` parameter `n_times: int` — **inner model class**, this is the time-dim it operates on. Rename to `n_samples: int` to match mne convention.
- Line 28: `dummy = torch.zeros(1, 1, n_channels, n_times)` — references line 18 param. Update to `n_samples`.
- Line 45: `CNNDecoder` docstring already says `(n_epochs, n_channels, n_samples)` — no change.
- Line 60: `n_epochs, n_channels, n_times = X.shape` — **local unpacking** where `n_times` here is the time dim per epoch. This actually **does** follow mne convention `(n_epochs, n_channels, n_times)`, where mne uses `n_times` for time samples. So this line is **correct under mne convention** and is **NOT changed**.
- Line 61, 66: pass `n_times` forward to `_EEGCNN` — once line 18/28 is renamed, update to pass `n_samples`.

**`bci/decoder/csp.py`, `bci/decoder/base.py`:** docstrings already say `n_samples` — verified, no change.

**`bci/decoder/transformer_bert.py:216`:** `n_samples, n_channels, n_times = X.shape` is a **local unpacking** where `n_samples` means "number of epochs" and `n_times` means "time samples per epoch". This follows the **mne convention** `(n_epochs, n_channels, n_times)` where mne uses `n_times` for the time dim. **Not changed** — local variable names in unpack statements are out of scope.

**Net effect:** `bci/decoder/deep.py` lines 18, 28, 61, 66 change (rename `_EEGCNN.__init__` param `n_times` → `n_samples` and propagate to the 3 internal references). The local unpack at line 60 (`n_epochs, n_channels, n_times = X.shape`) is **not changed** — `n_times` here refers to the time dim per epoch under mne's `(n_epochs, n_channels, n_times)` convention and is the correct name in that context.

### 4.8 README Updates

`README.md:50` and `:101` describe `BatchWorker (QThread)`. Update to:
```
│  │ BatchWorker │     │ StreamWorker        │       │
│  │ (QObject)   │     │ (QObject+QTimer)    │       │
```
and
```
**Batch tab**: 4-step pipeline with visual progress indicators, background execution via `BaseWorker.moveToThread` pattern.
```

## 5. Data Flow

Unchanged. The refactor is purely about types, names, and lifecycle ownership — no behavior change.

## 6. Error Handling

Unchanged. Removing the epocher dead-error check has no functional impact (the condition was unreachable).

## 7. Testing Strategy

1. **Existing tests:** all non-`@skip`'d tests must continue to pass. Run `pytest -m "not skip"` (or equivalent deselect) to verify.
2. **Skipped tests:** the 3 tests skipped for `moveToThread CI abort` reasons stay skipped. Add a one-line comment to each `@pytest.mark.skip` noting the investigation is deferred to a future spec.
3. **New type annotations:** if a `mypy` or `pyright` config exists in the project, run it to verify annotation correctness. If neither is configured, do not introduce one (out of scope).
4. **Manual smoke:** run `python -m bci.main` and `python -m bci.main --stream`, load a file, walk through Preprocess/Epoch/Decode to confirm `refresh_chart(data=...)` still works with renamed parameters.

## 8. Out of Scope

- Fixing the 3 moveToThread CI test aborts (separate investigation).
- New `mypy.ini` / `pyrightconfig.json` (not requested).
- Refactoring the deep CNN/Transformer code beyond the `n_samples` rename.
- Restructuring `gui/batch_tab.py` / `stream_tab.py` further (the `_stop_workers` simplification is the limit).
- Adding the `EEGSource` protocol to `bci.source.__init__` exports as `EEGSource` for cleaner imports — discussed but not done here (low value, can be added later if desired).

## 9. Open Questions

None. All decisions confirmed during brainstorming:

- `EEGSource` Protocol: yes
- `BaseWorker.stop() + cleanup()`: yes
- Decoder dropdown dynamic from `list_methods()`: yes
- `n_times` → `n_samples` (decoder side, not transformer_bert local unpacks): yes
- `refresh_chart(data=...)` uniform parameter name: yes
- `StreamSource.source_path` → `filepath`: yes
- Skipped tests stay skipped: yes
- Verify with `pytest -m "not skip"`: yes
