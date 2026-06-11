"""
Worker Threads
=============
Background workers for batch processing and real-time streaming.

All workers inherit BaseWorker (QObject) and use the moveToThread pattern:
- QThread manages the thread lifecycle only
- Worker logic lives in QObject methods
- Communication via Qt signals (thread-safe across thread boundaries)
"""
from __future__ import annotations
from abc import abstractmethod
from typing import Optional, List
import numpy as np
from pathlib import Path
from scipy.signal import welch

from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QThread

from bci.config import PipelineConfig
from bci.source import FileSource, StreamSource
from bci.pipeline import BCIPipeline

class BaseWorker(QObject):
    """Abstract base for all background workers.

    Defines the common interface:
    - ``finished = pyqtSignal(object)``  — work done, carries result or None
    - ``error    = pyqtSignal(str)``     — work failed, carries message
    - ``run()``                         — abstract entry point (same name as QThread.run)
    - ``start_in_thread()``              — moveToThread + start, returns QThread
    """
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

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
        self._thread.start()
        return self._thread


class LoadWorker(BaseWorker):
    """Background data loading worker.

    Loads EEG data via FileSource in a background thread,
    emitting progress updates so the GUI can show a loading bar.
    Once complete, emits the loaded MNE Raw object.
    """
    load_progress = pyqtSignal(int, int)

    def __init__(self, filepaths: List[str]):
        super().__init__()
        self.filepaths = list(filepaths)

    def run(self):
        try:
            eeg = FileSource.load(self.filepaths)
            self.load_progress.emit(1, 1)
            self.finished.emit(eeg)
        except Exception as e:
            self.error.emit(str(e))


class BatchWorker(BaseWorker):
    """Background pipeline execution worker (offline batch mode).

    Accepts an optional BCIPipeline instance — when provided the pipeline's
    internal state is reused so only steps whose parameters changed are
    re-executed.

    Emits progress: 0=start, 100=done.
    finished emits (PipelineResult, BCIPipeline) as a tuple.
    """
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    steps_skipped = pyqtSignal(list)

    def __init__(self, source, config: PipelineConfig,
                 pipeline: Optional['BCIPipeline'] = None):
        super().__init__()
        self.source = source
        self.config = config
        self._pipeline = pipeline

    def run(self):
        try:
            pipeline = self._pipeline or BCIPipeline(self.config)
            self.progress.emit(0)
            self.log.emit("Processing loaded data")

            if 'load' not in pipeline._steps:
                pipeline.load_raw(self.source)

            result = pipeline.run()

            if result.success:
                self.log.emit(
                    f"Accuracy: {result.accuracy:.3f} ± {result.std:.3f}")
                self.steps_skipped.emit(list(result.steps_skipped))
                self.progress.emit(100)
                self.log.emit("Pipeline complete")
                self.finished.emit((result, pipeline))
            else:
                self.error.emit(
                    result.errors[0] if result.errors else "Unknown error")
        except Exception as e:
            self.error.emit(str(e))


class StreamWorker(BaseWorker):
    """Real-time streaming worker.

    Wraps an MNE Raw in a StreamSource, optionally applies online
    filtering, and emits processed chunks via Qt signals.

    Designed to run in a dedicated QThread via moveToThread so that
    chunk processing (filtering, decoding, PSD) does not block the GUI.
    """

    chunk_processed = pyqtSignal(np.ndarray)
    spectrum_updated = pyqtSignal(np.ndarray, np.ndarray)
    prediction = pyqtSignal(str, float)
    progress = pyqtSignal(int)

    _start_timer_signal = pyqtSignal()
    _stop_timer_signal = pyqtSignal()

    def __init__(self, source, chunk_duration: float = 0.1):
        super().__init__()

        if isinstance(source, StreamSource):
            self.source = source
        else:
            self.source = StreamSource(source, chunk_duration)

        self._model = None
        self._label_names: List[str] = []

        self._timer: Optional[QTimer] = None
        self._filter_enabled = True
        self._l_freq = 0.5
        self._h_freq = 40.0
        self._speed = 1.0
        self._online_proc = None
        self._chunk_samples = 0
        self._chunk_duration = chunk_duration
        self.sliding_window = None

        self._start_timer_signal.connect(self._start_timer)
        self._stop_timer_signal.connect(self._stop_timer)

    def run(self):
        self._chunk_samples = int(self.source.sfreq * self.source.chunk_duration)
        self._online_proc = __import__('bci.processor.online',
                                       fromlist=['OnlineProcessor']).OnlineProcessor(
            sfreq=self.source.sfreq, n_channels=self.source.n_channels
        )
        self._start_timer_signal.emit()

    def _start_timer(self):
        """Create and start the QTimer — must run in the worker's thread."""
        interval_ms = int(self.source.chunk_duration * 1000 / max(0.01, self._speed))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._emit_chunk)
        self._timer.start(max(1, interval_ms))

    def pause(self):
        """Pause streaming without closing the source."""
        self._stop_timer_signal.emit()

    def _stop_timer(self):
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def stop(self):
        """Stop streaming and close the source."""
        self._stop_timer_signal.emit()
        self.source.close()
        self.finished.emit(None)

    def _emit_chunk(self):
        """Read chunk from source, process, emit signals."""
        if self.source is None or self._chunk_samples == 0:
            return
        chunk = self.source.read_chunk(self._chunk_samples)
        if chunk is None:
            self.stop()
            return

        if self._filter_enabled and self._online_proc is not None:
            chunk = self._online_proc.bandpass(chunk, self._l_freq, self._h_freq)

        self.chunk_processed.emit(chunk)

        if self._model is not None:
            try:
                if self.sliding_window is not None:
                    self.sliding_window.push(chunk)
                    if not self.sliding_window.ready():
                        freqs, psd = welch(chunk[0], self.source.sfreq,
                                           nperseg=min(128, chunk.shape[1]))
                        self.spectrum_updated.emit(freqs, psd)
                        self.progress.emit(self.source.progress)
                        return
                    window = self.sliding_window.get_window()
                    X = window[None, :, :]
                    self.sliding_window.consume()
                else:
                    X = chunk[None, :, :]

                proba = self._model.predict_proba(X)[0]
                pred_idx = int(np.argmax(proba))
                label = (self._label_names[pred_idx]
                         if pred_idx < len(self._label_names)
                         else str(pred_idx))
                confidence = float(proba[pred_idx])
                self.prediction.emit(label, confidence)
            except Exception:
                pass

        freqs, psd = welch(chunk[0], self.source.sfreq,
                           nperseg=min(128, chunk.shape[1]))
        self.spectrum_updated.emit(freqs, psd)
        self.progress.emit(self.source.progress)

    def set_filter(self, l_freq: float, h_freq: float):
        self._l_freq = l_freq
        self._h_freq = h_freq

    def set_filter_enabled(self, enabled: bool):
        self._filter_enabled = enabled

    def set_speed(self, speed: float):
        self._speed = max(0.01, speed)
        self.source.set_speed(speed)

    @property
    def speed(self) -> float:
        return self._speed

    def seek(self, sample_idx: int):
        self.source.seek(sample_idx)

    def reset(self):
        self.source.reset()
        if self._online_proc is not None:
            self._online_proc.reset_state()

    def set_loop(self, enabled: bool):
        self.source.set_loop(enabled)

    def load_model(self, model_path: str | Path):
        """Load a decoder from file for online prediction."""
        from bci.decoder.base import Decoder
        self._model = Decoder.load(model_path)
        self._label_names = [str(c) for c in
                             getattr(self._model, 'classes_', np.array([]))]

    def set_sliding_window(self, sw: "SlidingWindow") -> None:
        """Configure a SlidingWindow for windowed online prediction."""
        self.sliding_window = sw

    @property
    def has_model(self) -> bool:
        return self._model is not None
