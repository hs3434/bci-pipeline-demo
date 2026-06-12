"""
Stream Tab — Real-Time Viewing
==============================
Simulated live feed from file with playback controls.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QDoubleSpinBox, QSlider, QCheckBox, QTextEdit,
    QProgressBar, QMessageBox, QFileDialog, QSpinBox,
)
from PyQt6.QtCore import Qt, QThread

from bci.gui.widgets import (
    EEGWaveformWidget, SpectrumWidget, TopomapWidget, ResultPanel,
    EEGInfoPanel,
)
from bci.gui.worker import StreamWorker, LoadWorker

if TYPE_CHECKING:
    from mne.io import Raw
    from bci.source.stream_source import StreamSource


class StreamTab(QWidget):
    """Real-time streaming analysis tab.

    Speed control: slider (25-10000 → 0.25x-100x) + input box (0.25-100).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._filepaths: List[str] = []
        self._source: Optional[Raw] = None
        self._stream_source: Optional[StreamSource] = None
        self._worker: Optional[StreamWorker] = None
        self._worker_thread: Optional[QThread] = None
        self._load_worker: Optional[LoadWorker] = None
        self._load_thread: Optional[QThread] = None
        self._model_path: Optional[str] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        toolbar = QHBoxLayout()
        self.load_btn = QPushButton("Load EEG File")
        self.load_btn.clicked.connect(self._on_load)
        toolbar.addWidget(self.load_btn)

        self.start_btn = QPushButton("▶ Start")
        self.start_btn.clicked.connect(self._on_start)
        self.start_btn.setEnabled(False)
        toolbar.addWidget(self.start_btn)

        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.clicked.connect(self._on_pause)
        self.pause_btn.setEnabled(False)
        toolbar.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setEnabled(False)
        toolbar.addWidget(self.stop_btn)

        toolbar.addSpacing(20)
        toolbar.addWidget(QLabel("Speed:"))

        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(25, 10000)
        self.speed_slider.setValue(100)
        self.speed_slider.setMaximumWidth(150)
        self.speed_slider.valueChanged.connect(self._on_speed_slider_changed)
        toolbar.addWidget(self.speed_slider)

        self.speed_input = QDoubleSpinBox()
        self.speed_input.setRange(0.25, 100.0)
        self.speed_input.setValue(1.0)
        self.speed_input.setDecimals(2)
        self.speed_input.setSuffix("x")
        self.speed_input.valueChanged.connect(self._on_speed_input_changed)
        toolbar.addWidget(self.speed_input)

        self.model_btn = QPushButton("Load Model")
        self.model_btn.clicked.connect(self._on_load_model)
        toolbar.addWidget(self.model_btn)

        self.model_status = QLabel("No model")
        self.model_status.setStyleSheet("color: #666; font-size: 11px;")
        toolbar.addWidget(self.model_status)

        self.loop_cb = QCheckBox("Loop")
        self.loop_cb.setChecked(False)
        toolbar.addWidget(self.loop_cb)

        toolbar.addSpacing(20)
        toolbar.addWidget(QLabel("Window:"))
        self.window_size_input = QSpinBox()
        self.window_size_input.setRange(50, 5000)
        self.window_size_input.setValue(1000)
        self.window_size_input.setSuffix(" smp")
        toolbar.addWidget(self.window_size_input)

        toolbar.addWidget(QLabel("Step:"))
        self.decision_interval_input = QSpinBox()
        self.decision_interval_input.setRange(1, 1000)
        self.decision_interval_input.setValue(25)
        self.decision_interval_input.setSuffix(" smp")
        toolbar.addWidget(self.decision_interval_input)

        toolbar.addStretch()
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888;")
        toolbar.addWidget(self.status_label)
        layout.addLayout(toolbar)

        self.info_panel = EEGInfoPanel()
        layout.addWidget(self.info_panel)

        params = QGroupBox("Filter")
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Lowcut:"))
        self.l_freq = QDoubleSpinBox()
        self.l_freq.setRange(0.1, 10)
        self.l_freq.setValue(0.5)
        self.l_freq.setSuffix(" Hz")
        params_layout.addWidget(self.l_freq)

        params_layout.addWidget(QLabel("Highcut:"))
        self.h_freq = QDoubleSpinBox()
        self.h_freq.setRange(10, 100)
        self.h_freq.setValue(40)
        self.h_freq.setSuffix(" Hz")
        params_layout.addWidget(self.h_freq)
        params_layout.addStretch()
        params.setLayout(params_layout)
        layout.addWidget(params)

        content = QHBoxLayout()
        left_panel = QVBoxLayout()

        self.waveform_widget = EEGWaveformWidget()
        left_panel.addWidget(self.waveform_widget, stretch=3)

        bottom_row = QHBoxLayout()
        self.spectrum_widget = SpectrumWidget()
        bottom_row.addWidget(self.spectrum_widget)

        self.topomap_widget = TopomapWidget()
        bottom_row.addWidget(self.topomap_widget)
        left_panel.addLayout(bottom_row, stretch=1)

        content.addLayout(left_panel, stretch=3)

        right_panel = QVBoxLayout()
        self.result_panel = ResultPanel()
        right_panel.addWidget(self.result_panel, stretch=1)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet(
            "background-color: #2d2d2d; color: #aaa; font-family: monospace;"
        )
        right_panel.addWidget(self.log_area, stretch=1)
        content.addLayout(right_panel, stretch=1)
        layout.addLayout(content, stretch=1)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        self.load_label = QLabel("")
        self.load_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self.load_label.setVisible(False)
        layout.addWidget(self.load_label)

        self.load_progress_bar = QProgressBar()
        self.load_progress_bar.setVisible(False)
        self.load_progress_bar.setMaximumHeight(16)
        layout.addWidget(self.load_progress_bar)

    def _on_load(self):
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
        self._stream_source = None
        self._source = None
        self._model_path = None
        self.model_status.setText("No model")
        self.model_status.setStyleSheet("color: #666; font-size: 11px;")
        from bci.gui.session_loader import open_session_files
        filepaths = open_session_files(self)
        if filepaths:
            self._on_files_loaded([str(p) for p in filepaths])

    def _stop_workers(self):
        for w in (self._worker, self._load_worker):
            if w is not None:
                w.cleanup()
        self._worker = None
        self._load_worker = None
        self._worker_thread = None
        self._load_thread = None
        self.info_panel.clear()

    def shutdown(self):
        self._stop_workers()

    def _on_files_loaded(self, filepaths: List[str]):
        import re
        self._filepaths = filepaths
        n = len(filepaths)
        if n > 1:
            stem = Path(filepaths[0]).stem
            match = re.match(r'^(.*)R\d+$', stem)
            base = match.group(1) if match else stem
            self.status_label.setText(f"Session: {base} ({n} runs)")
        else:
            self.status_label.setText(f"Loaded: {Path(filepaths[0]).name}")

        self._start_loading()

    def _start_loading(self):
        self.start_btn.setEnabled(False)
        self.load_progress_bar.setValue(0)
        self.load_progress_bar.setVisible(True)
        self.load_label.setText("Loading...")
        self.load_label.setVisible(True)

        self._load_worker = LoadWorker(self._filepaths)
        self._load_worker.load_progress.connect(self._on_load_progress)
        self._load_worker.finished.connect(self._on_load_finished)
        self._load_worker.error.connect(self._on_load_error)
        self._load_thread = self._load_worker.start_in_thread()

    def _on_load_progress(self, current: int, total: int):
        self.load_label.setText(f"Loading run {current}/{total}...")
        self.load_progress_bar.setMaximum(total)
        self.load_progress_bar.setValue(current)

    def _on_load_finished(self, source: Raw):
        from bci.source import StreamSource
        self._stream_source = StreamSource(source, filepath=self._filepaths[0] if self._filepaths else None)
        self._load_worker = None
        self._load_thread = None
        self.load_progress_bar.setVisible(False)
        self.load_label.setVisible(False)
        self.info_panel.show_stream(self._stream_source)
        self.status_label.setText(
            f"Ready — {self._stream_source.n_channels} ch, "
            f"{self._stream_source.total_samples / self._stream_source.sfreq:.1f}s"
        )
        self.start_btn.setEnabled(True)

    def _on_load_error(self, msg: str):
        self._load_worker = None
        self._load_thread = None
        self.load_progress_bar.setVisible(False)
        self.load_label.setVisible(False)
        self.log_area.append(f"ERROR: {msg}")
        self.status_label.setText(f"Load error: {msg[:50]}")
        QMessageBox.warning(self, "Load Error", msg)

    def _on_start(self):
        if self._stream_source is None:
            return
        if self._worker is not None:
            return

        self.status_label.setText("Streaming...")
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)

        self._worker = StreamWorker(self._stream_source)
        self._worker.set_speed(self.speed_input.value())
        self._worker.set_filter(self.l_freq.value(), self.h_freq.value())
        self._worker.set_loop(self.loop_cb.isChecked())

        from bci.streaming import SlidingWindow
        swin = SlidingWindow(
            n_channels=self._stream_source.n_channels,
            window_size=self.window_size_input.value(),
            decision_interval=self.decision_interval_input.value(),
        )
        self._worker.set_sliding_window(swin)

        n_ch = self._stream_source.n_channels
        sfreq = self._stream_source.sfreq
        ch_names = [f'Ch {i}' for i in range(n_ch)]
        self.waveform_widget._init_buffer(n_ch, sfreq, ch_names)

        self._worker.chunk_processed.connect(self._on_chunk)
        self._worker.finished.connect(self._on_stream_finished)
        self._worker.error.connect(self._on_error)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.prediction.connect(self._on_prediction)

        if self._model_path:
            try:
                self._worker.load_model(self._model_path)
            except Exception as e:
                self.log_area.append(f"Model load error: {e}")

        self._worker_thread = self._worker.start_in_thread()

    def _on_pause(self):
        if self._worker is not None:
            self._worker.pause()
            self._worker = None
            self.pause_btn.setEnabled(False)
            self.start_btn.setEnabled(True)
            self.start_btn.setText("▶ Resume")
            self.status_label.setText("Paused")

    def _on_stop(self):
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
        self._stream_source = None
        self._source = None
        self.info_panel.clear()
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Load EEG File")
        self.status_label.setText("Stopped")
        self.progress.setValue(0)
        self.waveform_widget.clear()

    def _on_chunk(self, chunk):
        self.waveform_widget.update_stream(chunk)
        if self._stream_source is not None:
            self.spectrum_widget.update_psd(chunk, self._stream_source.sfreq)
            self.info_panel.update_elapsed(self._stream_source)

    def _on_stream_finished(self, _=None):
        self.status_label.setText("Playback complete")
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.start_btn.setText("▶ Start")
        self._worker = None

    def _on_error(self, msg: str):
        self.log_area.append(f"ERROR: {msg}")
        self.status_label.setText(f"Error: {msg[:50]}")
        QMessageBox.warning(self, "Stream Error", msg)

    def _on_speed_slider_changed(self, value: int):
        speed = value / 100.0
        self.speed_input.blockSignals(True)
        self.speed_input.setValue(speed)
        self.speed_input.blockSignals(False)
        if self._worker is not None:
            self._worker.set_speed(speed)

    def _on_speed_input_changed(self, value: float):
        self.speed_slider.blockSignals(True)
        self.speed_slider.setValue(int(value * 100))
        self.speed_slider.blockSignals(False)
        if self._worker is not None:
            self._worker.set_speed(value)

    def _on_load_model(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "", "Pickle files (*.pkl);;All files (*)"
        )
        if not filepath:
            return
        self._model_path = filepath
        self.model_status.setText(f"Model: {Path(filepath).stem}")
        self.model_status.setStyleSheet("color: #00cc66; font-size: 11px;")

    def _on_prediction(self, label: str, confidence: float):
        self.result_panel.update_stream(f"{label} ({confidence:.0%})")
