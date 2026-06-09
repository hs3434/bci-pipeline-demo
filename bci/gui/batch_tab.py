"""
Batch Tab -- Offline Analysis
=============================
Load file -> configure per-step params -> Run pipeline -> view results.
"""
from __future__ import annotations
from typing import Optional, List
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QMessageBox, QStackedWidget,
)

from bci.config import create_default_config
from bci.gui.widgets import (
    MainPage, PreprocessPage, EpochPage, DecodePage,
    StepStrip, StepStatus,
)
from bci.gui.worker import BatchWorker, LoadWorker
from bci.source import SessionSource


class BatchTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._filepaths: List[str] = []
        self._source: Optional[SessionSource] = None
        self._config = create_default_config()
        self._worker: Optional[BatchWorker] = None
        self._load_worker: Optional[LoadWorker] = None
        self._pipeline: object = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        toolbar = QHBoxLayout()
        self.load_btn = QPushButton("Load EEG File")
        self.load_btn.clicked.connect(self._on_load)
        toolbar.addWidget(self.load_btn)

        self.run_btn = QPushButton("Run Pipeline")
        self.run_btn.clicked.connect(self._on_run)
        self.run_btn.setEnabled(False)
        toolbar.addWidget(self.run_btn)

        self.save_btn = QPushButton("Export Results")
        self.save_btn.clicked.connect(self._on_save)
        self.save_btn.setEnabled(False)
        toolbar.addWidget(self.save_btn)

        toolbar.addStretch()
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888;")
        toolbar.addWidget(self.status_label)
        layout.addLayout(toolbar)

        self.step_strip = StepStrip()
        self.step_strip.step_clicked.connect(self._on_step_clicked)
        self.step_strip.rerun_clicked.connect(self._on_run)
        layout.addWidget(self.step_strip)

        self._main_page = MainPage()
        self._preprocess_page = PreprocessPage()
        self._epoch_page = EpochPage()
        self._decode_page = DecodePage()

        self._pages = QStackedWidget()
        self._pages.addWidget(self._main_page)
        self._pages.addWidget(self._preprocess_page)
        self._pages.addWidget(self._epoch_page)
        self._pages.addWidget(self._decode_page)
        layout.addWidget(self._pages, stretch=1)

    def _on_step_clicked(self, idx: int):
        self._pages.setCurrentIndex(idx)
        if idx == 0:
            pass
        elif idx == 1:
            self._preprocess_page.refresh_chart(self._source)
        elif idx == 2:
            self._epoch_page.refresh_chart(self._pipeline)
        elif idx == 3:
            self._decode_page.refresh_chart()

    def _stop_workers(self):
        for w in (self._worker, self._load_worker):
            if w is not None and w.isRunning():
                w.quit()
                w.wait()
        self._worker = None
        self._load_worker = None

    def shutdown(self):
        self._stop_workers()

    def _on_load(self):
        self._stop_workers()
        self._main_page.clear_info()
        from bci.gui.session_loader import open_session_files
        filepaths = open_session_files(self)
        if filepaths:
            self._on_files_loaded([str(p) for p in filepaths])

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
        self.run_btn.setEnabled(False)
        self._main_page.show_load_progress(0, 1)
        self.step_strip.set_all_pending()

        self._load_worker = LoadWorker(self._filepaths)
        self._load_worker.load_progress.connect(self._main_page.show_load_progress)
        self._load_worker.finished.connect(self._on_load_finished)
        self._load_worker.error.connect(self._on_load_error)
        self._load_worker.start()

    def _on_load_finished(self, source):
        self._source = source
        self._load_worker = None
        self._main_page.hide_load_progress()
        self._main_page.show_batch_info(source)
        self.status_label.setText(
            f"Ready — {source.n_channels} ch, "
            f"{source.total_samples / source.sfreq:.1f}s"
        )
        n_ch = min(8, source.n_channels)
        ch_names = [f'Ch {i}' for i in range(n_ch)]
        if source._data_list:
            d = source._data_list[0]
            n_ch = min(8, d.shape[0])
            self._main_page.plot_waveform(d[:n_ch], source.sfreq, ch_names[:n_ch])

        self._pages.setCurrentIndex(0)
        self.step_strip.set_active(0)
        self.step_strip.set_status(0, StepStatus.DONE)
        self._preprocess_page.refresh_chart(self._source)
        self.run_btn.setEnabled(True)

    def _on_load_error(self, msg: str):
        self._load_worker = None
        self._main_page.hide_load_progress()
        self.status_label.setText(f"Load error: {msg[:50]}")
        QMessageBox.warning(self, "Load Error", msg)

    def _on_run(self):
        if not self._filepaths:
            return
        self._config.filter.l_freq = self._preprocess_page.l_freq
        self._config.filter.h_freq = self._preprocess_page.h_freq
        self._config.epoch.tmin = self._epoch_page.tmin
        self._config.epoch.tmax = self._epoch_page.tmax
        self._config.epoch.reject_threshold = {
            'eeg': self._epoch_page.reject_uv * 1e-6
        }
        self._config.decode.method = self._decode_page.method
        self._config.decode.cv_folds = self._decode_page.cv_folds

        self._pipeline = None
        self.run_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self._main_page.clear_log()
        self._main_page.reset_progress()
        self.status_label.setText("Running pipeline...")
        self.step_strip.set_all_pending()
        self.step_strip.set_status(0, StepStatus.DONE)
        self.step_strip.set_status(1, StepStatus.RUNNING)
        self._pages.setCurrentIndex(1)
        self.step_strip.set_active(1)

        self._worker = BatchWorker(self._filepaths, self._config)
        self._worker.log.connect(self._main_page.append_log)
        self._worker.progress.connect(self._on_pipeline_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_pipeline_progress(self, val: int):
        self._main_page.set_pipeline_progress(val)
        if val >= 50:
            self.step_strip.set_status(1, StepStatus.DONE)
            self.step_strip.set_status(2, StepStatus.RUNNING)
            self._pages.setCurrentIndex(2)
            self.step_strip.set_active(2)
        if val >= 70:
            self.step_strip.set_status(2, StepStatus.DONE)
            self.step_strip.set_status(3, StepStatus.RUNNING)
            self._pages.setCurrentIndex(3)
            self.step_strip.set_active(3)
            self._epoch_page.refresh_chart(self._pipeline)

    def _on_finished(self, result, pipeline):
        self._pipeline = pipeline
        self.step_strip.set_status(1, StepStatus.DONE)
        self.step_strip.set_status(2, StepStatus.DONE)
        self.step_strip.set_status(3, StepStatus.DONE)

        if result and result.accuracy is not None:
            self.status_label.setText(
                f"Done! Accuracy: {result.accuracy:.3f} "
                f"+/- {result.std:.3f}"
            )
            self.save_btn.setEnabled(True)
            self._main_page.show_result(
                result.accuracy, result.std,
                result.cv_scores, self._decode_page.method,
            )
            self._decode_page.show_result(
                result.accuracy, result.std,
                result.cv_scores, self._decode_page.method,
            )
            self._epoch_page.refresh_chart(pipeline)

        self._pages.setCurrentIndex(0)
        self.step_strip.set_active(0)
        self.run_btn.setEnabled(True)

    def _on_error(self, msg: str):
        cur = self.step_strip._active_idx
        self.step_strip.mark_error(max(0, cur))
        self._main_page.append_log(f"ERROR: {msg}")
        self.status_label.setText(f"Error: {msg[:50]}")
        self.run_btn.setEnabled(True)
        QMessageBox.warning(self, "Pipeline Error", msg)

    def _on_save(self):
        if self._worker is None or not self._filepaths:
            return
        from bci.pipeline import BCIPipeline
        pipeline = BCIPipeline(self._config)
        pipeline.run(Path(self._filepaths[0]))
        saved = pipeline.save_results()
        self._main_page.append_log(f"Saved to: {saved}")
        QMessageBox.information(
            self, "Export", f"Results saved to {self._config.output_dir}"
        )
