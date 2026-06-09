"""
Main Page
=========
Overview page: Info Panel + Waveform (scrollable) + Results + Log + Progress.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QProgressBar, QScrollBar,
)
from PyQt6.QtCore import Qt, pyqtSignal

from bci.gui.widgets.info_panel import EEGInfoPanel
from bci.gui.widgets.waveform import EEGWaveformWidget
from bci.gui.widgets.result_panel import ResultPanel


class MainPage(QFrame):

    load_btn_clicked = pyqtSignal()
    run_btn_clicked = pyqtSignal()
    save_btn_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._full_data: Optional[np.ndarray] = None
        self._sfreq: float = 256.0
        self._ch_names: list = []
        self._window_sec: float = 5.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)

        self._info_panel = EEGInfoPanel()
        layout.addWidget(self._info_panel)

        content = QHBoxLayout()
        self._waveform = EEGWaveformWidget()
        content.addWidget(self._waveform, stretch=4)

        self._result_panel = ResultPanel()
        content.addWidget(self._result_panel, stretch=1)
        layout.addLayout(content, stretch=1)

        self._scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self._scrollbar.setMaximumHeight(14)
        self._scrollbar.valueChanged.connect(self._on_scroll)
        self._scrollbar.setVisible(False)
        layout.addWidget(self._scrollbar)

        bottom = QHBoxLayout()
        self._log_area = QTextEdit()
        self._log_area.setReadOnly(True)
        self._log_area.setMaximumHeight(100)
        self._log_area.setStyleSheet(
            "background-color: #2d2d2d; color: #aaa; font-family: monospace;"
        )
        bottom.addWidget(self._log_area, stretch=3)

        progress_col = QVBoxLayout()
        self._load_label = QLabel("")
        self._load_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._load_label.setVisible(False)
        progress_col.addWidget(self._load_label)

        self._load_progress = QProgressBar()
        self._load_progress.setVisible(False)
        self._load_progress.setMaximumHeight(16)
        progress_col.addWidget(self._load_progress)

        self._pipeline_progress = QProgressBar()
        progress_col.addWidget(self._pipeline_progress)
        bottom.addLayout(progress_col, stretch=1)
        layout.addLayout(bottom)

    @property
    def waveform(self) -> EEGWaveformWidget:
        return self._waveform

    @property
    def result_panel(self) -> ResultPanel:
        return self._result_panel

    def show_batch_info(self, source):
        self._info_panel.show_batch(source)

    def show_stream_info(self, source):
        self._info_panel.show_stream(source)

    def clear_info(self):
        self._info_panel.clear()

    def plot_waveform(self, data, sfreq, ch_names, window_sec: float = 5.0):
        self._full_data = data
        self._sfreq = sfreq
        self._ch_names = ch_names
        self._window_sec = window_sec
        total_duration = data.shape[1] / sfreq
        if total_duration > window_sec:
            max_scroll = max(0, int((total_duration - window_sec) * 100))
            self._scrollbar.setRange(0, max_scroll)
            self._scrollbar.setValue(0)
            self._scrollbar.setVisible(True)
        else:
            self._scrollbar.setVisible(False)
        self._waveform.plot_batch_window(data, sfreq, ch_names, 0.0, window_sec)

    def _on_scroll(self, value: int):
        if self._full_data is None:
            return
        t_start = value / 100.0
        self._waveform.plot_batch_window(
            self._full_data, self._sfreq, self._ch_names,
            t_start, self._window_sec,
        )

    def show_result(self, accuracy, std, cv_scores, method):
        self._result_panel.update_batch(accuracy, std, cv_scores, method)

    def clear_result(self):
        self._result_panel.clear()

    def append_log(self, text: str):
        self._log_area.append(text)

    def clear_log(self):
        self._log_area.clear()

    def show_load_progress(self, current: int, total: int):
        self._load_label.setText(f"Loading run {current}/{total}...")
        self._load_label.setVisible(True)
        self._load_progress.setMaximum(total)
        self._load_progress.setValue(current)
        self._load_progress.setVisible(True)

    def hide_load_progress(self):
        self._load_label.setVisible(False)
        self._load_progress.setVisible(False)

    def set_pipeline_progress(self, value: int):
        self._pipeline_progress.setValue(value)

    def reset_progress(self):
        self._pipeline_progress.setValue(0)
