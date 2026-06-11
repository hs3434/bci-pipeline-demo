"""
Epoch Page
==========
Epoch parameter configuration + ERP average preview chart.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QDoubleSpinBox,
    QComboBox, QLineEdit,
)
from PyQt6.QtCore import pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class EpochPage(QFrame):

    epoch_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)

        left = QVBoxLayout()

        grp = QGroupBox("Epoch Parameters")
        glay = QHBoxLayout()
        glay.addWidget(QLabel("tmin:"))
        self._tmin = QDoubleSpinBox()
        self._tmin.setRange(-1.0, 0)
        self._tmin.setValue(-0.2)
        self._tmin.setSuffix(" s")
        self._tmin.valueChanged.connect(self.epoch_changed.emit)
        glay.addWidget(self._tmin)
        glay.addWidget(QLabel("tmax:"))
        self._tmax = QDoubleSpinBox()
        self._tmax.setRange(0.1, 2.0)
        self._tmax.setValue(0.5)
        self._tmax.setSuffix(" s")
        self._tmax.valueChanged.connect(self.epoch_changed.emit)
        glay.addWidget(self._tmax)
        glay.addWidget(QLabel("Reject:"))
        self._reject = QDoubleSpinBox()
        self._reject.setRange(50, 2000)
        self._reject.setValue(300)
        self._reject.setSuffix(" μV")
        self._reject.valueChanged.connect(self.epoch_changed.emit)
        glay.addWidget(self._reject)
        glay.addStretch()
        grp.setLayout(glay)
        left.addWidget(grp)

        evt_grp = QGroupBox("Event Config")
        elay = QHBoxLayout()
        elay.addWidget(QLabel("Source:"))
        self._event_source = QComboBox()
        self._event_source.addItems(['auto', 'stim', 'annotations'])
        self._event_source.currentTextChanged.connect(self.epoch_changed.emit)
        elay.addWidget(self._event_source)
        elay.addWidget(QLabel("ID map:"))
        self._event_id = QLineEdit()
        self._event_id.setPlaceholderText("auto")
        self._event_id.textChanged.connect(self.epoch_changed.emit)
        elay.addWidget(self._event_id)
        elay.addStretch()
        evt_grp.setLayout(elay)
        left.addWidget(evt_grp)
        left.addStretch()
        layout.addLayout(left)

        self._chart = self._make_chart()
        self._canvas = self._chart
        layout.addWidget(self._chart, stretch=1)
        self._update_figure_size()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_figure_size()

    def _update_figure_size(self):
        dpi = self.devicePixelRatio() * 100
        w = self.width() / dpi
        self._fig.set_size_inches(max(w * 0.5, 1), 1.2)

    @property
    def tmin(self) -> float:
        return self._tmin.value()

    @property
    def tmax(self) -> float:
        return self._tmax.value()

    @property
    def reject_uv(self) -> float:
        return self._reject.value()

    @property
    def event_source(self) -> str:
        return self._event_source.currentText()

    @property
    def event_id(self) -> Optional[dict[str, int]]:
        text = self._event_id.text().strip()
        if not text:
            return None
        mapping = {}
        for pair in text.split(','):
            pair = pair.strip()
            if ':' in pair:
                name, code = pair.split(':', 1)
                mapping[name.strip()] = int(code.strip())
        return mapping or None

    def refresh_chart(self, pipeline: Optional[object] = None):
        ax = self._fig.axes[0]
        ax.clear()
        ax.set_facecolor('#1e1e1e')
        for spine in ax.spines.values():
            spine.set_color('#444')
        try:
            epochs = pipeline.epochs if pipeline is not None else None
            if epochs is None:
                ax.text(0.5, 0.5, "Run pipeline to see epochs",
                        transform=ax.transAxes, ha='center', va='center',
                        color='#555')
            else:
                evoked = epochs.average()
                t = evoked.times
                d = evoked.data * 1e6
                n_ch = min(8, d.shape[0])
                for i in range(n_ch):
                    ax.plot(t, d[i] + i * 20, linewidth=0.3, color='#00ff88')
                ax.set_title(f"ERP average — {len(epochs)} epochs",
                             color='white', fontsize=8)
                ax.set_xlabel("Time (s)", color='white', fontsize=7)
                ax.set_yticks([i * 20 for i in range(n_ch)])
                ax.set_yticklabels(
                    evoked.ch_names[:n_ch] if hasattr(evoked, 'ch_names') else [f'Ch {i}' for i in range(n_ch)],
                    fontsize=6)
                ax.tick_params(colors='white', labelsize=6)
        except Exception:
            pass
        self._canvas.draw_idle()

    @staticmethod
    def _make_chart() -> FigureCanvasQTAgg:
        fig = Figure(facecolor='#1e1e1e')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#1e1e1e')
        ax.tick_params(colors='white', labelsize=6)
        for spine in ax.spines.values():
            spine.set_color('#444')
        canvas = FigureCanvasQTAgg(fig)
        return canvas

    @property
    def _fig(self):
        return self._chart.figure
