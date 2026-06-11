"""
Decode Page
===========
Decode method configuration + CV fold results chart.
"""
from __future__ import annotations
from typing import Optional, List

import numpy as np
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QGroupBox, QComboBox, QSpinBox,
)
from PyQt6.QtCore import pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from bci.decoder import list_methods


class DecodePage(QFrame):

    decode_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)

        grp = QGroupBox("Decode Parameters")
        glay = QHBoxLayout()
        glay.addWidget(QLabel("Method:"))
        self._method = QComboBox()
        self._method.addItems(list_methods())
        self._method.currentTextChanged.connect(self.decode_changed.emit)
        glay.addWidget(self._method)
        glay.addWidget(QLabel("CV Folds:"))
        self._cv_folds = QSpinBox()
        self._cv_folds.setRange(2, 10)
        self._cv_folds.setValue(5)
        self._cv_folds.valueChanged.connect(self.decode_changed.emit)
        glay.addWidget(self._cv_folds)
        glay.addStretch()
        grp.setLayout(glay)
        layout.addWidget(grp)

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
        self._fig.set_size_inches(max(w * 0.5, 1), 1.5)

    @property
    def method(self) -> str:
        return self._method.currentText()

    @property
    def cv_folds(self) -> int:
        return self._cv_folds.value()

    def show_result(self, accuracy: float, std: float,
                    cv_scores: Optional[List[float]] = None,
                    method: str = ""):
        ax = self._fig.axes[0]
        ax.clear()
        ax.set_facecolor('#1e1e1e')
        for spine in ax.spines.values():
            spine.set_color('#444')

        if cv_scores and len(cv_scores) > 0:
            folds = range(1, len(cv_scores) + 1)
            ax.bar(folds, cv_scores, color='#00cc66', edgecolor='#00994d', alpha=0.85)
            mean = np.mean(cv_scores)
            ax.axhline(y=mean, color='#ff6644', linewidth=1, linestyle='--',
                       label=f'Mean: {mean:.3f}')
            ax.legend(fontsize=7, facecolor='#1e1e1e', edgecolor='none',
                      labelcolor='#ff6644')
            ax.set_ylim(max(0, min(cv_scores) - 0.05), min(1, max(cv_scores) + 0.05))
            ax.set_xlabel("CV Fold", color='white', fontsize=7)
        else:
            ax.bar([1], [accuracy], color='#00cc66', alpha=0.85)
            ax.set_ylim(0, 1)

        ax.set_title(f"{method} — {accuracy:.3f} ± {std:.3f}",
                     color='white', fontsize=8)
        ax.set_ylabel("Accuracy", color='white', fontsize=7)
        ax.tick_params(colors='white', labelsize=6)
        self._fig.set_facecolor('#1e1e1e')
        self._canvas.draw_idle()

    def refresh_chart(self):
        ax = self._fig.axes[0]
        ax.clear()
        ax.set_facecolor('#1e1e1e')
        ax.text(0.5, 0.5, "Run pipeline\nto see results",
                transform=ax.transAxes, ha='center', va='center', color='#555')
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
