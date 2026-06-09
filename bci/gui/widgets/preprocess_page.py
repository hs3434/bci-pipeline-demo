"""
Preprocess Page
===============
Filter parameter configuration + raw data preview chart.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QGroupBox, QDoubleSpinBox,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class PreprocessPage(QFrame):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)

        grp = QGroupBox("Filter Parameters")
        glay = QHBoxLayout()
        glay.addWidget(QLabel("Lowcut:"))
        self._l_freq = QDoubleSpinBox()
        self._l_freq.setRange(0.1, 10)
        self._l_freq.setValue(0.5)
        self._l_freq.setSuffix(" Hz")
        glay.addWidget(self._l_freq)
        glay.addWidget(QLabel("Highcut:"))
        self._h_freq = QDoubleSpinBox()
        self._h_freq.setRange(10, 100)
        self._h_freq.setValue(40)
        self._h_freq.setSuffix(" Hz")
        glay.addWidget(self._h_freq)
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
        self._fig.set_size_inches(max(w * 0.5, 1), 1.2)

    @property
    def l_freq(self) -> float:
        return self._l_freq.value()

    @property
    def h_freq(self) -> float:
        return self._h_freq.value()

    def refresh_chart(self, source: Optional[object] = None):
        ax = self._fig.axes[0]
        ax.clear()
        ax.set_facecolor('#1e1e1e')
        for spine in ax.spines.values():
            spine.set_color('#444')
        try:
            if source is None or not source._data_list:
                ax.text(0.5, 0.5, "No data loaded", transform=ax.transAxes,
                        ha='center', va='center', color='#555')
            else:
                d = source._data_list[0]
                n_ch = min(8, d.shape[0])
                n_samples = min(500, d.shape[1])
                t = np.arange(n_samples) / source.sfreq
                for i in range(n_ch):
                    ax.plot(t, d[i, :n_samples] * 1e6 + i * 50,
                            linewidth=0.3, color='#00ff88')
                ax.set_title(f"Raw — first {n_ch} ch", color='white', fontsize=8)
                ax.set_xlabel("Time (s)", color='white', fontsize=7)
                ax.set_yticks([i * 50 for i in range(n_ch)])
                ax.set_yticklabels([f'Ch {i}' for i in range(n_ch)], fontsize=6)
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
