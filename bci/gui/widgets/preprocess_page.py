"""
Preprocess Page
===============
Filter parameter configuration + PSD before/after comparison chart.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QGroupBox, QDoubleSpinBox,
)
from PyQt6.QtCore import pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class PreprocessPage(QFrame):

    filter_changed = pyqtSignal()

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
        self._l_freq.valueChanged.connect(self.filter_changed.emit)
        glay.addWidget(self._l_freq)
        glay.addWidget(QLabel("Highcut:"))
        self._h_freq = QDoubleSpinBox()
        self._h_freq.setRange(10, 100)
        self._h_freq.setValue(40)
        self._h_freq.setSuffix(" Hz")
        self._h_freq.valueChanged.connect(self.filter_changed.emit)
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

    def refresh_chart(self, data: Optional['mne.io.Raw'] = None):
        ax = self._fig.axes[0]
        ax.clear()
        ax.set_facecolor('#1e1e1e')
        for spine in ax.spines.values():
            spine.set_color('#444')
        try:
            if data is None:
                ax.text(0.5, 0.5, "No data loaded", transform=ax.transAxes,
                        ha='center', va='center', color='#555')
            else:
                sfreq = data.info['sfreq']
                n_fft = min(2048, data.n_times)
                psd, freqs = data.compute_psd(
                    fmax=min(data.info['sfreq'] / 2, 80),
                    n_fft=n_fft, verbose=False).get_data(return_freqs=True)

                psd_db = 10 * np.log10(psd ** 2 + 1e-20)
                mean_psd = psd_db.mean(axis=0)

                ax.plot(freqs, mean_psd, linewidth=0.8, color='#5588ff',
                        alpha=0.6, label='Before')

                l_freq = self.l_freq
                h_freq = self.h_freq
                mask = (freqs >= l_freq) & (freqs <= h_freq)
                ax.fill_between(freqs, mean_psd.min(), mean_psd.max(),
                                where=mask, color='#00ff88', alpha=0.08)
                ax.plot(freqs[mask], mean_psd[mask], linewidth=1.0,
                        color='#00ff88', label='Passband')

                ax.axvline(l_freq, color='#ff6644', linewidth=0.8,
                           linestyle='--', alpha=0.7)
                ax.axvline(h_freq, color='#ff6644', linewidth=0.8,
                           linestyle='--', alpha=0.7)

                ax.set_title(
                    f"PSD — {l_freq:.1f}–{h_freq:.0f} Hz",
                    color='white', fontsize=8)
                ax.set_xlabel("Frequency (Hz)", color='white', fontsize=7)
                ax.set_ylabel("dB", color='white', fontsize=7)
                ax.legend(fontsize=6, facecolor='#1e1e1e', edgecolor='none',
                          labelcolor='white', loc='upper right')
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
