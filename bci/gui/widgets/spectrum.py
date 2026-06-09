"""
Spectrum Widget
===============
Real-time PSD (Power Spectral Density) display.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import welch
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class SpectrumWidget(FigureCanvasQTAgg):
    """Real-time power spectral density display."""

    def __init__(self, parent=None):
        self.fig = Figure(facecolor='#1e1e1e')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')
        super().__init__(self.fig)
        self.setParent(parent)
        self._update_figure_size()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_figure_size()

    def _update_figure_size(self):
        dpi = self.devicePixelRatio() * 100
        w = self.width() / dpi
        h = self.height() / dpi
        self.fig.set_size_inches(max(w, 1), max(h, 1))

    def update_psd(self, data: np.ndarray, sfreq: float):
        """Update PSD from chunk or full signal."""
        self.ax.clear()
        for ch in range(min(data.shape[0], 8)):
            freqs, psd = welch(data[ch], sfreq, nperseg=min(256, data.shape[1]))
            self.ax.semilogy(freqs, psd, linewidth=0.5,
                             alpha=0.7, label=f'Ch {ch}')
        self.ax.set_xlabel('Frequency (Hz)', color='white', fontsize=8)
        self.ax.set_ylabel('PSD', color='white', fontsize=8)
        self.ax.tick_params(colors='white', labelsize=7)
        self.ax.set_xlim(0, min(sfreq / 2, 80))
        self.draw_idle()
