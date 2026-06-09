"""
Topomap Widget
==============
Scalp topography display using MNE's plotting capabilities.
"""
from __future__ import annotations
import numpy as np
from typing import List
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class TopomapWidget(FigureCanvasQTAgg):
    """Scalp topography visualization."""

    def __init__(self, parent=None):
        self.fig = Figure(facecolor='#1e1e1e')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')
        super().__init__(self.fig)
        self.setParent(parent)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_figure_size()

    def _update_figure_size(self):
        dpi = self.devicePixelRatio() * 100
        size = min(self.width(), self.height()) / dpi
        self.fig.set_size_inches(size, size)

    def update_topo(self, data: np.ndarray, ch_names: List[str]):
        """Update topomap with channel data."""
        try:
            import mne
            info = mne.create_info(
                ch_names=ch_names[:len(data)], sfreq=256.0,
                ch_types=['eeg'] * min(len(data), len(ch_names))
            )
            info.set_montage('standard_1020')
            self.ax.clear()
            mne.viz.plot_topomap(data, info, axes=self.ax, show=False)
            self.draw_idle()
        except Exception:
            self.show_fallback("Unable to render topomap")

    def show_fallback(self, message: str):
        """Display fallback message when topomap can't render."""
        self.ax.clear()
        self.ax.text(0.5, 0.5, message, transform=self.ax.transAxes,
                     ha='center', va='center', color='#888888', fontsize=10)
        self.ax.set_facecolor('#1e1e1e')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.draw_idle()
