"""
EEG Waveform Widget
===================
Matplotlib-based scrolling EEG display supporting batch and stream modes.
Multi-channel: tiles into columns of N channels each for readability.
"""
from __future__ import annotations
import math
import numpy as np
from typing import List, Optional
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class EEGWaveformWidget(FigureCanvasQTAgg):
    """EEG waveform display with batch and streaming modes.

    plot_batch():  static view of entire dataset
    update_stream(): append chunk to rolling buffer

    Many channels → automatically tile horizontally into columns
    (ch_per_plot channels per column), 1 row tall.
    """

    def __init__(self, parent=None, ch_per_plot: int = 8):
        self._ch_per_plot = max(1, ch_per_plot)
        self.fig = Figure(facecolor='#1e1e1e')
        super().__init__(self.fig)
        self.setParent(parent)
        self._axes: list = []
        self._buffer = None
        self._ch_names: List[str] = []
        self._sfreq = 256.0
        self._window_samples = 0
        self._yscale = 50.0
        self._n_cols = 0
        self._clear_figure()

    # ----------------------------------------------------------------
    # Layout
    # ----------------------------------------------------------------

    def _clear_figure(self):
        self.fig.clear()
        self._axes = []

    def _rebuild_axes(self, n_ch: int):
        """Create 1×N subplot grid: one column per `ch_per_plot` channels."""
        self._clear_figure()
        self._n_cols = math.ceil(n_ch / self._ch_per_plot)
        self.fig.subplots_adjust(
            left=0.06, right=0.98, bottom=0.15, top=0.95,
            wspace=0.08, hspace=0,
        )

        for col in range(self._n_cols):
            ax = self.fig.add_subplot(1, self._n_cols, col + 1)
            ax.set_facecolor('#1e1e1e')
            self._axes.append(ax)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_figure_size()

    def _update_figure_size(self):
        dpi = 100
        w = self.width() / dpi
        h = self.height() / dpi
        self.fig.set_size_inches(w, h)

    # ----------------------------------------------------------------
    # Channel range helpers
    # ----------------------------------------------------------------

    def _col_slice(self, col: int, n_ch: int):
        a = col * self._ch_per_plot
        b = min(a + self._ch_per_plot, n_ch)
        return a, b

    def _draw_col(self, col: int, data: np.ndarray, sfreq: float,
                  n_ch: int):
        ax = self._axes[col]
        ax.clear()
        a, b = self._col_slice(col, n_ch)
        t = np.arange(data.shape[1]) / sfreq
        data_uv = data * 1e6  # V → μV
        n_in_col = b - a

        for i in range(n_in_col):
            ax.plot(t, data_uv[i] + i * self._yscale,
                    linewidth=0.3, color='#00ff88')

        ax.tick_params(colors='white', labelsize=7)
        ax.set_xlabel('Time (s)', color='white', fontsize=8)
        ax.set_ylim(-self._yscale * 0.5, (n_in_col - 0.5) * self._yscale)

        tick_positions = [i * self._yscale for i in range(n_in_col)]
        tick_labels = []
        for ch_idx in range(a, b):
            if ch_idx < len(self._ch_names):
                tick_labels.append(self._ch_names[ch_idx])
            else:
                tick_labels.append(f'Ch {ch_idx}')
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=6)

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def _init_buffer(self, n_channels: int, sfreq: float,
                     ch_names: List[str], window_sec: float = 5.0):
        self._sfreq = sfreq
        self._ch_names = ch_names
        self._window_samples = int(window_sec * sfreq)
        self._buffer = np.zeros((n_channels, self._window_samples))
        self._yscale = 50.0
        self._rebuild_axes(n_channels)

    def update_stream(self, chunk: np.ndarray):
        if self._buffer is None:
            raise RuntimeError("Call _init_buffer() before update_stream()")
        n_ch, n_new = chunk.shape
        if n_ch != self._buffer.shape[0]:
            raise ValueError(
                f"Expected {self._buffer.shape[0]} channels, got {n_ch}"
            )
        self._buffer = np.roll(self._buffer, -n_new, axis=1)
        self._buffer[:, -n_new:] = chunk

        for col in range(self._n_cols):
            a, b = self._col_slice(col, n_ch)
            self._draw_col(col, self._buffer[a:b, :], self._sfreq, n_ch)
        self.draw_idle()

    def plot_batch(self, data: np.ndarray, sfreq: float,
                   ch_names: Optional[List[str]] = None):
        if ch_names is not None:
            self._ch_names = ch_names
        n_ch, _ = data.shape
        self._rebuild_axes(n_ch)

        for col in range(self._n_cols):
            a, b = self._col_slice(col, n_ch)
            self._draw_col(col, data[a:b, :], sfreq, n_ch)
        self.draw_idle()

    def plot_batch_window(self, data: np.ndarray, sfreq: float,
                          ch_names: Optional[List[str]], t_start: float,
                          window_sec: float = 5.0):
        if ch_names is not None:
            self._ch_names = ch_names
        n_ch = data.shape[0]
        start_sample = int(t_start * sfreq)
        window_samples = int(window_sec * sfreq)
        end_sample = min(start_sample + window_samples, data.shape[1])
        window_data = data[:, start_sample:end_sample]

        self._rebuild_axes(n_ch)
        self.fig.subplots_adjust(bottom=0.18)

        for col in range(self._n_cols):
            a, b = self._col_slice(col, n_ch)
            self._draw_col(col, window_data[a:b, :], sfreq, n_ch)
        self.draw_idle()

    def clear(self):
        self._clear_figure()
        self._buffer = None
        self.draw_idle()
