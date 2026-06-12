"""
Epoch Page
==========
Epoch parameter + event config + butterfly ERP + rejection stats chart.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from bci.pipeline import BCIPipeline
from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QDoubleSpinBox,
    QComboBox, QLineEdit,
)
from PyQt6.QtCore import pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec


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

        right = QVBoxLayout()
        right.setSpacing(2)

        self._info_label = QLabel()
        self._info_label.setStyleSheet("color: #aaa; font-size: 10px; padding: 2px;")
        self._info_label.setWordWrap(True)
        right.addWidget(self._info_label)

        self._chart = self._make_chart()
        self._canvas = self._chart
        right.addWidget(self._chart, stretch=1)
        layout.addLayout(right, stretch=1)
        self._update_figure_size()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_figure_size()

    def _update_figure_size(self):
        dpi = self.devicePixelRatio() * 100
        w = self.width() / dpi
        self._fig.set_size_inches(max(w * 0.5, 1), 2.8)

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

    def refresh_chart(self, data: Optional[BCIPipeline] = None):
        self._fig.clear()
        gs = gridspec.GridSpec(2, 1, figure=self._fig, height_ratios=[3, 1],
                              hspace=0.3)
        ax_erp = self._fig.add_subplot(gs[0, 0])
        ax_rej = self._fig.add_subplot(gs[1, 0])
        self._fig.set_facecolor('#1e1e1e')

        try:
            epochs = data.epochs if data is not None else None
            if epochs is None or len(epochs) == 0:
                self._info_label.setText("")
                for ax in (ax_erp, ax_rej):
                    ax.set_facecolor('#1e1e1e')
                    ax.text(0.5, 0.5, "Run pipeline to see epochs",
                            transform=ax.transAxes, ha='center', va='center',
                            color='#555')
                    for spine in ax.spines.values():
                        spine.set_color('#444')
                    ax.tick_params(colors='white', labelsize=6)
            else:
                self._draw_butterfly(ax_erp, epochs)
                self._draw_rejection(ax_rej, epochs)
                self._update_info(epochs)
        except Exception:
            pass
        self._canvas.draw_idle()

    def _draw_butterfly(self, ax, epochs) -> None:
        ax.set_facecolor('#1e1e1e')
        for spine in ax.spines.values():
            spine.set_color('#444')
        evoked = epochs.average()
        t = evoked.times
        n_ch = min(8, evoked.data.shape[0])

        trials = epochs.get_data(copy=False)[:, :n_ch, :]
        for trial in trials:
            for i in range(n_ch):
                ax.plot(t, trial[i] * 1e6 + i * 20, linewidth=0.1,
                        color='#336644', alpha=0.15)

        for i in range(n_ch):
            ax.plot(t, evoked.data[i] * 1e6 + i * 20, linewidth=0.5,
                    color='#00ff88')
            ax.axhline(i * 20, color='#444', linewidth=0.3)

        yticks = [i * 20 for i in range(n_ch)]
        ylabels = evoked.ch_names[:n_ch] if hasattr(evoked, 'ch_names') else [f'Ch {i}' for i in range(n_ch)]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=5)
        ax.set_title(f"ERP — {len(epochs)} epochs × {n_ch}/{evoked.data.shape[0]} ch",
                     color='white', fontsize=8)
        ax.set_xlabel("Time (s)", color='white', fontsize=7)
        ax.tick_params(colors='white', labelsize=6)
        ax.axvline(0, color='#ff6644', linewidth=0.5, linestyle='--', alpha=0.5)

    def _draw_rejection(self, ax, epochs) -> None:
        ax.set_facecolor('#1e1e1e')
        for spine in ax.spines.values():
            spine.set_color('#444')

        drop_log = epochs.drop_log
        total = len(drop_log)
        if total == 0:
            ax.text(0.5, 0.5, "No epochs to analyze",
                    transform=ax.transAxes, ha='center', va='center',
                    color='#555')
            ax.tick_params(colors='white', labelsize=6)
            return

        kept = sum(1 for d in drop_log if not d)
        dropped = total - kept

        labels = ['Kept', 'Dropped']
        counts = [kept, dropped]
        colors = ['#00ff88', '#ff6644']
        bars = ax.bar(labels, counts, color=colors, alpha=0.85, width=0.5)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(count), ha='center', va='bottom', color='white', fontsize=7)

        ax.set_title(f"Rejection — {dropped}/{total} dropped ({dropped/max(1,total)*100:.0f}%)",
                     color='white', fontsize=8)
        ax.tick_params(colors='white', labelsize=6)
        ax.set_ylim(0, max(counts) * 1.15)
        ax.set_ylabel("Count", color='white', fontsize=7)

    def _update_info(self, epochs) -> None:
        event_ids = epochs.event_id if hasattr(epochs, 'event_id') else {}
        parts = []
        if event_ids:
            ids_str = ', '.join(f'{k}={v}' for k, v in sorted(event_ids.items()))
            parts.append(f"IDs: {ids_str}")
        parts.append(f"Events: {len(epochs.events)}")
        n_dropped = sum(1 for d in epochs.drop_log if d)
        parts.append(f"Kept: {len(epochs)}  Dropped: {n_dropped}")
        self._info_label.setText('  |  '.join(parts))

    @staticmethod
    def _make_chart() -> FigureCanvasQTAgg:
        fig = Figure(facecolor='#1e1e1e')
        canvas = FigureCanvasQTAgg(fig)
        return canvas

    @property
    def _fig(self):
        return self._chart.figure
