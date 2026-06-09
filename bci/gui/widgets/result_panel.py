"""
Result Panel
============
Displays decoding results: accuracy bar chart + text summary.
"""
from __future__ import annotations
from typing import Optional, List
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTextEdit
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class ResultPanel(QWidget):
    """Decoding results display with bar chart.

    Batch mode: CV fold scores as bar chart + text summary.
    Stream mode: real-time prediction label.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._title = QLabel("Results")
        self._title.setStyleSheet("color: white; font-weight: bold;")
        layout.addWidget(self._title)

        fig = Figure(facecolor='#1e1e1e')
        self._ax = fig.add_subplot(111)
        self._ax.set_facecolor('#1e1e1e')
        self._canvas = FigureCanvasQTAgg(fig)
        self._fig = fig
        layout.addWidget(self._canvas, stretch=1)

        self._summary = QTextEdit()
        self._summary.setReadOnly(True)
        self._summary.setStyleSheet(
            "background-color: #2d2d2d; color: #00ff88; font-family: monospace;"
        )
        layout.addWidget(self._summary)

        self._prediction_label = QLabel("Prediction: --")
        self._prediction_label.setStyleSheet("color: #88ccff; font-size: 14px;")
        self._prediction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._prediction_label)

        self._show_empty()
        self._update_figure_size()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_figure_size()

    def _update_figure_size(self):
        dpi = self.devicePixelRatio() * 100
        w = self.width() / dpi
        h = max(self.height() - 40, 20) / dpi
        self._fig.set_size_inches(max(w, 1), max(h, 0.5))

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def update_batch(self, accuracy: float, std: float,
                     cv_scores: Optional[List[float]] = None,
                     method: str = ""):
        """Display batch decoding results with bar chart."""
        self._ax.clear()
        self._ax.set_facecolor('#1e1e1e')

        if cv_scores and len(cv_scores) > 0:
            folds = range(1, len(cv_scores) + 1)
            self._ax.bar(folds, cv_scores,
                                color='#00cc66', edgecolor='#00994d',
                                alpha=0.85)
            mean = np.mean(cv_scores)
            self._ax.axhline(y=mean, color='#ff6644', linewidth=1,
                             linestyle='--', label=f'Mean: {mean:.3f}')
            self._ax.legend(fontsize=7, facecolor='#1e1e1e',
                            edgecolor='none', labelcolor='#ff6644')
            self._ax.set_ylim(max(0, min(cv_scores) - 0.05), min(1, max(cv_scores) + 0.05))
        else:
            self._ax.bar([1], [accuracy], color='#00cc66', alpha=0.85)
            self._ax.set_ylim(0, 1)

        self._ax.tick_params(colors='white', labelsize=7)
        self._ax.set_ylabel('Accuracy', color='white', fontsize=8)
        self._ax.set_xlabel('CV Fold', color='white', fontsize=8)
        self._ax.spines['bottom'].set_color('#444')
        self._ax.spines['left'].set_color('#444')
        self._ax.spines['top'].set_visible(False)
        self._ax.spines['right'].set_visible(False)
        self._canvas.draw_idle()

        lines = [f"Method: {method}", f"Accuracy: {accuracy:.3f} ± {std:.3f}"]
        if cv_scores:
            lines.append("Scores: " + ", ".join(f"{s:.3f}" for s in cv_scores))
        self._summary.setText("\n".join(lines))
        self._prediction_label.setText("")

    def update_stream(self, label: str):
        self._ax.clear()
        self._ax.set_facecolor('#1e1e1e')
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        for spine in self._ax.spines.values():
            spine.set_visible(False)
        self._canvas.draw_idle()
        self._summary.clear()
        self._prediction_label.setText(f"Prediction: {label}")

    def clear(self):
        self._show_empty()

    # ----------------------------------------------------------------
    # Internal
    # ----------------------------------------------------------------

    def _show_empty(self):
        self._ax.clear()
        self._ax.set_facecolor('#1e1e1e')
        self._ax.text(0.5, 0.5, "Run pipeline\nto see results",
                      transform=self._ax.transAxes,
                      ha='center', va='center', color='#555', fontsize=10)
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._ax.spines['top'].set_visible(False)
        self._ax.spines['right'].set_visible(False)
        self._ax.spines['bottom'].set_visible(False)
        self._ax.spines['left'].set_visible(False)
        self._canvas.draw_idle()
        self._summary.clear()
        self._prediction_label.setText("Prediction: --")
