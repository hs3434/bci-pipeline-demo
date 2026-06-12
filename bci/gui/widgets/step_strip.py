"""
Step Strip
==========
Horizontal pipeline step indicator with clickable labels and rerun button.
"""
from __future__ import annotations
from typing import List
from enum import Enum

from PyQt6.QtWidgets import (
    QHBoxLayout, QPushButton, QLabel, QFrame
)
from PyQt6.QtCore import pyqtSignal


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    STALE = "stale"
    ERROR = "error"


class StepStrip(QFrame):
    """Pipeline step indicator bar."""

    step_clicked = pyqtSignal(int)
    rerun_clicked = pyqtSignal()

    STYLES = {
        StepStatus.PENDING: "color: #555; padding: 4px 16px; font-size: 12px;",
        StepStatus.RUNNING: "color: #ffaa00; padding: 4px 16px; font-size: 12px; font-weight: bold;",
        StepStatus.DONE: "color: #00cc66; padding: 4px 16px; font-size: 12px;",
        StepStatus.STALE: "color: #cc6600; padding: 4px 16px; font-size: 12px;",
        StepStatus.ERROR: "color: #ff4444; padding: 4px 16px; font-size: 12px;",
    }

    ACTIVE_STYLE = (
        "background-color: #333; border-bottom: 2px solid #00cc66; "
        "border-radius: 2px;"
    )

    SYMBOLS = {
        StepStatus.PENDING: "○",
        StepStatus.RUNNING: "◉",
        StepStatus.DONE: "●",
        StepStatus.STALE: "◐",
        StepStatus.ERROR: "✕",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet("background-color: #2a2a2a; border-radius: 4px;")
        self.setMaximumHeight(36)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(4)

        self._step_names = ["Main", "Preprocess", "Epoch", "Decode"]
        self._statuses = [StepStatus.PENDING] * 4
        self._active_idx = 0
        self._labels: List[QLabel] = []
        self._arrows: List[QLabel] = []

        for i, name in enumerate(self._step_names):
            label = QLabel(f"{self.SYMBOLS[StepStatus.PENDING]} {name}")
            label.setStyleSheet(self.STYLES[StepStatus.PENDING])
            idx = i
            label.mousePressEvent = lambda e, n=idx: self._on_click(n)  # type: ignore[assignment]  # Qt lambda override
            self._labels.append(label)
            layout.addWidget(label)

            if i < len(self._step_names) - 1:
                arrow = QLabel("→")
                arrow.setStyleSheet("color: #444; padding: 4px 6px; font-size: 14px;")
                self._arrows.append(arrow)
                layout.addWidget(arrow)

        self._rerun_btn = QPushButton("↻ Rerun")
        self._rerun_btn.setStyleSheet(
            "QPushButton { color: #00cc66; background: #333; border: 1px solid #555; "
            "border-radius: 3px; padding: 3px 12px; font-size: 11px; } "
            "QPushButton:hover { background: #444; } "
            "QPushButton:disabled { color: #555; }"
        )
        self._rerun_btn.clicked.connect(self.rerun_clicked.emit)
        layout.addWidget(self._rerun_btn)
        layout.addStretch()

        self.set_active(0)

    def _on_click(self, idx: int):
        self.set_active(idx)
        self.step_clicked.emit(idx)

    def set_active(self, idx: int):
        for i, label in enumerate(self._labels):
            base = self.STYLES[self._statuses[i]]
            if i == idx:
                label.setStyleSheet(base + self.ACTIVE_STYLE)
            else:
                label.setStyleSheet(base)
        self._active_idx = idx

    # ---- public API ----

    def set_status(self, idx: int, status: StepStatus):
        self._statuses[idx] = status
        sym = self.SYMBOLS[status]
        name = self._step_names[idx]
        self._labels[idx].setText(f"{sym} {name}")
        self._labels[idx].setStyleSheet(self.STYLES[status])

    def set_all_pending(self):
        for i in range(4):
            self.set_status(i, StepStatus.PENDING)

    def set_all_stale_from(self, idx: int):
        for i in range(idx, 4):
            self.set_status(i, StepStatus.STALE)

    def mark_error(self, idx: int):
        self.set_status(idx, StepStatus.ERROR)
        for i in range(idx + 1, 4):
            self.set_status(i, StepStatus.PENDING)

    def current_step(self) -> int:
        for i, s in enumerate(self._statuses):
            if s in (StepStatus.RUNNING, StepStatus.PENDING):
                return i
        return 4
