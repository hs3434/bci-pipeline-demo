"""
Session Loader — File Dialog + Auto-Detect + Confirmation
==========================================================
Orchestrates the complete file loading flow: file dialog with
multi-select, auto-detection of same-subject runs, and a
confirmation dialog for selecting/deselecting runs.
"""
from __future__ import annotations
from typing import List, Optional
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QFileDialog, QWidget, QDialogButtonBox,
)
from PyQt6.QtCore import Qt
import re
import glob as glob_lib


class SessionDialog(QDialog):
    """Confirmation dialog for session run selection.

    Shows detected runs with checkboxes, summary info, and
    confirm/cancel buttons.
    """

    def __init__(self, detected_runs: List[Path], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._runs = detected_runs
        self._setup_ui()

    def _setup_ui(self):
        session_name = self._runs[0].stem
        match = __import__('re').match(r'^(.*)R\d+$', session_name)
        if match:
            session_name = match.group(1)

        self.setWindowTitle(f"Session: {session_name} — 检测到 {len(self._runs)} 个 run")
        self.setMinimumWidth(400)
        layout = QVBoxLayout(self)

        btn_row = QHBoxLayout()
        select_all_btn = QPushButton("全选")
        select_all_btn.clicked.connect(self._select_all)
        btn_row.addWidget(select_all_btn)

        deselect_all_btn = QPushButton("取消全选")
        deselect_all_btn.clicked.connect(self._deselect_all)
        btn_row.addWidget(deselect_all_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._list_widget = QListWidget()
        for run_path in self._runs:
            item = QListWidgetItem(run_path.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setData(Qt.ItemDataRole.UserRole, str(run_path))
            item.setCheckState(Qt.CheckState.Checked)
            self._list_widget.addItem(item)
        self._list_widget.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._list_widget)

        n_runs = len(self._runs)
        n_ch, sfreq, n_samples = self._read_metadata(self._runs[0])
        if n_runs > 1:
            n_samples *= n_runs
        self._info_label = QLabel(
            f"📊 {n_runs} runs · {n_ch} channels · {sfreq} Hz · {n_samples} samples"
        )
        self._info_label.setStyleSheet("color: #888; padding: 4px;")
        layout.addWidget(self._info_label)

        button_box = QDialogButtonBox()
        self._confirm_btn = QPushButton(f"确定 (合并 {n_runs} 个)")
        self._confirm_btn.clicked.connect(self.accept)
        self._confirm_btn.setDefault(True)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        button_box.addButton(cancel_btn, QDialogButtonBox.ButtonRole.RejectRole)
        button_box.addButton(self._confirm_btn, QDialogButtonBox.ButtonRole.AcceptRole)
        layout.addWidget(button_box)

    @staticmethod
    def _read_metadata(filepath: Path):
        import mne
        raw = mne.io.read_raw(filepath, preload=False, verbose=False)
        n_samples = raw.n_times
        return raw.info['nchan'], int(raw.info['sfreq']), n_samples

    def _select_all(self):
        for i in range(self._list_widget.count()):
            self._list_widget.item(i).setCheckState(Qt.CheckState.Checked)

    def _deselect_all(self):
        for i in range(self._list_widget.count()):
            self._list_widget.item(i).setCheckState(Qt.CheckState.Unchecked)

    def _on_item_changed(self):
        n = self.checked_count()
        self._confirm_btn.setText(f"确定 (合并 {n} 个)")
        self._confirm_btn.setEnabled(n > 0)

    def checked_count(self) -> int:
        count = 0
        for i in range(self._list_widget.count()):
            if self._list_widget.item(i).checkState() == Qt.CheckState.Checked:
                count += 1
        return count

    def selected_runs(self) -> List[Path]:
        runs = []
        for i in range(self._list_widget.count()):
            item = self._list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                runs.append(Path(item.data(Qt.ItemDataRole.UserRole)))
        return runs


def open_session_files(parent: QWidget) -> List[Path]:
    """Orchestrate file loading: dialog → auto-detect → confirmation.

    Flow:
    1. Open QFileDialog with multi-select and EEG filters
    2. If 1 file selected → auto-detect sibling runs → confirm
    3. If N>1 files selected → use as-is (no auto-detect)
    4. Cancel → return empty list
    """
    dialog = QFileDialog(parent, "Select EEG File(s)")
    dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
    dialog.setNameFilters([
        "EEG Files (*.edf *.fif *.set *.vhdr)",
        "All Files (*)",
    ])
    if dialog.exec() != QFileDialog.DialogCode.Accepted:
        return []

    selected = [Path(p) for p in dialog.selectedFiles()]
    if not selected:
        return []

    if len(selected) == 1:
        runs = find_session_runs(selected[0])
        if len(runs) > 1:
            confirm = SessionDialog(runs, parent)
            if confirm.exec() == QDialog.DialogCode.Accepted:
                return confirm.selected_runs()
            return []
        return runs

    return selected


def find_session_runs(filepath: Path) -> List[Path]:
    """Discover all runs belonging to the same subject session.

    Given ``S001R04.edf``, glob for ``S001R*.edf`` in the same
    directory and sort by ascending run number.
    """
    stem = filepath.stem
    match = re.match(r'^(.*R)0?(\d+)$', stem)
    if match is None:
        return [filepath]

    base = match.group(1)
    ext = filepath.suffix
    pattern = f"{filepath.parent}/{base}*{ext}"
    runs = sorted(
        glob_lib.glob(pattern),
        key=lambda p: int(re.search(r'R(\d+)', p).group(1)),
    )
    return [Path(p) for p in runs]
