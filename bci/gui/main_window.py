"""
Main Window
===========
Tabbed BCI viewer: offline analysis + real-time viewing.
"""
from __future__ import annotations
import sys

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QStatusBar, QMessageBox, QApplication,
)
from PyQt6.QtGui import QCloseEvent

from bci.gui.batch_tab import BatchTab
from bci.gui.stream_tab import StreamTab


class BCIMainWindow(QMainWindow):
    """BCI Signal Viewer — dual-mode analysis system.

    Tab 1: 离线分析 (batch processing)
    Tab 2: 实时查看 (simulated real-time streaming)
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("BCI Signal Viewer")
        self.setMinimumSize(1200, 800)
        self._setup_menu()
        self._setup_tabs()
        self._setup_status()

    def closeEvent(self, event: QCloseEvent):
        self.batch_tab.shutdown()
        self.stream_tab.shutdown()
        event.accept()

    def _setup_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        load_action = file_menu.addAction("Load EEG File...")
        load_action.triggered.connect(self._on_load)
        load_action.setShortcut("Ctrl+O")
        file_menu.addSeparator()
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)
        exit_action.setShortcut("Ctrl+Q")

        help_menu = menubar.addMenu("Help")
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self._on_about)

    def _setup_tabs(self):
        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        self.batch_tab = BatchTab()
        self._tabs.addTab(self.batch_tab, "离线分析")

        self.stream_tab = StreamTab()
        self._tabs.addTab(self.stream_tab, "实时查看")

        self._tabs.currentChanged.connect(self._on_tab_changed)

    def _setup_status(self):
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — Load an EEG file to begin")

    def _on_load(self):
        from bci.gui.session_loader import open_session_files
        filepaths = open_session_files(self)
        if not filepaths:
            return

        current = self._tabs.currentWidget()
        if current == self.batch_tab:
            self.batch_tab._on_files_loaded([str(p) for p in filepaths])
        else:
            self.stream_tab._on_files_loaded([str(p) for p in filepaths])

    def _on_tab_changed(self, index: int):
        names = ["离线分析", "实时查看"]
        self._status.showMessage(f"Switched to: {names[index]}")

    def _on_about(self):
        QMessageBox.about(
            self, "About BCI Signal Viewer",
            "BCI Signal Viewer v1.0\n\n"
            "Dual-mode EEG analysis system:\n"
            "• 离线分析 — Batch pipeline processing\n"
            "• 实时查看 — Simulated real-time streaming\n\n"
            "Built with PyQt6 + MNE + Matplotlib"
        )


def _setup_chinese_fonts():
    """Configure Qt and matplotlib for Chinese character support."""
    from PyQt6.QtGui import QFont
    chinese_fonts = [
        'WenQuanYi Micro Hei', 'Noto Sans CJK SC',
        'WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei',
    ]
    available = QFont()
    default_family = available.defaultFamily()
    matched = None
    for name in chinese_fonts:
        font = QFont(name)
        if font.exactMatch():
            matched = name
            break

    if matched is None:
        return default_family

    QApplication.instance().setFont(QFont(matched, 10))
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = [matched, 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    return matched


def main():
    """GUI entry point."""
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QFont
    except ImportError:
        print("Error: PyQt6 is required for GUI", file=sys.stderr)
        print("Install with: pip install PyQt6", file=sys.stderr)
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    font_family = _setup_chinese_fonts()
    print(f"Using font: {font_family}", file=sys.stderr)

    window = BCIMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
