"""
Tests for bci.gui.session_loader module
=======================================
"""
from __future__ import annotations
import pytest
import os
import tempfile
import numpy as np
from pathlib import Path

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


@pytest.fixture(scope='module')
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([""])
    yield app


def _create_fake_fif(filepath, n_channels=4, n_samples=1000, sfreq=256.0):
    import mne
    info = mne.create_info(
        ch_names=[f'EEG {i:03d}' for i in range(n_channels)],
        sfreq=sfreq, ch_types=['eeg'] * n_channels,
    )
    data = np.random.randn(n_channels, n_samples) * 50e-6
    raw = mne.io.RawArray(data, info)
    raw.save(filepath, overwrite=True)


class TestSessionDialog:
    """SessionDialog confirmation dialog"""

    def test_all_checked_by_default(self, qapp):
        with tempfile.TemporaryDirectory() as tmp:
            for run in [4, 6, 8, 10]:
                _create_fake_fif(os.path.join(tmp, f'S001R{run:02d}.fif'))
            from bci.gui.session_loader import SessionDialog
            runs = [Path(tmp) / f'S001R{run:02d}.fif' for run in [4, 6, 8, 10]]
            dialog = SessionDialog(runs)
            assert dialog.checked_count() == 4

    def test_selected_runs_returns_all_by_default(self, qapp):
        with tempfile.TemporaryDirectory() as tmp:
            for run in [4, 6, 8, 10]:
                _create_fake_fif(os.path.join(tmp, f'S001R{run:02d}.fif'))
            runs = [Path(tmp) / f'S001R{run:02d}.fif' for run in [4, 6, 8, 10]]
            from bci.gui.session_loader import SessionDialog
            dialog = SessionDialog(runs)
            selected = dialog.selected_runs()
            assert len(selected) == 4

    def test_deselect_all_disables_confirm(self, qapp):
        with tempfile.TemporaryDirectory() as tmp:
            for run in [4, 6, 8, 10]:
                _create_fake_fif(os.path.join(tmp, f'S001R{run:02d}.fif'))
            runs = [Path(tmp) / f'S001R{run:02d}.fif' for run in [4, 6, 8, 10]]
            from bci.gui.session_loader import SessionDialog
            dialog = SessionDialog(runs)
            dialog._deselect_all()
            assert dialog.checked_count() == 0
            assert not dialog._confirm_btn.isEnabled()

    def test_deselect_all_then_select_all(self, qapp):
        with tempfile.TemporaryDirectory() as tmp:
            for run in [4, 6, 8, 10]:
                _create_fake_fif(os.path.join(tmp, f'S001R{run:02d}.fif'))
            runs = [Path(tmp) / f'S001R{run:02d}.fif' for run in [4, 6, 8, 10]]
            from bci.gui.session_loader import SessionDialog
            dialog = SessionDialog(runs)
            dialog._deselect_all()
            dialog._select_all()
            assert dialog.checked_count() == 4
            assert dialog._confirm_btn.isEnabled()

    def test_info_label_contains_metadata(self, qapp):
        with tempfile.TemporaryDirectory() as tmp:
            for run in [4, 6, 8, 10]:
                _create_fake_fif(os.path.join(tmp, f'S001R{run:02d}.fif'), n_channels=4, sfreq=256.0)
            runs = [Path(tmp) / f'S001R{run:02d}.fif' for run in [4, 6, 8, 10]]
            from bci.gui.session_loader import SessionDialog
            dialog = SessionDialog(runs)
            text = dialog._info_label.text()
            assert "4 runs" in text
            assert "4 channels" in text
            assert "256 Hz" in text

    def test_double_digit_run_numbers(self, qapp):
        with tempfile.TemporaryDirectory() as tmp:
            for run in range(1, 13):
                _create_fake_fif(os.path.join(tmp, f'S001R{run:02d}.fif'))
            runs = [Path(tmp) / f'S001R{run:02d}.fif' for run in range(1, 13)]
            from bci.gui.session_loader import SessionDialog
            dialog = SessionDialog(runs)
            assert dialog.checked_count() == 12
            assert "12 runs" in dialog._info_label.text()
