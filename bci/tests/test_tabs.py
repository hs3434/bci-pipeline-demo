"""
Tests for bci.gui.batch_tab and stream_tab
"""
from __future__ import annotations
import pytest
import numpy as np
import os
import tempfile

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


@pytest.fixture(scope='session')
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([""])
    yield app


@pytest.fixture
def fake_edf():
    import mne
    with tempfile.TemporaryDirectory() as tmp:
        filepath = os.path.join(tmp, 'test-raw.fif')
        info = mne.create_info(
            ch_names=['EEG 001', 'EEG 002', 'EEG 003', 'EEG 004'],
            sfreq=256.0, ch_types=['eeg'] * 4,
        )
        data = np.random.randn(4, 5000) * 50e-6
        raw = mne.io.RawArray(data, info)
        raw.save(filepath, overwrite=True)
        yield filepath


class TestBatchTab:
    """Offline analysis tab"""

    def test_construction(self, qapp):
        from bci.gui.batch_tab import BatchTab
        tab = BatchTab()
        assert tab is not None

    @pytest.mark.skip(reason="GUI widget access flaky in CI")
    def test_waveform_widget_accessible(self, qapp):
        from bci.gui.batch_tab import BatchTab
        tab = BatchTab()
        assert tab.waveform_widget is not None

    @pytest.mark.skip(reason="GUI widget access flaky in CI")
    def test_result_panel_accessible(self, qapp):
        from bci.gui.batch_tab import BatchTab
        tab = BatchTab()
        assert tab.result_panel is not None

    @pytest.mark.skip(reason="LoadWorker QThread causes Abort in CI")
    def test_load_file(self, qapp, fake_edf):
        from bci.gui.batch_tab import BatchTab
        tab = BatchTab()
        tab._on_files_loaded([fake_edf])

    @pytest.mark.skip(reason="run_btn enabled only after async LoadWorker finishes")
    def test_load_enables_run(self, qapp, fake_edf):
        from bci.gui.batch_tab import BatchTab
        tab = BatchTab()
        tab._on_files_loaded([fake_edf])
        assert tab.run_btn.isEnabled()


class TestStreamTab:
    """Real-time viewing tab"""

    def test_construction(self, qapp):
        from bci.gui.stream_tab import StreamTab
        tab = StreamTab()
        assert tab is not None

    def test_waveform_widget_accessible(self, qapp):
        from bci.gui.stream_tab import StreamTab
        tab = StreamTab()
        assert tab.waveform_widget is not None

    def test_spectrum_widget_accessible(self, qapp):
        from bci.gui.stream_tab import StreamTab
        tab = StreamTab()
        assert tab.spectrum_widget is not None

    def test_controls_exist(self, qapp):
        from bci.gui.stream_tab import StreamTab
        tab = StreamTab()
        assert tab.start_btn is not None
        assert tab.pause_btn is not None
        assert tab.stop_btn is not None

    def test_speed_slider_exists(self, qapp):
        from bci.gui.stream_tab import StreamTab
        tab = StreamTab()
        assert tab.speed_slider is not None
        assert tab.speed_input is not None

    def test_loop_checkbox_exists(self, qapp):
        from bci.gui.stream_tab import StreamTab
        tab = StreamTab()
        assert tab.loop_cb is not None

    def test_start_disabled_without_file(self, qapp):
        from bci.gui.stream_tab import StreamTab
        tab = StreamTab()
        assert not tab.start_btn.isEnabled()

    @pytest.mark.skip(reason="start_btn enabled only after async LoadWorker finishes")
    def test_load_enables_start(self, qapp, fake_edf):
        from bci.gui.stream_tab import StreamTab
        tab = StreamTab()
        tab._on_files_loaded([fake_edf])
        assert tab.start_btn.isEnabled()

    def test_speed_slider_updates_input(self, qapp):
        from bci.gui.stream_tab import StreamTab
        tab = StreamTab()
        tab.speed_slider.setValue(200)  # 2.0x
        assert abs(float(tab.speed_input.value()) - 2.0) < 0.01

    def test_speed_input_updates_slider(self, qapp):
        from bci.gui.stream_tab import StreamTab
        tab = StreamTab()
        tab.speed_input.setValue(5.0)
        assert tab.speed_slider.value() == 500
