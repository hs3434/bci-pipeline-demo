"""
Tests for bci.gui.widgets module
"""
from __future__ import annotations
import pytest
import numpy as np
import os

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


@pytest.fixture(scope='session')
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([""])
    yield app


def _gen_eeg(n_channels=8, n_samples=1000, sfreq=256.0):
    t = np.arange(n_samples) / sfreq
    data = np.zeros((n_channels, n_samples))
    for i in range(n_channels):
        data[i] = np.sin(2 * np.pi * 10 * t + i * 0.5) * 50
    ch_names = [f'EEG {i:03d}' for i in range(n_channels)]
    return data, sfreq, ch_names


class TestEEGWaveformWidget:
    """EEG waveform display — batch and stream modes"""

    def test_construction(self, qapp):
        from bci.gui.widgets.waveform import EEGWaveformWidget
        widget = EEGWaveformWidget()
        assert widget is not None

    def test_plot_batch_updates_without_error(self, qapp):
        from bci.gui.widgets.waveform import EEGWaveformWidget
        data, sfreq, ch_names = _gen_eeg()
        widget = EEGWaveformWidget()
        widget.plot_batch(data, sfreq, ch_names)

    def test_update_stream_appends_to_buffer(self, qapp):
        from bci.gui.widgets.waveform import EEGWaveformWidget
        _, sfreq, ch_names = _gen_eeg(n_channels=4, n_samples=500)
        widget = EEGWaveformWidget()
        widget._init_buffer(4, sfreq, ch_names, window_sec=2.0)
        chunk = np.random.randn(4, 64)
        widget.update_stream(chunk)

    def test_update_stream_with_different_channels_raises(self, qapp):
        from bci.gui.widgets.waveform import EEGWaveformWidget
        _, sfreq, ch_names = _gen_eeg(n_channels=4, n_samples=500)
        widget = EEGWaveformWidget()
        widget._init_buffer(4, sfreq, ch_names, window_sec=2.0)
        chunk = np.random.randn(8, 64)
        with pytest.raises(ValueError):
            widget.update_stream(chunk)

    def test_clear_resets_plot(self, qapp):
        from bci.gui.widgets.waveform import EEGWaveformWidget
        data, sfreq, ch_names = _gen_eeg(n_channels=4, n_samples=200)
        widget = EEGWaveformWidget()
        widget.plot_batch(data, sfreq, ch_names)
        widget.clear()


class TestSpectrumWidget:
    """PSD spectrum display"""

    def test_construction(self, qapp):
        from bci.gui.widgets.spectrum import SpectrumWidget
        widget = SpectrumWidget()
        assert widget is not None

    def test_update_with_chunk(self, qapp):
        from bci.gui.widgets.spectrum import SpectrumWidget
        data, sfreq, _ = _gen_eeg(n_channels=4, n_samples=1024)
        widget = SpectrumWidget()
        widget.update_psd(data, sfreq)

    def test_update_with_multiple_calls(self, qapp):
        from bci.gui.widgets.spectrum import SpectrumWidget
        widget = SpectrumWidget()
        for _ in range(5):
            data = np.random.randn(4, 256)
            widget.update_psd(data, sfreq=256.0)


class TestTopomapWidget:
    """Scalp topography display"""

    def test_construction(self, qapp):
        from bci.gui.widgets.topomap import TopomapWidget
        widget = TopomapWidget()
        assert widget is not None

    def test_update_with_data_and_positions(self, qapp):
        from bci.gui.widgets.topomap import TopomapWidget
        widget = TopomapWidget()
        data = np.random.randn(8)
        ch_names = [f'EEG {i:03d}' for i in range(8)]
        widget.update_topo(data, ch_names)

    def test_update_without_positions_graceful(self, qapp):
        from bci.gui.widgets.topomap import TopomapWidget
        widget = TopomapWidget()
        widget.show_fallback("No position data available")


class TestResultPanel:
    """Decoding result display"""

    def test_construction(self, qapp):
        from bci.gui.widgets.result_panel import ResultPanel
        widget = ResultPanel()
        assert widget is not None

    def test_update_batch_result(self, qapp):
        from bci.gui.widgets.result_panel import ResultPanel
        widget = ResultPanel()
        widget.update_batch(accuracy=0.85, std=0.05,
                           cv_scores=[0.82, 0.88, 0.85, 0.84, 0.86],
                           method='LDA')

    def test_update_stream_prediction(self, qapp):
        from bci.gui.widgets.result_panel import ResultPanel
        widget = ResultPanel()
        widget.update_stream("LEFT")

    def test_clear(self, qapp):
        from bci.gui.widgets.result_panel import ResultPanel
        widget = ResultPanel()
        widget.update_batch(accuracy=0.75, std=0.03,
                           cv_scores=[0.70, 0.80], method='LDA')
        widget.clear()
