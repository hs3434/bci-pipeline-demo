"""
Tests for bci.source.file_source module
=========================================
Session concatenation via FileSource.load(list_of_paths) + StreamSource.
"""
from __future__ import annotations
import pytest
import numpy as np
import os
import re
import tempfile
from pathlib import Path

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


@pytest.fixture(scope='module')
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([""])
    yield app


def _create_fake_fif(filepath: str, n_channels: int = 4,
                     n_samples: int = 1000, sfreq: float = 256.0):
    """Create a fake FIF file (MNE's preferred format)."""
    import mne
    info = mne.create_info(
        ch_names=[f'EEG {i:03d}' for i in range(n_channels)],
        sfreq=sfreq, ch_types=['eeg'] * n_channels,
    )
    data = np.random.randn(n_channels, n_samples) * 50e-6
    raw = mne.io.RawArray(data, info)
    raw.save(filepath, overwrite=True)


class TestFindSessionRuns:
    """find_session_runs() utility."""

    def test_glob_finds_4_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            for run in [4, 6, 8, 10]:
                fif = os.path.join(tmp, f'S001R{run:02d}.fif')
                _create_fake_fif(fif)
            from bci.gui.session_loader import find_session_runs
            runs = find_session_runs(Path(tmp) / 'S001R04.fif')
            assert len(runs) == 4

    def test_glob_single_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            fif = os.path.join(tmp, 'solo.fif')
            _create_fake_fif(fif)
            from bci.gui.session_loader import find_session_runs
            runs = find_session_runs(Path(tmp) / 'solo.fif')
            assert len(runs) == 1

    def test_run_order_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            for run in [4, 6, 8, 10]:
                fif = os.path.join(tmp, f'S001R{run:02d}.fif')
                _create_fake_fif(fif)
            from bci.gui.session_loader import find_session_runs
            runs = find_session_runs(Path(tmp) / 'S001R04.fif')
            run_nums = [int(re.search(r'R(\d+)', str(r)).group(1)) for r in runs]
            assert run_nums == [4, 6, 8, 10]


class TestSessionConcatenation:
    """FileSource.load(list_of_paths) concatenates multiple runs."""

    def test_total_samples_4_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for run in [4, 6, 8, 10]:
                fp = os.path.join(tmp, f'S001R{run:02d}.fif')
                _create_fake_fif(fp, n_samples=1000)
                paths.append(fp)
            from bci.source import FileSource
            raw = FileSource.load(paths)
            assert raw.n_times == 4000

    def test_n_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for run in [4, 6, 8, 10]:
                fp = os.path.join(tmp, f'S001R{run:02d}.fif')
                _create_fake_fif(fp)
                paths.append(fp)
            from bci.source import FileSource
            raw = FileSource.load(paths)
            assert raw.info['nchan'] == 4

    def test_stream_after_concat(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for run in [4, 6, 8, 10]:
                fp = os.path.join(tmp, f'S001R{run:02d}.fif')
                _create_fake_fif(fp, n_samples=1000)
                paths.append(fp)
            from bci.source import FileSource, StreamSource
            raw = FileSource.load(paths)
            stream = StreamSource(raw)
            chunk = stream.read_chunk(500)
            assert chunk.shape == (4, 500)

    def test_read_all_chunks_exhausts(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for run in [4, 6, 8, 10]:
                fp = os.path.join(tmp, f'S001R{run:02d}.fif')
                _create_fake_fif(fp, n_samples=1000)
                paths.append(fp)
            from bci.source import FileSource, StreamSource
            raw = FileSource.load(paths)
            stream = StreamSource(raw)
            total = 0
            while True:
                chunk = stream.read_chunk(500)
                if chunk is None:
                    break
                total += chunk.shape[1]
            assert total == 4000

    def test_loop_wraps_at_eof(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for run in [4, 6, 8, 10]:
                fp = os.path.join(tmp, f'S001R{run:02d}.fif')
                _create_fake_fif(fp, n_samples=1000)
                paths.append(fp)
            from bci.source import FileSource, StreamSource
            raw = FileSource.load(paths)
            stream = StreamSource(raw)
            stream.set_loop(True)
            while stream.position < stream.total_samples:
                stream.read_chunk(500)
            chunk = stream.read_chunk(200)
            assert chunk is not None
            assert chunk.shape[1] == 200

    def test_progress_0_at_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for run in [4, 6, 8, 10]:
                fp = os.path.join(tmp, f'S001R{run:02d}.fif')
                _create_fake_fif(fp)
                paths.append(fp)
            from bci.source import FileSource, StreamSource
            raw = FileSource.load(paths)
            stream = StreamSource(raw)
            assert stream.progress == 0

    def test_progress_50_at_middle(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for run in [4, 6, 8, 10]:
                fp = os.path.join(tmp, f'S001R{run:02d}.fif')
                _create_fake_fif(fp, n_samples=1000)
                paths.append(fp)
            from bci.source import FileSource, StreamSource
            raw = FileSource.load(paths)
            stream = StreamSource(raw)
            stream.seek(2000)
            assert stream.progress == 50

    def test_reset_to_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for run in [4, 6, 8, 10]:
                fp = os.path.join(tmp, f'S001R{run:02d}.fif')
                _create_fake_fif(fp)
                paths.append(fp)
            from bci.source import FileSource, StreamSource
            raw = FileSource.load(paths)
            stream = StreamSource(raw)
            stream.read_chunk(500)
            stream.reset()
            assert stream.position == 0

    @pytest.mark.realdata
    def test_with_real_bci_data(self):
        """Integration test using real /data/bci files."""
        if not os.path.exists('/data/bci/S001R04.edf'):
            pytest.skip("Real BCI data not available")
        from bci.source import FileSource, StreamSource
        from bci.gui.session_loader import find_session_runs
        runs = find_session_runs(Path('/data/bci/S001R04.edf'))
        raw = FileSource.load([str(r) for r in runs])
        assert raw.info['nchan'] == 64
        assert raw.info['sfreq'] == 160.0
        assert raw.n_times == 20000 * len(runs)
        stream = StreamSource(raw)
        chunk = stream.read_chunk(1600)
        assert chunk.shape == (64, 1600)


class TestBatchTabSessionLoading:
    """BatchTab _on_files_loaded interface"""

    @pytest.mark.skip(reason="LoadWorker moveToThread causes Abort in CI")
    def test_multi_files_session_display(self, qapp):
        with tempfile.TemporaryDirectory() as tmp:
            for run in [4, 6, 8, 10]:
                _create_fake_fif(os.path.join(tmp, f'S001R{run:02d}.fif'))
            paths = [os.path.join(tmp, f'S001R{run:02d}.fif')
                     for run in [4, 6, 8, 10]]
            from bci.gui.batch_tab import BatchTab
            tab = BatchTab()
            tab._on_files_loaded(paths)
            assert len(tab._filepaths) == 4
            assert "4 runs" in tab.status_label.text()

    @pytest.mark.skip(reason="LoadWorker moveToThread causes Abort in CI")
    def test_load_single_file(self, qapp):
        with tempfile.TemporaryDirectory() as tmp:
            _create_fake_fif(os.path.join(tmp, 'solo.fif'))
            from bci.gui.batch_tab import BatchTab
            tab = BatchTab()
            tab._on_files_loaded([os.path.join(tmp, 'solo.fif')])
            assert len(tab._filepaths) == 1
            assert "Loaded" in tab.status_label.text()
