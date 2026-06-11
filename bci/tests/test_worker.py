"""
Tests for bci.gui.worker module
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
        filepath = os.path.join(tmp, 'test.fif')
        info = mne.create_info(
            ch_names=['EEG 001', 'EEG 002'], sfreq=256.0,
            ch_types=['eeg'] * 2,
        )
        data = np.random.randn(2, 5000) * 50e-6
        raw = mne.io.RawArray(data, info)
        raw.save(filepath, overwrite=True)
        yield filepath


@pytest.fixture
def default_config():
    from bci.config import create_default_config
    return create_default_config()


class TestBatchWorker:
    """BatchWorker: run BCIPipeline in background"""

    def test_construction(self, qapp, fake_edf, default_config):
        from bci.gui.worker import BatchWorker
        worker = BatchWorker([fake_edf], default_config)
        assert worker is not None
        assert worker.filepaths == [str(fake_edf)]

    def test_signals_exist(self, qapp, fake_edf, default_config):
        from bci.gui.worker import BatchWorker
        worker = BatchWorker([fake_edf], default_config)
        assert hasattr(worker, 'progress')
        assert hasattr(worker, 'log')
        assert hasattr(worker, 'finished')
        assert hasattr(worker, 'error')

    def test_run_emits_signals(self, qapp, fake_edf, default_config):
        from bci.gui.worker import BatchWorker

        worker = BatchWorker([fake_edf], default_config)
        logs = []
        progresses = []
        errors = []
        results = []

        worker.log.connect(logs.append)
        worker.progress.connect(progresses.append)
        worker.error.connect(errors.append)
        worker.finished.connect(results.append)

        worker.run()

        assert len(logs) > 0, "Should emit log messages"
        assert len(progresses) > 0, "Should emit progress"
        assert progresses[0] == 0  # first progress emission
        # Fake data has no events, so pipeline may fail at epoch step
        # Both finished and error signals should not both be emitted
        assert len(results) + len(errors) > 0, "Should emit finished or error"

    def test_run_with_invalid_file_emits_error(self, qapp, default_config):
        from bci.gui.worker import BatchWorker
        worker = BatchWorker(["/nonexistent/file.edf"], default_config)
        errors = []
        worker.error.connect(errors.append)
        worker.run()
        assert len(errors) > 0


class TestLoadWorker:
    """LoadWorker: background SessionSource loading with progress"""

    def test_construction(self, qapp, fake_edf):
        from bci.gui.worker import LoadWorker
        worker = LoadWorker([fake_edf])
        assert worker is not None
        assert worker.filepaths == [fake_edf]

    def test_signals_exist(self, qapp, fake_edf):
        from bci.gui.worker import LoadWorker
        worker = LoadWorker([fake_edf])
        assert hasattr(worker, 'load_progress')
        assert hasattr(worker, 'finished')
        assert hasattr(worker, 'error')

    def test_run_emits_finished_with_source(self, qapp, fake_edf):
        from bci.gui.worker import LoadWorker
        import mne

        worker = LoadWorker([fake_edf])
        results = []
        progresses = []
        errors = []

        worker.finished.connect(results.append)
        worker.load_progress.connect(lambda c, t: progresses.append((c, t)))
        worker.error.connect(errors.append)

        worker.run()

        assert len(errors) == 0, f"Load error: {errors}"
        assert len(results) == 1, "Should emit finished with source"
        assert isinstance(results[0], mne.io.BaseRaw)
        assert len(progresses) > 0, "Should emit load progress"
        source = results[0]
        assert source.info['nchan'] > 0
        assert source.n_times > 0

    def test_run_with_invalid_file_emits_error(self, qapp):
        from bci.gui.worker import LoadWorker
        worker = LoadWorker(["/nonexistent/file.edf"])
        errors = []
        worker.error.connect(errors.append)
        worker.run()
        assert len(errors) > 0


class TestStreamWorker:
    """StreamWorker: StreamSource → OnlineProcessor → Qt signals"""

    def test_construction(self, qapp, fake_edf):
        from bci.gui.worker import StreamWorker
        from bci.source import FileSource
        eeg = FileSource.load(fake_edf)
        worker = StreamWorker(eeg)
        assert worker is not None

    def test_signals_exist(self, qapp, fake_edf):
        from bci.gui.worker import StreamWorker
        from bci.source import FileSource
        eeg = FileSource.load(fake_edf)
        worker = StreamWorker(eeg)
        assert hasattr(worker, 'chunk_processed')
        assert hasattr(worker, 'spectrum_updated')
        assert hasattr(worker, 'error')
        assert hasattr(worker, 'finished')

    def test_start_emits_chunk_in_realtime(self, qapp, fake_edf):
        from bci.gui.worker import StreamWorker
        from bci.source import FileSource
        from PyQt6.QtCore import QEventLoop, QTimer

        eeg = FileSource.load(fake_edf)
        worker = StreamWorker(eeg, chunk_duration=0.05)
        worker.set_speed(100.0)
        chunks = []
        worker.chunk_processed.connect(chunks.append)

        loop = QEventLoop()
        def check_done():
            if len(chunks) >= 3:
                loop.quit()

        QTimer.singleShot(3000, loop.quit)  # timeout
        worker.chunk_processed.connect(lambda _: check_done())
        worker.run()
        loop.exec()

        assert len(chunks) >= 3, f"Expected >=3 chunks, got {len(chunks)}"
        for chunk in chunks:
            assert chunk.shape[0] == 2  # 2 channels

    def test_stop_stops_streaming(self, qapp, fake_edf):
        from bci.gui.worker import StreamWorker
        from bci.source import FileSource
        from PyQt6.QtCore import QEventLoop, QTimer

        eeg = FileSource.load(fake_edf)
        worker = StreamWorker(eeg, chunk_duration=0.05)
        worker.set_speed(100.0)
        chunks = []
        worker.chunk_processed.connect(chunks.append)

        loop = QEventLoop()
        QTimer.singleShot(500, lambda: (worker.stop(), loop.quit()))
        QTimer.singleShot(2000, loop.quit)  # safety timeout
        worker.run()
        loop.exec()

        count_after_stop = len(chunks)
        import time
        time.sleep(0.3)
        assert len(chunks) == count_after_stop

    def test_set_speed(self, qapp, fake_edf):
        from bci.gui.worker import StreamWorker
        from bci.source import FileSource
        eeg = FileSource.load(fake_edf)
        worker = StreamWorker(eeg)
        worker.set_speed(5.0)
        assert worker.speed == 5.0


class TestStreamWorkerSource:
    """Tests that StreamWorker properly wraps StreamSource"""

    def test_source_is_accessible(self, qapp, fake_edf):
        from bci.gui.worker import StreamWorker
        from bci.source import FileSource
        eeg = FileSource.load(fake_edf)
        worker = StreamWorker(eeg)
        assert worker.source is not None
        assert worker.source.is_stream is True

    def test_seek_updates_source(self, qapp, fake_edf):
        from bci.gui.worker import StreamWorker
        from bci.source import FileSource
        eeg = FileSource.load(fake_edf)
        worker = StreamWorker(eeg)
        worker.seek(100)
        assert worker.source.position == 100

    def test_reset(self, qapp, fake_edf):
        from bci.gui.worker import StreamWorker
        from bci.source import FileSource
        eeg = FileSource.load(fake_edf)
        worker = StreamWorker(eeg)
        worker.source.read_chunk(500)
        worker.reset()
        assert worker.source.position == 0
