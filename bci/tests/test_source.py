"""
Tests for bci.source module
============================
"""
from __future__ import annotations
import pytest
import numpy as np
from pathlib import Path
import tempfile
import os


def _create_fake_edf(filepath: str, n_channels: int = 8,
                     n_samples: int = 5000, sfreq: float = 256.0):
    """Create a fake EDF file using MNE for testing"""
    import mne
    data = np.random.randn(n_channels, n_samples) * 50e-6
    info = mne.create_info(
        ch_names=[f'EEG {i:03d}' for i in range(n_channels)],
        sfreq=sfreq,
        ch_types=['eeg'] * n_channels,
    )
    raw = mne.io.RawArray(data, info)
    raw.save(filepath, overwrite=True)


class TestDataSourceABC:
    """DataSource abstract base class tests"""

    def test_cannot_instantiate_abstract(self):
        from bci.source.base import DataSource
        with pytest.raises(TypeError):
            DataSource()  # type: ignore[abstract]

    def test_concrete_must_implement_all_abstract(self):
        from bci.source.base import DataSource

        class PartialSource(DataSource):
            def open(self): pass
            # missing read_chunk, seek, close, sfreq, n_channels

        with pytest.raises(TypeError):
            PartialSource()  # type: ignore[abstract]


class TestFileSource:
    """FileSource tests using fake EDF"""

    @pytest.fixture
    def fake_edf(self):
        with tempfile.TemporaryDirectory() as tmp:
            filepath = os.path.join(tmp, 'test.fif')
            info = __import__('mne').create_info(
                ch_names=['EEG 001', 'EEG 002', 'EEG 003', 'EEG 004'],
                sfreq=256.0,
                ch_types=['eeg'] * 4,
            )
            data = np.random.randn(4, 5000) * 50e-6
            raw = __import__('mne').io.RawArray(data, info)
            raw.save(filepath, overwrite=True)
            yield filepath

    def test_open_loads_file(self, fake_edf):
        from bci.source.file_source import FileSource
        source = FileSource(fake_edf)
        source.open()
        assert source.sfreq == 256.0
        assert source.n_channels == 4

    def test_total_samples_known_after_open(self, fake_edf):
        from bci.source.file_source import FileSource
        source = FileSource(fake_edf)
        source.open()
        assert source.total_samples == 5000

    def test_get_data_returns_correct_shape(self, fake_edf):
        from bci.source.file_source import FileSource
        source = FileSource(fake_edf)
        source.open()
        data, times = source.get_data()
        assert data.shape == (4, 5000)
        assert len(times) == 5000

    def test_is_stream_returns_false(self, fake_edf):
        from bci.source.file_source import FileSource
        source = FileSource(fake_edf)
        source.open()
        assert source.is_stream is False

    def test_get_data_with_range(self, fake_edf):
        from bci.source.file_source import FileSource
        source = FileSource(fake_edf)
        source.open()
        data, times = source.get_data(start=100, stop=200)
        assert data.shape == (4, 100)
        assert len(times) == 100

    def test_close_cleans_up(self, fake_edf):
        from bci.source.file_source import FileSource
        source = FileSource(fake_edf)
        source.open()
        source.close()
        data, _ = source.get_data()
        assert data.size == 0


class TestStreamSource:
    """StreamSource tests"""

    @pytest.fixture
    def fake_edf(self):
        with tempfile.TemporaryDirectory() as tmp:
            filepath = os.path.join(tmp, 'test.fif')
            info = __import__('mne').create_info(
                ch_names=['EEG 001', 'EEG 002'],
                sfreq=256.0,
                ch_types=['eeg'] * 2,
            )
            data = np.random.randn(2, 2560) * 50e-6
            raw = __import__('mne').io.RawArray(data, info)
            raw.save(filepath, overwrite=True)
            yield filepath

    def test_open_loads_file(self, fake_edf):
        from bci.source.stream_source import StreamSource
        source = StreamSource(fake_edf)
        source.open()
        assert source.sfreq == 256.0
        assert source.n_channels == 2

    def test_read_chunk_returns_correct_shape(self, fake_edf):
        from bci.source.stream_source import StreamSource
        source = StreamSource(fake_edf)
        source.open()
        chunk = source.read_chunk(128)
        assert chunk.shape == (2, 128)

    def test_read_chunk_advances_position(self, fake_edf):
        from bci.source.stream_source import StreamSource
        source = StreamSource(fake_edf)
        source.open()
        chunk1 = source.read_chunk(100)
        chunk2 = source.read_chunk(100)
        assert not np.array_equal(chunk1, chunk2)

    def test_read_chunk_returns_none_at_eof(self, fake_edf):
        from bci.source.stream_source import StreamSource
        source = StreamSource(fake_edf)
        source.open()
        _ = source.read_chunk(2560)
        result = source.read_chunk(100)
        assert result is None

    def test_read_chunk_returns_partial_at_boundary(self, fake_edf):
        from bci.source.stream_source import StreamSource
        source = StreamSource(fake_edf)
        source.open()
        source.read_chunk(2500)  # advance near end
        chunk = source.read_chunk(200)  # ask more than remaining
        assert chunk is not None
        assert chunk.shape[1] == 60  # 2560 - 2500 = 60

    def test_seek_moves_to_position(self, fake_edf):
        from bci.source.stream_source import StreamSource
        source = StreamSource(fake_edf)
        source.open()
        chunk_before = source.read_chunk(100)
        source.seek(0)
        chunk_after = source.read_chunk(100)
        assert np.array_equal(chunk_before, chunk_after)

    def test_seek_middle_position(self, fake_edf):
        from bci.source.stream_source import StreamSource
        source = StreamSource(fake_edf)
        source.open()
        source.read_chunk(500)
        chunk_at_500 = source.read_chunk(100)
        source.seek(500)
        chunk_after_seek = source.read_chunk(100)
        assert np.array_equal(chunk_at_500, chunk_after_seek)

    def test_is_stream_returns_true(self, fake_edf):
        from bci.source.stream_source import StreamSource
        source = StreamSource(fake_edf)
        source.open()
        assert source.is_stream is True

    def test_total_samples_known(self, fake_edf):
        from bci.source.stream_source import StreamSource
        source = StreamSource(fake_edf)
        source.open()
        assert source.total_samples == 2560

    def test_progress_reports_correct_value(self, fake_edf):
        from bci.source.stream_source import StreamSource
        source = StreamSource(fake_edf)
        source.open()
        assert source.progress == 0
        source.read_chunk(1280)
        assert source.progress == 50

    def test_position_reports_correct_value(self, fake_edf):
        from bci.source.stream_source import StreamSource
        source = StreamSource(fake_edf)
        source.open()
        assert source.position == 0
        source.read_chunk(512)
        assert source.position == 512

    def test_reset_returns_to_start(self, fake_edf):
        from bci.source.stream_source import StreamSource
        source = StreamSource(fake_edf)
        source.open()
        chunk_before = source.read_chunk(100)
        source.read_chunk(500)
        source.reset()
        chunk_after = source.read_chunk(100)
        assert np.array_equal(chunk_before, chunk_after)

    def test_loop_mode_wraps_at_eof(self, fake_edf):
        from bci.source.stream_source import StreamSource
        source = StreamSource(fake_edf)
        source.open()
        source.set_loop(True)
        source.read_chunk(2560)  # exhaust
        chunk = source.read_chunk(100)
        assert chunk is not None
        assert chunk.shape == (2, 100)

    def test_non_loop_mode_stops_at_eof(self, fake_edf):
        from bci.source.stream_source import StreamSource
        source = StreamSource(fake_edf)
        source.open()
        source.set_loop(False)
        source.read_chunk(2560)
        chunk = source.read_chunk(100)
        assert chunk is None

    def test_close_cleans_up(self, fake_edf):
        from bci.source.stream_source import StreamSource
        source = StreamSource(fake_edf)
        source.open()
        source.close()
        result = source.read_chunk(100)
        assert result is None
