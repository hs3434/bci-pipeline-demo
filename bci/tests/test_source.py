"""
Tests for bci.source module
============================
Updated for EEGData / EEGReader / FileSource / StreamSource design.
"""
from __future__ import annotations
import pytest
import numpy as np
from pathlib import Path
import tempfile
import os


def _create_fake_fif(filepath: str, n_channels: int = 4,
                     n_samples: int = 5000, sfreq: float = 256.0):
    """Create a fake FIF file using MNE for testing."""
    import mne
    data = np.random.randn(n_channels, n_samples) * 50e-6
    info = mne.create_info(
        ch_names=[f'EEG {i:03d}' for i in range(n_channels)],
        sfreq=sfreq,
        ch_types=['eeg'] * n_channels,
    )
    raw = mne.io.RawArray(data, info)
    raw.save(filepath, overwrite=True)


class TestEEGData:
    """EEGData container unit tests."""

    def test_basic_properties(self):
        from bci.source.base import EEGData
        data = np.zeros((4, 500), dtype=float)
        eeg = EEGData(data=data, sfreq=256.0,
                      ch_names=['Ch1', 'Ch2', 'Ch3', 'Ch4'])
        assert eeg.n_channels == 4
        assert eeg.n_samples == 500
        assert eeg.total_samples == 500
        assert eeg.duration == pytest.approx(500 / 256.0)

    def test_source_path_optional(self):
        from bci.source.base import EEGData
        eeg = EEGData(data=np.zeros((1, 10)), sfreq=100.0, ch_names=['A'])
        assert eeg.source_path is None
        eeg.source_path = '/tmp/test.edf'
        assert eeg.source_path == '/tmp/test.edf'


class TestEEGReaderABC:
    """EEGReader abstract base class tests."""

    def test_cannot_instantiate_abstract(self):
        from bci.source.base import EEGReader
        with pytest.raises(TypeError):
            EEGReader()  # type: ignore[abstract]

    def test_concrete_must_implement_read(self):
        from bci.source.base import EEGReader

        class PartialReader(EEGReader):
            @classmethod
            def suffixes(cls): return ('.x',)
            # missing read()

        with pytest.raises(TypeError):
            PartialReader()  # type: ignore[abstract]


class TestReaderRegistry:
    """@register_reader decorator tests."""

    def test_register_and_resolve(self):
        from bci.source.base import (
            EEGData, EEGReader, register_reader, get_reader, _reader_registry,
        )
        key = '.testreg'

        @register_reader(key)
        class DummyReader(EEGReader):
            @classmethod
            def suffixes(cls): return ('.testreg',)
            def read(self, filepath):
                return EEGData(
                    data=np.zeros((2, 10)), sfreq=100.0,
                    ch_names=['A', 'B'],
                )

        assert key in _reader_registry
        reader = get_reader(Path('data.testreg'))
        assert isinstance(reader, DummyReader)


class TestFileSource:
    """FileSource facade tests."""

    @pytest.fixture
    def fake_fif(self):
        with tempfile.TemporaryDirectory() as tmp:
            filepath = os.path.join(tmp, 'test.fif')
            _create_fake_fif(filepath, n_channels=4, n_samples=5000)
            yield filepath

    def test_load_returns_eegdata(self, fake_fif):
        from bci.source import FileSource
        eeg = FileSource.load(fake_fif)
        assert eeg.n_channels == 4
        assert eeg.n_samples == 5000
        assert eeg.sfreq == 256.0

    def test_load_sets_source_path(self, fake_fif):
        from bci.source import FileSource
        eeg = FileSource.load(fake_fif)
        assert eeg.source_path == fake_fif

    def test_load_raw_returns_mne_raw(self, fake_fif):
        from bci.source import FileSource
        raw = FileSource.load_raw(fake_fif)
        assert len(raw.ch_names) == 4
        assert raw.info['sfreq'] == 256.0

    def test_load_list_of_paths(self, fake_fif):
        from bci.source import FileSource
        import tempfile as tmp_mod
        with tempfile.TemporaryDirectory() as tmp2:
            fp2 = os.path.join(tmp2, 'test2.fif')
            _create_fake_fif(fp2, n_channels=4, n_samples=3000)
            eeg = FileSource.load([fake_fif, fp2])
            assert eeg.n_samples == 8000  # 5000 + 3000
            assert eeg.n_channels == 4


class TestStreamSource:
    """StreamSource tests using synthetic EEGData."""

    @pytest.fixture
    def eeg_data(self):
        from bci.source.base import EEGData
        return EEGData(
            data=np.arange(2 * 2560, dtype=float).reshape(2, 2560),
            sfreq=256.0,
            ch_names=['EEG 001', 'EEG 002'],
            source_path='/tmp/test.fif',
        )

    @pytest.fixture
    def stream(self, eeg_data):
        from bci.source import StreamSource
        return StreamSource(eeg_data, chunk_duration=0.1)

    def test_properties(self, stream):
        assert stream.sfreq == 256.0
        assert stream.n_channels == 2
        assert stream.total_samples == 2560
        assert stream.ch_names == ['EEG 001', 'EEG 002']
        assert stream.source_path == '/tmp/test.fif'

    def test_read_chunk_default(self, stream):
        chunk = stream.read_chunk()
        expected = max(1, int(256.0 * 0.1))
        assert chunk.shape == (2, expected)

    def test_read_chunk_explicit_samples(self, stream):
        chunk = stream.read_chunk(128)
        assert chunk.shape == (2, 128)

    def test_read_chunk_advances_position(self, stream):
        chunk1 = stream.read_chunk(100)
        chunk2 = stream.read_chunk(100)
        assert not np.array_equal(chunk1, chunk2)

    def test_read_chunk_none_at_eof(self, stream):
        _ = stream.read_chunk(2560)
        assert stream.read_chunk(100) is None

    def test_read_chunk_partial_at_boundary(self, stream):
        stream.read_chunk(2500)
        chunk = stream.read_chunk(200)
        assert chunk is not None
        assert chunk.shape[1] == 60

    def test_seek(self, stream):
        chunk_before = stream.read_chunk(100)
        stream.seek(0)
        chunk_after = stream.read_chunk(100)
        assert np.array_equal(chunk_before, chunk_after)

    def test_seek_middle(self, stream):
        stream.read_chunk(500)
        chunk_at_500 = stream.read_chunk(100)
        stream.seek(500)
        chunk_after_seek = stream.read_chunk(100)
        assert np.array_equal(chunk_at_500, chunk_after_seek)

    def test_is_stream(self, stream):
        assert stream.is_stream is True

    def test_progress(self, stream):
        assert stream.progress == 0
        stream.read_chunk(1280)
        assert stream.progress == 50

    def test_position(self, stream):
        assert stream.position == 0
        stream.read_chunk(512)
        assert stream.position == 512

    def test_reset(self, stream):
        chunk_before = stream.read_chunk(100)
        stream.read_chunk(500)
        stream.reset()
        chunk_after = stream.read_chunk(100)
        assert np.array_equal(chunk_before, chunk_after)

    def test_loop_mode(self, stream):
        stream.set_loop(True)
        stream.read_chunk(2560)
        chunk = stream.read_chunk(100)
        assert chunk is not None
        assert chunk.shape == (2, 100)

    def test_non_loop_stops(self, stream):
        stream.set_loop(False)
        stream.read_chunk(2560)
        assert stream.read_chunk(100) is None

    def test_close(self, stream):
        stream.close()
        assert stream.read_chunk(100) is None


class TestFindSessionRuns:
    """find_session_runs tests."""

    def test_single_file_no_pattern(self, tmp_path):
        from bci.source.file_source import find_session_runs
        p = tmp_path / 'random.edf'
        p.touch()
        runs = find_session_runs(p)
        assert runs == [p]

    def test_pattern_matches_runs(self, tmp_path):
        from bci.source.file_source import find_session_runs
        for name in ['S001R04.edf', 'S001R06.edf', 'S001R08.edf', 'S001R10.edf']:
            (tmp_path / name).touch()
        # also create an unrelated file
        (tmp_path / 'S002R01.edf').touch()
        runs = find_session_runs(tmp_path / 'S001R04.edf')
        assert len(runs) == 4
        assert [p.name for p in runs] == [
            'S001R04.edf', 'S001R06.edf', 'S001R08.edf', 'S001R10.edf',
        ]
