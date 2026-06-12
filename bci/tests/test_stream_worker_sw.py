"""StreamWorker + SlidingWindow integration tests."""
import numpy as np
import pytest


def _mock_eeg(n_channels=4, sfreq=160.0, n_samples=1600):
    """Create a minimal MNE Raw for StreamWorker construction."""
    import mne
    data = np.zeros((n_channels, n_samples), dtype=np.float32)
    info = mne.create_info(
        [f'Ch{i}' for i in range(n_channels)],
        sfreq, ch_types='eeg',
    )
    return mne.io.RawArray(data, info, verbose=False)


class TestStreamWorkerSlidingWindow:
    def test_worker_has_sliding_window_attribute(self):
        from bci.gui.worker import StreamWorker
        from bci.source import StreamSource
        sw = StreamWorker(StreamSource(_mock_eeg()), chunk_duration=0.1)
        assert hasattr(sw, "sliding_window")
        assert sw.sliding_window is None  # default: not configured

    def test_set_sliding_window_stores_config(self):
        from bci.gui.worker import StreamWorker
        from bci.source import StreamSource
        from bci.streaming import SlidingWindow
        sw = StreamWorker(StreamSource(_mock_eeg()), chunk_duration=0.1)
        swin = SlidingWindow(n_channels=64, window_size=1000, decision_interval=25)
        sw.set_sliding_window(swin)
        assert sw.sliding_window is swin


class _FakeSource:
    """Minimal source stub: deterministic read_chunk, fixed attrs."""

    def __init__(self, n_channels=4, sfreq=160.0, chunk_duration=0.1,
                 chunk=None):
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.chunk_duration = chunk_duration
        self._chunk = chunk if chunk is not None else np.ones(
            (n_channels, int(sfreq * chunk_duration)), dtype=np.float32
        )
        self.progress = 0

    def read_chunk(self, n):
        return self._chunk


class TestStreamWorkerEmitChunkWithSW:
    """Test that _emit_chunk uses SlidingWindow when configured."""

    def test_emit_chunk_uses_sliding_window_when_ready(self, monkeypatch):
        from bci.gui.worker import StreamWorker
        from bci.source import StreamSource
        from bci.streaming import SlidingWindow

        sw = StreamWorker(StreamSource(_mock_eeg()), chunk_duration=0.1)
        sw._chunk_samples = 16
        chunk = np.ones((4, 16), dtype=np.float32)  # 4ch × 16 samples
        monkeypatch.setattr(sw, "source",
                            _FakeSource(n_channels=4, sfreq=160.0,
                                        chunk_duration=0.1, chunk=chunk))

        # Configure SW that becomes ready after one push
        swin = SlidingWindow(n_channels=4, window_size=16, decision_interval=16)
        sw.set_sliding_window(swin)

        # Mock model
        class FakeModel:
            classes_ = np.array([0, 1])
            def predict_proba(self, X):
                # Record the shape that was passed
                self.last_X_shape = X.shape
                return np.array([[0.3, 0.7]])
        fake_model = FakeModel()
        sw._model = fake_model  # type: ignore[assignment]  # test mock
        sw._label_names = ["L", "R"]

        # Capture prediction emission
        predictions = []
        sw.prediction.connect(lambda lbl, conf: predictions.append((lbl, conf)))

        # Emit one chunk → SW becomes ready → predict_proba called
        sw._emit_chunk()
        assert fake_model.last_X_shape == (1, 4, 16)
        assert len(predictions) == 1
        assert predictions[0] == ("R", 0.7)
        # SW should have been pushed to (proves SW path was taken)
        assert swin._n_filled == 16
        # SW should be consumed (since_last reset)
        assert swin._since_last == 0

    def test_emit_chunk_skips_prediction_when_not_ready(self, monkeypatch):
        from bci.gui.worker import StreamWorker
        from bci.source import StreamSource
        from bci.streaming import SlidingWindow

        sw = StreamWorker(StreamSource(_mock_eeg()), chunk_duration=0.1)
        sw._chunk_samples = 16
        chunk = np.ones((4, 16), dtype=np.float32)
        monkeypatch.setattr(sw, "source",
                            _FakeSource(n_channels=4, sfreq=160.0,
                                        chunk_duration=0.1, chunk=chunk))

        # SW requires window_size=100 → not ready after one 16-sample chunk
        swin = SlidingWindow(n_channels=4, window_size=100, decision_interval=25)
        sw.set_sliding_window(swin)

        class FakeModel:
            predict_called = False
            def predict_proba(self, X):
                self.predict_called = True
                return np.array([[0.5, 0.5]])
            classes_ = np.array([0, 1])
        fake = FakeModel()
        sw._model = fake  # type: ignore[assignment]  # test mock

        predictions = []
        sw.prediction.connect(lambda lbl, conf: predictions.append((lbl, conf)))
        sw._emit_chunk()
        assert not fake.predict_called
        assert len(predictions) == 0

    def test_emit_chunk_falls_back_to_per_chunk_without_sw(self, monkeypatch):
        """Backward compat: no SW → predict on chunk directly (existing behavior)."""
        from bci.gui.worker import StreamWorker
        from bci.source import StreamSource

        sw = StreamWorker(StreamSource(_mock_eeg()), chunk_duration=0.1)
        sw._chunk_samples = 16
        chunk = np.ones((4, 16), dtype=np.float32)
        monkeypatch.setattr(sw, "source",
                            _FakeSource(n_channels=4, sfreq=160.0,
                                        chunk_duration=0.1, chunk=chunk))

        class FakeModel:
            last_X_shape = None
            def predict_proba(self, X):
                self.last_X_shape = X.shape
                return np.array([[0.5, 0.5]])
            classes_ = np.array([0, 1])
        fake = FakeModel()
        sw._model = fake  # type: ignore[assignment]  # test mock
        sw._label_names = ["L", "R"]

        predictions = []
        sw.prediction.connect(lambda lbl, conf: predictions.append((lbl, conf)))
        sw._emit_chunk()
        assert fake.last_X_shape == (1, 4, 16)  # direct chunk prediction
        assert len(predictions) == 1
