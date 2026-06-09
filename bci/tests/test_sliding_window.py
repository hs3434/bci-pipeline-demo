import numpy as np
import pytest
from bci.streaming.sliding_window import SlidingWindow


class TestSlidingWindowConstructor:
    def test_valid_construction(self):
        sw = SlidingWindow(n_channels=4, window_size=1000, decision_interval=25)
        assert sw.n_channels == 4
        assert sw.window_size == 1000
        assert sw.decision_interval == 25

    def test_rejects_decision_interval_zero(self):
        with pytest.raises(ValueError, match="decision_interval"):
            SlidingWindow(n_channels=4, window_size=1000, decision_interval=0)

    def test_rejects_decision_interval_larger_than_window(self):
        with pytest.raises(ValueError, match="decision_interval"):
            SlidingWindow(n_channels=4, window_size=100, decision_interval=200)

    def test_rejects_zero_window_size(self):
        with pytest.raises(ValueError, match="window_size"):
            SlidingWindow(n_channels=4, window_size=0, decision_interval=10)


class TestSlidingWindowBehavior:
    def test_not_ready_until_buffer_full(self):
        sw = SlidingWindow(n_channels=2, window_size=100, decision_interval=25)
        sw.push(np.zeros((2, 50), dtype=np.float32))
        assert not sw.ready()
        sw.push(np.zeros((2, 50), dtype=np.float32))
        assert sw.ready()

    def test_ready_requires_decision_interval(self):
        sw = SlidingWindow(n_channels=2, window_size=100, decision_interval=50)
        sw.push(np.zeros((2, 100), dtype=np.float32))
        assert sw.ready()
        sw.consume()
        assert not sw.ready()

    def test_get_window_returns_chronological_order(self):
        sw = SlidingWindow(n_channels=1, window_size=5, decision_interval=1)
        sw.push(np.array([[10, 20, 30, 40, 50]], dtype=np.float32))
        window = sw.get_window()
        np.testing.assert_array_equal(window, [[10, 20, 30, 40, 50]])

    def test_get_window_handles_wrap_around(self):
        sw = SlidingWindow(n_channels=1, window_size=5, decision_interval=1)
        sw.push(np.array([[1, 2, 3, 4, 5]], dtype=np.float32))
        sw.push(np.array([[6, 7, 8]], dtype=np.float32))
        window = sw.get_window()
        np.testing.assert_array_equal(window, [[4, 5, 6, 7, 8]])

    def test_get_window_before_full_returns_partial(self):
        sw = SlidingWindow(n_channels=1, window_size=10, decision_interval=5)
        sw.push(np.array([[1, 2, 3]], dtype=np.float32))
        window = sw.get_window()
        assert window.shape == (1, 3)
        np.testing.assert_array_equal(window, [[1, 2, 3]])

    def test_reset_clears_buffer(self):
        sw = SlidingWindow(n_channels=1, window_size=5, decision_interval=1)
        sw.push(np.array([[1, 2, 3, 4, 5]], dtype=np.float32))
        sw.reset()
        assert not sw.ready()
        assert sw.get_window().shape == (1, 0)

    def test_push_rejects_wrong_n_channels(self):
        sw = SlidingWindow(n_channels=4, window_size=10, decision_interval=1)
        with pytest.raises(ValueError, match="n_channels"):
            sw.push(np.zeros((2, 5), dtype=np.float32))

    def test_push_accepts_1d_chunk(self):
        sw = SlidingWindow(n_channels=3, window_size=10, decision_interval=1)
        sw.push(np.array([1, 2, 3], dtype=np.float32))
        assert sw._n_filled == 1
