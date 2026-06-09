"""
Tests for bci.decoder.deep module (CNNDecoder)
"""
from __future__ import annotations
import pytest
import numpy as np
import tempfile
from pathlib import Path


from bci.decoder.deep import CNNDecoder


def _generate_epochs(n_epochs=20, n_channels=4, n_times=200, n_classes=2, seed=42):
    """Generate synthetic EEG epochs: (n_epochs, n_channels, n_times)"""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_epochs, n_channels, n_times)) * 10 + 50
    y = rng.integers(0, n_classes, size=n_epochs)
    return X, y


class TestCNNDecoder:
    """Tests for CNNDecoder fit/predict/save/load lifecycle."""

    @pytest.fixture
    def decoder(self):
        return CNNDecoder(epochs=5, lr=1e-3, dropout=0.25)

    @pytest.fixture
    def epochs(self):
        return _generate_epochs(n_epochs=20, n_channels=4, n_times=200, n_classes=2)

    def test_fit_trains_model(self, decoder, epochs):
        X, y = epochs
        decoder.fit(X, y)
        assert decoder.model is not None
        assert decoder._n_classes == 2
        assert decoder.classes_.shape == (2,)

    def test_predict_returns_class_labels(self, decoder, epochs):
        X, y = epochs
        decoder.fit(X, y)
        preds = decoder.predict(X)
        assert preds.shape == y.shape
        assert set(preds).issubset(set(y))

    def test_predict_proba_sum_to_one(self, decoder, epochs):
        X, y = epochs
        decoder.fit(X, y)
        probs = decoder.predict_proba(X)
        assert probs.shape == (X.shape[0], decoder._n_classes)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_predict_proba_requires_fit(self, decoder):
        with pytest.raises(RuntimeError, match="not fitted"):
            decoder.predict_proba(np.zeros((5, 4, 200)))

    def test_save_and_load_round_trip(self, decoder, epochs):
        X, y = epochs
        decoder.fit(X, y)
        preds_before = decoder.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cnn.pt"
            decoder.save(path)

            loaded = CNNDecoder.load(path)
            preds_after = loaded.predict(X)

        assert np.array_equal(preds_before, preds_after)
        assert loaded._n_classes == decoder._n_classes
        assert loaded.epochs == decoder.epochs
        assert loaded.lr == decoder.lr
        assert loaded.dropout == decoder.dropout

    def test_save_without_fit_raises(self, decoder):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cnn.pt"
            with pytest.raises(AttributeError):
                decoder.save(path)

    def test_multi_class_prediction(self):
        X, y = _generate_epochs(n_epochs=30, n_channels=4, n_times=200, n_classes=3)
        decoder = CNNDecoder(epochs=5)
        decoder.fit(X, y)
        preds = decoder.predict(X)
        assert set(preds).issubset({0, 1, 2})

    def test_input_shape_recorded(self, decoder, epochs):
        X, y = epochs
        decoder.fit(X, y)
        assert decoder._input_shape == (X.shape[1], X.shape[2])