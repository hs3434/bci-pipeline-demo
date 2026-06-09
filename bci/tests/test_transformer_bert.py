"""Tests for TransformerBertDecoder (bidirectional + CLS readout)."""
from __future__ import annotations
import numpy as np
import pytest
import torch as t

from bci.decoder.transformer_bert import (
    TransformerBertDecoder,
    _BidirectionalMHA,
    _EEGTransformerBert,
)


@pytest.fixture
def epochs():
    """Small synthetic dataset: 30 trials, 4 channels, 200 timepoints, 2 classes."""
    rng = np.random.default_rng(0)
    n = 30
    X = rng.normal(size=(n, 4, 200)).astype(np.float32)
    y = np.array([0, 1] * (n // 2))
    # Inject class-conditional signal so model can actually learn
    X[y == 1, :, 50:100] += 1.5
    return X, y


class TestBidirectionalMHA:
    """Bidirectional MHA: every position attends to every other position."""

    def test_no_causal_mask_means_full_attention(self):
        mha = _BidirectionalMHA(d_model=16, n_heads=4, dropout=0.0,
                                max_seq_len=32)
        x = t.randn(2, 8, 16)
        out = mha(x)
        assert out.shape == (2, 8, 16)

    def test_position_zero_can_see_future(self):
        """In bidirectional attention, swapping later tokens changes pos-0 output."""
        t.manual_seed(0)
        mha = _BidirectionalMHA(d_model=16, n_heads=4, dropout=0.0,
                                max_seq_len=32)
        mha.eval()
        x = t.randn(1, 5, 16)
        out_a = mha(x)[:, 0, :]
        # Modify token at position 3 (would not affect causal pos-0, but affects bidirectional pos-0)
        x_mod = x.clone()
        x_mod[:, 3, :] += 10.0
        out_b = mha(x_mod)[:, 0, :]
        # Outputs must differ — proves position 0 sees position 3
        assert not t.allclose(out_a, out_b, atol=1e-5)


class TestEEGTransformerBertForward:
    """Forward pass with CLS prepending."""

    def test_output_shape_includes_cls_logic(self):
        model = _EEGTransformerBert(
            n_channels=4, n_classes=2,
            d_model=16, n_heads=4, n_layers=2,
            kernel=20, stride=10,
        )
        x = t.randn(3, 4, 200)
        out = model(x)
        assert out.shape == (3, 2)  # (batch, n_classes)

    def test_cls_token_is_learnable(self):
        model = _EEGTransformerBert(
            n_channels=4, n_classes=2,
            d_model=16, n_heads=4, n_layers=2,
            kernel=20, stride=10,
        )
        assert model.cls_token.requires_grad
        assert model.cls_token.shape == (1, 1, 16)


class TestTransformerBertDecoder:
    """End-to-end decoder behavior."""

    def test_fit_predict_proba_shapes(self, epochs):
        X, y = epochs
        dec = TransformerBertDecoder(
            d_model=16, n_heads=4, n_layers=2,
            kernel=20, stride=10, epochs=3, lr=1e-3,
        )
        dec.fit(X, y)
        probs = dec.predict_proba(X)
        assert probs.shape == (len(y), 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_returns_original_class_labels(self, epochs):
        X, y = epochs
        dec = TransformerBertDecoder(
            d_model=16, n_heads=4, n_layers=2,
            kernel=20, stride=10, epochs=3, lr=1e-3,
        )
        dec.fit(X, y)
        preds = dec.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_normalization_stats_persisted_after_fit(self, epochs):
        X, y = epochs
        dec = TransformerBertDecoder(
            d_model=16, n_heads=4, n_layers=2,
            kernel=20, stride=10, epochs=2, lr=1e-3, normalize=True,
        )
        dec.fit(X, y)
        assert dec._mean is not None
        assert dec._mean.shape == (1, 4, 1)
        assert dec._std is not None

    def test_save_load_roundtrip(self, epochs, tmp_path):
        X, y = epochs
        dec = TransformerBertDecoder(
            d_model=16, n_heads=4, n_layers=2,
            kernel=20, stride=10, epochs=2, lr=1e-3,
        )
        dec.fit(X, y)
        probs1 = dec.predict_proba(X)
        path = tmp_path / "bert_decoder.pt"
        dec.save(path)
        dec2 = TransformerBertDecoder.load(path)
        probs2 = dec2.predict_proba(X)
        np.testing.assert_allclose(probs1, probs2, atol=1e-5)

    def test_length_agnostic_inference(self, epochs):
        """Trained on 200 timepoints, can predict on shorter signals."""
        X, y = epochs
        dec = TransformerBertDecoder(
            d_model=16, n_heads=4, n_layers=2,
            kernel=20, stride=10, epochs=2, lr=1e-3,
        )
        dec.fit(X, y)
        X_short = X[:, :, :150]
        probs = dec.predict_proba(X_short)
        assert probs.shape == (len(y), 2)

    def test_auto_kernel_stride(self):
        dec = TransformerBertDecoder(epochs=1, target_tokens=128)
        X = np.random.randn(8, 4, 600).astype(np.float32)
        y = np.array([0, 1] * 4)
        dec.fit(X, y)
        # 600 // 128 = 4 → stride=4, kernel=8
        assert dec.stride == 4
        assert dec.kernel == 8

    def test_registry_creates_bert_decoder(self):
        from bci.decoder import create_decoder
        dec = create_decoder('transformer_bert', epochs=1)
        assert isinstance(dec, TransformerBertDecoder)
