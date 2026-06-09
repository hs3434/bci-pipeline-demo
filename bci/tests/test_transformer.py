"""
Tests for bci.decoder.transformer module (TransformerDecoder)
"""
from __future__ import annotations
import pytest
import numpy as np


from bci.decoder.transformer import TransformerDecoder


def _generate_epochs(n_epochs=20, n_channels=4, n_times=200, n_classes=2, seed=42):
    """Synthetic EEG epochs with class-discriminative time-locked signal."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_epochs, n_channels, n_times)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n_epochs)
    for cls in range(n_classes):
        mask = y == cls
        X[mask, :, 50:100] += (cls + 1) * 0.5
    return X, y


class TestTransformerDecoder:
    """Unit + integration tests for TransformerDecoder."""

    @pytest.fixture
    def decoder(self):
        return TransformerDecoder(d_model=64, n_heads=4, n_layers=2,
                                   epochs=3, lr=1e-3)

    @pytest.fixture
    def epochs(self):
        return _generate_epochs(n_epochs=16, n_channels=4, n_times=200,
                                n_classes=2)

    def test_module_imports(self):
        assert TransformerDecoder is not None

    def test_constructor_validates_d_model_divisible_by_n_heads(self):
        with pytest.raises(ValueError, match="d_model"):
            TransformerDecoder(d_model=64, n_heads=3)

    def test_constructor_validates_kernel_ge_stride(self):
        with pytest.raises(ValueError, match="kernel"):
            TransformerDecoder(kernel=5, stride=10)

    def test_constructor_validates_positive_kernel_stride(self):
        with pytest.raises(ValueError):
            TransformerDecoder(kernel=0, stride=10)
        with pytest.raises(ValueError):
            TransformerDecoder(kernel=10, stride=0)

    def test_fit_infers_n_channels_and_classes(self, decoder, epochs):
        X, y = epochs
        decoder.fit(X, y)
        assert decoder.n_channels == 4
        assert decoder._n_classes == 2
        assert tuple(decoder.classes_.shape) == (2,)
        assert decoder.model is not None

    def test_fit_rejects_mismatched_x_y(self, decoder, epochs):
        X, _ = epochs
        with pytest.raises(ValueError):
            decoder.fit(X, np.array([0, 1, 2]))

    def test_fit_rejects_few_classes(self, decoder):
        X = np.random.randn(10, 4, 200).astype(np.float32)
        y = np.zeros(10, dtype=np.int64)
        with pytest.raises(ValueError, match="classes"):
            decoder.fit(X, y)

    def test_predict_returns_labels(self, decoder, epochs):
        X, y = epochs
        decoder.fit(X, y)
        preds = decoder.predict(X)
        assert preds.shape == y.shape
        assert set(np.unique(preds)).issubset(set(y))

    def test_predict_proba_sums_to_one(self, decoder, epochs):
        X, _ = epochs
        decoder.fit(X, np.array([0] * 8 + [1] * 8))
        probs = decoder.predict_proba(X)
        assert probs.shape == (16, 2)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_proba_before_fit_raises(self, decoder):
        X = np.random.randn(2, 4, 200).astype(np.float32)
        with pytest.raises(RuntimeError, match="fit"):
            decoder.predict_proba(X)

    def test_predict_proba_n_channels_mismatch_raises(self, decoder, epochs):
        X, y = epochs
        decoder.fit(X, y)
        X_bad = np.random.randn(2, 8, 200).astype(np.float32)
        with pytest.raises(ValueError, match="n_channels"):
            decoder.predict_proba(X_bad)

    def test_predict_proba_n_times_too_small_raises(self, epochs):
        # Use explicit kernel to make "too small" check deterministic
        # (auto-mode with n_times=200 → kernel=2, too small for this test)
        decoder = TransformerDecoder(d_model=64, n_heads=4, n_layers=2,
                                     kernel=20, stride=10, epochs=3, lr=1e-3)
        X, y = epochs
        decoder.fit(X, y)
        X_bad = np.random.randn(2, 4, 5).astype(np.float32)
        with pytest.raises(ValueError, match="kernel"):
            decoder.predict_proba(X_bad)

    def test_predict_proba_warns_on_rope_extrapolation(self, decoder, epochs):
        X, y = epochs
        decoder.fit(X, y)
        X_long = np.random.randn(2, 4, 800).astype(np.float32)
        with pytest.warns(UserWarning, match="extrapolation"):
            decoder.predict_proba(X_long)

    def test_length_agnostic_inference(self, decoder, epochs):
        X, y = epochs
        decoder.fit(X, y)
        for n_times in (100, 500, 200, 1000):
            Xi = np.random.randn(3, 4, n_times).astype(np.float32)
            probs = decoder.predict_proba(Xi)
            assert probs.shape == (3, 2)

    def test_save_load_roundtrip(self, decoder, epochs, tmp_path):
        X, y = epochs
        decoder.fit(X, y)
        preds_before = decoder.predict(X)
        path = tmp_path / "transformer.pt"
        decoder.save(path)
        assert path.exists()
        loaded = TransformerDecoder.load(path)
        preds_after = loaded.predict(X)
        np.testing.assert_array_equal(preds_before, preds_after)

    def test_save_uses_torch_format(self, decoder, epochs, tmp_path):
        """Save format must be torch.save (consistent with CNNDecoder)."""
        import torch as t
        X, y = epochs
        decoder.fit(X, y)
        path = tmp_path / "transformer.pt"
        decoder.save(path)
        state = t.load(path, map_location='cpu', weights_only=False)
        assert 'model_state' in state
        assert 'config' in state

    def test_decode_integration(self, epochs):
        from bci.decoder import decode
        X, y = epochs
        result = decode(X, y, method='transformer',
                        d_model=64, n_heads=4, n_layers=2, epochs=3)
        assert result.method == 'transformer'
        assert 0.0 <= result.accuracy <= 1.0
        assert len(result.cv_scores) > 0

    def test_registered_in_registry(self):
        from bci.decoder import list_methods, decode
        assert 'transformer' in list_methods()
        X = np.random.randn(20, 4, 200).astype(np.float32)
        y = np.array([0] * 10 + [1] * 10)
        result = decode(X, y, method='transformer',
                        d_model=64, n_heads=4, n_layers=1, epochs=1)
        assert result is not None


class TestTokenEmbedding:
    """Conv1d token embedding output length formula."""

    def test_output_length_with_default_kernel_stride(self):
        import torch as t
        from bci.decoder.transformer import _TokenEmbedding
        emb = _TokenEmbedding(n_channels=4, kernel=20, stride=10)
        x = t.zeros(2, 4, 1000)
        out = emb(x)
        assert tuple(out.shape) == (2, 99, 4)

    def test_output_length_with_minimum_input(self):
        import torch as t
        from bci.decoder.transformer import _TokenEmbedding
        emb = _TokenEmbedding(n_channels=4, kernel=20, stride=10)
        x = t.zeros(1, 4, 20)
        out = emb(x)
        assert tuple(out.shape) == (1, 1, 4)


class TestCausalMask:
    """Causal mask shape and values."""

    def test_mask_shape_and_values(self):
        from bci.decoder.transformer import _CausalMask
        m = _CausalMask(5)
        mask = m(5)
        assert tuple(mask.shape) == (5, 5)
        for q in range(5):
            for k in range(5):
                if k > q:
                    assert mask[q, k].item() == float("-inf")
                else:
                    assert mask[q, k].item() == 0.0


class TestRoPE:
    """Rotary position embedding invariants."""

    def test_position_zero_is_identity(self):
        from bci.decoder.transformer import _RotaryPositionalEmbedding
        import torch as t
        rope = _RotaryPositionalEmbedding(d_model=8, max_seq_len=16)
        q = t.randn(1, 3, 8)
        q_rot = rope(q, positions=t.tensor([0, 0, 0]))
        assert t.allclose(q_rot, q, atol=1e-5)

    def test_relative_position_invariant(self):
        from bci.decoder.transformer import _RotaryPositionalEmbedding
        import torch as t
        rope = _RotaryPositionalEmbedding(d_model=8, max_seq_len=32)
        q = t.randn(1, 1, 8)
        k = t.randn(1, 1, 8)
        q1 = rope(q, positions=t.tensor([5]))
        k1 = rope(k, positions=t.tensor([3]))
        q2 = rope(q, positions=t.tensor([7]))
        k2 = rope(k, positions=t.tensor([5]))
        score1 = (q1 * k1).sum().item()
        score2 = (q2 * k2).sum().item()
        assert abs(score1 - score2) < 1e-5


class TestEEGTransformerForward:
    """Full model forward pass — shape and length-agnostic."""

    def test_forward_shape_default_size(self):
        import torch as t
        from bci.decoder.transformer import _EEGTransformer
        model = _EEGTransformer(n_channels=4, n_classes=2, d_model=64,
                                n_heads=4, n_layers=2, kernel=20, stride=10,
                                dropout=0.0, max_seq_len=128)
        x = t.randn(3, 4, 1000)
        out = model(x)
        assert tuple(out.shape) == (3, 2)

    def test_forward_handles_different_n_times(self):
        import torch as t
        from bci.decoder.transformer import _EEGTransformer
        model = _EEGTransformer(n_channels=4, n_classes=2, d_model=64,
                                n_heads=4, n_layers=2, kernel=20, stride=10,
                                dropout=0.0, max_seq_len=256)
        for n_times in (100, 500, 1000):
            x = t.randn(2, 4, n_times)
            out = model(x)
            assert tuple(out.shape) == (2, 2)


class TestAutoKernelStride:
    """Auto-pick kernel/stride so n_tokens stays in target range."""

    def test_auto_picks_kernel_stride_when_none(self):
        dec = TransformerDecoder(epochs=1, target_tokens=128)
        X = np.random.randn(8, 4, 600).astype(np.float32)
        y = np.array([0, 1] * 4)
        dec.fit(X, y)
        # 600 // 128 = 4 → stride=4, kernel=8 → tokens=(600-8)/4+1=149
        assert dec.stride == 4
        assert dec.kernel == 8
        assert 100 <= dec._train_n_tokens <= 150

    def test_explicit_kernel_stride_override_auto(self):
        dec = TransformerDecoder(epochs=1, kernel=20, stride=10)
        X = np.random.randn(8, 4, 600).astype(np.float32)
        y = np.array([0, 1] * 4)
        dec.fit(X, y)
        assert dec.kernel == 20 and dec.stride == 10

    def test_auto_for_short_signal_uses_max_resolution(self):
        # n_times=50, target=128 → stride=max(1, 50//128)=1, kernel=2
        dec = TransformerDecoder(epochs=1, target_tokens=128)
        X = np.random.randn(8, 4, 50).astype(np.float32)
        y = np.array([0, 1] * 4)
        dec.fit(X, y)
        assert dec.stride == 1 and dec.kernel == 2

    def test_partial_kernel_stride_raises(self):
        with pytest.raises(ValueError, match="both"):
            TransformerDecoder(kernel=10, stride=None)
