"""
Transformer BERT Decoder
========================
BERT-style bidirectional Transformer for EEG classification.

Single-variable ablation vs causal TransformerDecoder:
- Causal mask → bidirectional (no mask)
- Last-token classifier → learnable [CLS] token prepended to sequence,
  classifier reads CLS output

Everything else (TokenEmbedding, RoPE, FFN, optimizer, normalization,
auto kernel/stride, batching) is identical to TransformerDecoder so the
comparison isolates "attention direction + readout position".
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from bci.decoder.base import Decoder
from bci.decoder.transformer import (
    _TokenEmbedding,
    _RotaryPositionalEmbedding,
    _FeedForward,
)


class _BidirectionalMHA(nn.Module):
    """Bidirectional multi-head self-attention with RoPE (no causal mask)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float,
                 max_seq_len: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # NOTE: RoPE applied to Q/K of EEG-derived tokens.
        # CLS token (position 0) also gets RoPE rotation — same as treating
        # it as token at position 0 of a longer sequence. This is fine because
        # RoPE is relative, and CLS's relation to other tokens is well-defined.
        self.rope = _RotaryPositionalEmbedding(d_model, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        positions = torch.arange(T, device=x.device)
        Q = self.rope(Q, positions)
        K = self.rope(K, positions)
        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # NO causal mask — every position attends to every other position
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.resid_drop(self.out_proj(out))


class _EncoderBlock(nn.Module):
    """Pre-LN encoder block: bidirectional MHA + FFN with residuals."""

    def __init__(self, d_model: int, n_heads: int, dropout: float,
                 max_seq_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _BidirectionalMHA(d_model, n_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = _FeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class _EEGTransformerBert(nn.Module):
    """[CLS] + TokenEmbedding + N×EncoderBlock + classifier(CLS_output).

    Forward:
      (B, n_ch, n_times)
        → conv embed → (B, n_tokens, n_ch)
        → input proj → (B, n_tokens, d_model)
        → prepend CLS → (B, n_tokens+1, d_model)
        → N × bidirectional encoder blocks
        → take CLS at position 0 → (B, d_model)
        → classifier → (B, n_classes)
    """

    def __init__(self, n_channels: int, n_classes: int,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 3,
                 kernel: int = 20, stride: int = 10,
                 dropout: float = 0.2, max_seq_len: int = 1024):
        super().__init__()
        self.token_embed = _TokenEmbedding(n_channels, kernel, stride)
        self.input_proj = nn.Linear(n_channels, d_model)
        # Learnable [CLS] token, shared across the batch
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            _EncoderBlock(d_model, n_heads, dropout, max_seq_len)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.token_embed(x)            # (B, n_tokens, n_ch)
        x = self.input_proj(x)             # (B, n_tokens, d_model)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)     # (B, n_tokens+1, d_model)
        for block in self.blocks:
            x = block(x)
        cls_out = self.ln_final(x[:, 0, :])  # (B, d_model)
        return self.classifier(cls_out)


class TransformerBertDecoder(Decoder):
    """BERT-style bidirectional Transformer for EEG classification.

    Single-variable ablation vs TransformerDecoder (causal):
    - Attention: causal → bidirectional
    - Readout: last token → CLS token

    All other behavior (auto kernel/stride, normalization, mini-batch,
    save/load) matches TransformerDecoder for fair comparison.

    Length-agnostic at inference: accepts any n_times >= kernel.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        kernel: int | None = None,
        stride: int | None = None,
        target_tokens: int = 128,
        dropout: float = 0.2,
        epochs: int = 50,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        normalize: bool = True,
        device: str = 'cpu',
    ) -> None:
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        if kernel is not None and stride is not None:
            if kernel <= 0 or stride <= 0:
                raise ValueError(
                    f"kernel={kernel} and stride={stride} must be positive"
                )
            if kernel < stride:
                raise ValueError(
                    f"kernel={kernel} must be >= stride={stride} (no info loss)"
                )
        elif (kernel is None) != (stride is None):
            raise ValueError(
                "kernel and stride must both be set or both be None (auto)"
            )
        if target_tokens < 8:
            raise ValueError(f"target_tokens={target_tokens} too small (>= 8)")
        if batch_size < 1:
            raise ValueError(f"batch_size={batch_size} must be >= 1")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel = kernel
        self.stride = stride
        self.target_tokens = target_tokens
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device
        self.n_channels: int | None = None
        self.classes_: np.ndarray = np.array([])
        self._n_classes = 0
        self._train_n_tokens: int = 0
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self.model: nn.Module | None = None

    @staticmethod
    def _auto_kernel_stride(n_times: int, target_tokens: int) -> tuple[int, int]:
        """Same auto-pick logic as TransformerDecoder for fair comparison."""
        target = min(max(target_tokens, 16), 256)
        stride = max(1, n_times // target)
        kernel = stride * 2
        if kernel > n_times:
            kernel = n_times
            stride = max(1, n_times)
        return kernel, stride

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TransformerBertDecoder':
        if X.shape[0] != len(y):
            raise ValueError(
                f"X.shape[0]={X.shape[0]} != len(y)={len(y)}"
            )
        n_samples, n_channels, n_times = X.shape
        self.n_channels = n_channels

        if self.kernel is None or self.stride is None:
            self.kernel, self.stride = self._auto_kernel_stride(
                n_times, self.target_tokens
            )

        self._train_n_tokens = (n_times - self.kernel) // self.stride + 1
        classes, y_idx = np.unique(y, return_inverse=True)
        if len(classes) < 2:
            raise ValueError(
                f"Need at least 2 classes, got {len(classes)}"
            )
        self.classes_ = classes
        self._n_classes = len(classes)

        if self.normalize:
            self._mean = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
            self._std = X.std(axis=(0, 2), keepdims=True).astype(np.float32)
            self._std = np.where(self._std < 1e-8, 1.0, self._std)  # type: ignore[operator]  # self._std is ndarray, not None after assignment above
            X_norm = (X - self._mean) / self._std
        else:
            X_norm = X

        self.model = _EEGTransformerBert(
            n_channels=n_channels, n_classes=self._n_classes,
            d_model=self.d_model, n_heads=self.n_heads, n_layers=self.n_layers,
            kernel=self.kernel, stride=self.stride, dropout=self.dropout,
        ).to(self.device)
        assert self.model is not None
        opt = optim.AdamW(self.model.parameters(), lr=self.lr,
                          weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        Xt = torch.tensor(X_norm, dtype=torch.float32, device=self.device)
        yt = torch.tensor(y_idx, dtype=torch.long, device=self.device)

        bs = min(self.batch_size, n_samples)
        n_batches = (n_samples + bs - 1) // bs

        self.model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n_samples, device=self.device)
            for b in range(n_batches):
                idx = perm[b * bs:(b + 1) * bs]
                opt.zero_grad()
                loss = criterion(self.model(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        self.model.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Must call fit() before predict_proba()")
        if self.kernel is None or self.stride is None:
            raise RuntimeError("Must call fit() before predict_proba()")
        if X.shape[1] != self.n_channels:
            raise ValueError(
                f"n_channels mismatch: expected {self.n_channels}, "
                f"got {X.shape[1]}"
            )
        if X.shape[2] < self.kernel:
            raise ValueError(
                f"X.shape[2]={X.shape[2]} < kernel={self.kernel}; "
                f"need at least 1 token"
            )
        n_tokens = (X.shape[2] - self.kernel) // self.stride + 1
        if n_tokens > self._train_n_tokens:
            import warnings
            warnings.warn(
                f"n_tokens={n_tokens} > train n_tokens={self._train_n_tokens}; "
                f"RoPE position extrapolation; accuracy may degrade",
                stacklevel=2,
            )
        if self.normalize and self._mean is not None:
            X = (X - self._mean) / self._std
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(Xt)
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Must call fit() before save()")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            'model_state': self.model.state_dict(),
            'config': {
                'd_model': self.d_model,
                'n_heads': self.n_heads,
                'n_layers': self.n_layers,
                'kernel': self.kernel,
                'stride': self.stride,
                'target_tokens': self.target_tokens,
                'dropout': self.dropout,
                'batch_size': self.batch_size,
                'normalize': self.normalize,
                'n_channels': self.n_channels,
                'n_classes': self._n_classes,
                'train_n_tokens': self._train_n_tokens,
            },
            'classes_': self.classes_,
            'mean': self._mean,
            'std': self._std,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path) -> 'TransformerBertDecoder':
        state = torch.load(path, map_location='cpu', weights_only=False)
        cfg = state['config']
        obj = cls(
            d_model=cfg['d_model'], n_heads=cfg['n_heads'],
            n_layers=cfg['n_layers'], kernel=cfg['kernel'],
            stride=cfg['stride'],
            target_tokens=cfg.get('target_tokens', 128),
            dropout=cfg['dropout'],
            batch_size=cfg.get('batch_size', 32),
            normalize=cfg.get('normalize', True),
        )
        obj.n_channels = cfg['n_channels']
        obj._n_classes = cfg['n_classes']
        obj._train_n_tokens = cfg['train_n_tokens']
        obj.classes_ = state['classes_']
        obj._mean = state.get('mean')
        obj._std = state.get('std')
        obj.model = _EEGTransformerBert(
            n_channels=cfg['n_channels'], n_classes=cfg['n_classes'],
            d_model=cfg['d_model'], n_heads=cfg['n_heads'],
            n_layers=cfg['n_layers'], kernel=cfg['kernel'],
            stride=cfg['stride'], dropout=cfg['dropout'],
        )
        assert obj.model is not None
        obj.model.load_state_dict(state['model_state'])
        obj.model.eval()
        return obj
