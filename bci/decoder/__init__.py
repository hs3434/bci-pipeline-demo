"""
Decoder Module
===============
BCI Decoding with pluggable backends via Decoder ABC.

Supported methods:
    'lda'            — StandardScaler + PCA + LDA
    'ssvep'          — Single-band CCA
    'fbcca'          — Filter-Bank CCA
    'cnn'            — 2D CNN (PyTorch)
    'transformer'    — GPT-style causal + RoPE
    'transformer_bert' — BERT-style bidirectional + CLS
"""
from __future__ import annotations
from typing import List, Dict, Callable
import logging
import numpy as np
from dataclasses import dataclass

from bci.decoder.base import Decoder

logger = logging.getLogger(__name__)


@dataclass
class DecodeResult:
    accuracy: float
    std: float
    cv_scores: List[float]
    method: str


# ----------------------------------------------------------------
# Decoder registry — lazily resolves Decoder classes to avoid
# eager-loading heavy dependencies (torch, sklearn, MNE) at
# import time.
# ----------------------------------------------------------------

_registry: Dict[str, Callable[..., Decoder]] = {}


def _lazy(module_path: str, class_name: str) -> Callable[..., Decoder]:
    """Return a factory that lazy-imports and instantiates a Decoder."""
    def factory(**kw) -> Decoder:
        mod = __import__(module_path, fromlist=[class_name])
        cls = getattr(mod, class_name)
        return cls(**kw)
    return factory


_registry['lda'] = _lazy('bci.decoder.lda', 'LDADecoder')
_registry['ssvep'] = _lazy('bci.decoder.ssvep', 'SSVEPDecoder')
_registry['fbcca'] = _lazy('bci.decoder.ssvep', 'FBCCADecoder')
_registry['cnn'] = _lazy('bci.decoder.deep', 'CNNDecoder')
_registry['transformer'] = _lazy('bci.decoder.transformer', 'TransformerDecoder')
_registry['transformer_bert'] = _lazy('bci.decoder.transformer_bert',
                                       'TransformerBertDecoder')


def list_methods() -> List[str]:
    return sorted(_registry.keys())


# ----------------------------------------------------------------
# Top-level API
# ----------------------------------------------------------------

def decode(epochs_data: np.ndarray, labels: np.ndarray,
           method: str = 'lda', cv_folds: int = 5,
           **decoder_kwargs) -> DecodeResult:
    """Decode EEG epochs.

    Args:
        epochs_data: (n_epochs, n_channels, n_samples)
        labels: (n_epochs,) class labels
        method: 'lda' | 'ssvep' | 'fbcca' | 'cnn' | 'transformer'
        cv_folds: cross-validation folds (ignored for SSVEP/FBCCA)
        **decoder_kwargs: passed to decoder constructor

    Returns:
        DecodeResult with accuracy, std, per-fold cv_scores
    """
    from sklearn.model_selection import StratifiedKFold

    if method not in _registry:
        raise ValueError(
            f"Unknown method '{method}'. Available: {list_methods()}"
        )

    factory = _registry[method]
    decoder = factory(**decoder_kwargs)

    if method in ('ssvep', 'fbcca'):
        decoder.fit(epochs_data, labels)
        preds = decoder.predict(epochs_data)
        acc = float(np.mean(preds == labels))
        return DecodeResult(accuracy=acc, std=0.0,
                            cv_scores=[acc], method=method)

    cv = StratifiedKFold(n_splits=min(cv_folds, len(labels)),
                          shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in cv.split(epochs_data, labels):
        fold_decoder = factory(**decoder_kwargs)
        fold_decoder.fit(epochs_data[train_idx], labels[train_idx])
        preds = fold_decoder.predict(epochs_data[test_idx])
        scores.append(float(np.mean(preds == labels[test_idx])))

    return DecodeResult(
        accuracy=float(np.mean(scores)),
        std=float(np.std(scores)),
        cv_scores=scores,
        method=method,
    )


def create_decoder(method: str, **kwargs) -> Decoder:
    """Create and return a decoder instance (without training)."""
    if method not in _registry:
        raise ValueError(
            f"Unknown method '{method}'. Available: {list_methods()}"
        )
    return _registry[method](**kwargs)
