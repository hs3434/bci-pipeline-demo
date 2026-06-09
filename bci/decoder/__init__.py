"""
Decoder Module
===============
BCI Decoding with pluggable backends via Decoder ABC.

Supported methods:
    'lda'   — StandardScaler + PCA + LDA
    'ssvep' — Single-band CCA
    'fbcca' — Filter-Bank CCA
    'cnn'   — 2D CNN (PyTorch)
"""
from __future__ import annotations
from typing import List, Dict, Type
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
# Decoder registry
# ----------------------------------------------------------------

_registry: Dict[str, Type[Decoder]] = {}


def register(method: str):
    """Decorator: register a Decoder subclass under a method name."""
    def wrapper(cls: Type[Decoder]) -> Type[Decoder]:
        _registry[method] = cls
        return cls
    return wrapper


def list_methods() -> List[str]:
    return sorted(_registry.keys())


# ----------------------------------------------------------------
# Auto-register built-in decoders
# ----------------------------------------------------------------

@register('lda')
class _LDADecoder:
    @staticmethod
    def create(**kw):
        from bci.decoder.lda import LDADecoder
        return LDADecoder(**kw)


@register('ssvep')
class _SSVEPDecoder:
    @staticmethod
    def create(**kw):
        from bci.decoder.ssvep import SSVEPDecoder
        return SSVEPDecoder(**kw)


@register('fbcca')
class _FBCCADecoder:
    @staticmethod
    def create(**kw):
        from bci.decoder.ssvep import FBCCADecoder
        return FBCCADecoder(**kw)


@register('cnn')
class _CNNDecoder:
    @staticmethod
    def create(**kw):
        from bci.decoder.deep import CNNDecoder
        return CNNDecoder(**kw)


@register('transformer')
class _TransformerDecoder:
    @staticmethod
    def create(**kw):
        from bci.decoder.transformer import TransformerDecoder
        return TransformerDecoder(**kw)


@register('transformer_bert')
class _TransformerBertDecoder:
    @staticmethod
    def create(**kw):
        from bci.decoder.transformer_bert import TransformerBertDecoder
        return TransformerBertDecoder(**kw)


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

    decoder_cls = _registry[method]
    decoder = decoder_cls.create(**decoder_kwargs)

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
        fold_decoder = decoder_cls.create(**decoder_kwargs)
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
    return _registry[method].create(**kwargs)
