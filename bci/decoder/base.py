"""
Abstract Decoder Base
=====================
All decoders inherit from this ABC with fit/predict/predict_proba/save/load.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import pickle
from pathlib import Path
import numpy as np


class Decoder(ABC):
    """Abstract base for all BCI decoders.

    Subclasses implement:
      - fit(X, y)         : train on (n_epochs, n_channels, n_samples)
      - predict(X) -> [int]: class indices
      - predict_proba(X) -> [[float]]: per-class probabilities

    Serialisation is pickle by default; subclasses can override.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Decoder':
        """Train the decoder."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities, shape (n_samples, n_classes)."""
        ...

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> 'Decoder':
        with open(path, 'rb') as f:
            return pickle.load(f)
