"""
LDA Decoder
===========
StandardScaler + PCA + LDA pipeline. Handles high-dim EEG gracefully.
"""
from __future__ import annotations
import numpy as np
from bci.decoder.base import Decoder

try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    StandardScaler = None  # type: ignore[assignment,misc]
    PCA = None  # type: ignore[assignment,misc]
    LinearDiscriminantAnalysis = None  # type: ignore[assignment,misc]


class LDADecoder(Decoder):
    """LDA classifier with scaling + PCA dimensionality reduction.

    n_components: int or float in (0,1] — passed to PCA.
    Default 0.95 keeps 95% variance.
    """

    def __init__(self, n_components: float = 0.95, **kwargs):
        if not SKLEARN_OK:
            raise ImportError("scikit-learn required for LDADecoder")
        self.n_components = n_components
        assert StandardScaler is not None
        assert PCA is not None
        assert LinearDiscriminantAnalysis is not None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.clf = LinearDiscriminantAnalysis()

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LDADecoder':
        X2d = X.reshape(X.shape[0], -1)
        Xs = self.scaler.fit_transform(X2d)
        Xp = self.pca.fit_transform(Xs)
        self.clf.fit(Xp, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X2d = X.reshape(X.shape[0], -1)
        Xs = self.scaler.transform(X2d)
        Xp = self.pca.transform(Xs)
        return self.clf.predict(Xp)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X2d = X.reshape(X.shape[0], -1)
        Xs = self.scaler.transform(X2d)
        Xp = self.pca.transform(Xs)
        return self.clf.predict_proba(Xp)
