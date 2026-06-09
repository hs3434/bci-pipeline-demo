"""
CSP Decoder
===========
Common Spatial Pattern + LDA for Motor Imagery classification.

Pipeline: CSP(log-var features) → StandardScaler → LDA
Reference: Koles et al., 1990; Ramoser et al., 2000
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from bci.decoder.base import Decoder


class CSPDecoder(Decoder):
    """CSP + log-var + LDA decoder for Motor Imagery.

    Uses mne.decoding.CSP under the hood.
    Expects (n_epochs, n_channels, n_samples).
    """

    def __init__(self, n_components: int = 4, reg: str | None = None,
                 log: bool = True, random_state: int = 42):
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.random_state = random_state
        self.pipeline: SkPipeline | None = None
        self.classes_: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CSPDecoder':
        from mne.decoding import CSP
        classes = np.unique(y)
        self.classes_ = classes
        self.pipeline = SkPipeline([
            ('csp', CSP(n_components=self.n_components, reg=self.reg,
                        log=self.log, random_state=self.random_state)),
            ('scaler', StandardScaler()),
            ('lda', LinearDiscriminantAnalysis()),
        ])
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Must call fit() before predict()")
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Must call fit() before predict_proba()")
        return self.pipeline.predict_proba(X)

    def save(self, path: str | Path) -> None:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'pipeline': self.pipeline,
            'classes_': self.classes_,
            'n_components': self.n_components,
            'reg': self.reg,
            'log': self.log,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> 'CSPDecoder':
        import joblib
        state = joblib.load(path)
        obj = cls(n_components=state['n_components'],
                  reg=state['reg'], log=state['log'])
        obj.pipeline = state['pipeline']
        obj.classes_ = state['classes_']
        return obj
