"""
SSVEP Decoders
==============
Single-band CCA and Filter-Bank CCA (FBCCA) for SSVEP detection.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
from bci.decoder.base import Decoder


class SSVEPDecoder(Decoder):
    """Single-band CCA for SSVEP detection.

    Does not require training — fit() is a no-op.
    predict() returns argmax over per-frequency CCA scores.
    """

    def __init__(self, target_freqs: Sequence[float], fs: float,
                 n_harmonics: int = 5):
        self.target_freqs = list(target_freqs)
        self.fs = fs
        self.n_harmonics = n_harmonics
        self.classes_ = np.array([str(f) + 'Hz' for f in target_freqs])
        self._templates: dict[float, float] = {}
        self._low: float = 0.0
        self._high: float = 0.0
        self._generate_templates()

    def _generate_templates(self):
        for freq in self.target_freqs:
            self._templates[freq] = freq

    def _get_template(self, freq: float, n_samples: int) -> np.ndarray:
        t = np.arange(n_samples) / self.fs
        refs = []
        for h in range(1, self.n_harmonics + 1):
            refs.append(np.sin(2 * np.pi * h * freq * t))
            refs.append(np.cos(2 * np.pi * h * freq * t))
        return np.array(refs)

    def _cca_score(self, data: np.ndarray, freq: float) -> float:
        try:
            X = data
            Y = self._get_template(freq, X.shape[1])
            C_xx = np.cov(X)
            C_yy = np.cov(Y)
            C_xx_inv = np.linalg.inv(C_xx)
            C_yy_inv = np.linalg.inv(C_yy)
            corr = np.corrcoef(C_xx_inv @ X @ Y.T @ C_yy_inv @ Y)
            r = float(corr[0, 1]) if corr.ndim == 2 else 0.0
            return abs(r) if not np.isnan(r) else 0.0
        except np.linalg.LinAlgError:
            return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SSVEPDecoder':
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.predict_proba(X)
        return np.argmax(scores, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0] if X.ndim == 3 else 1
        X2d = X.reshape(n_samples, X.shape[-2], X.shape[-1])
        scores = np.zeros((n_samples, len(self.target_freqs)))
        for i in range(n_samples):
            for j, freq in enumerate(self.target_freqs):
                scores[i, j] = self._cca_score(X2d[i], freq)
        return scores


class FBCCADecoder(Decoder):
    """Filter-Bank CCA for improved SSVEP detection.

    Decomposes signal into sub-bands, runs CCA per band,
    combines with weighted sum (weight ∝ n^(-1.25) + 0.25).
    """

    def __init__(self, target_freqs: Sequence[float], fs: float,
                 n_harmonics: int = 5, n_bands: int = 5):
        self.target_freqs = list(target_freqs)
        self.fs = fs
        self.n_harmonics = n_harmonics
        self.n_bands = n_bands
        self.classes_ = np.array([str(f) + 'Hz' for f in target_freqs])
        self._sub_bands: list[SSVEPDecoder] = []
        self._weights: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FBCCADecoder':
        self._sub_bands = []
        for k in range(1, self.n_bands + 1):
            low = k * 8.0
            high = min(90.0, low + 8.0)
            sub = SSVEPDecoder(self.target_freqs, self.fs, self.n_harmonics)
            sub._low = low
            sub._high = high
            self._sub_bands.append(sub)
        self._weights = np.array([k ** (-1.25) + 0.25
                                   for k in range(1, self.n_bands + 1)])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.predict_proba(X)
        return np.argmax(scores, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0] if X.ndim == 3 else 1
        X2d = X.reshape(n_samples, X.shape[-2], X.shape[-1])

        all_scores = np.zeros((n_samples, len(self.target_freqs)))
        for k, sub in enumerate(self._sub_bands):
            filtered = self._bandpass(X2d, sub._low, sub._high)
            for i in range(n_samples):
                for j, freq in enumerate(self.target_freqs):
                    all_scores[i, j] += (self._weights[k] *
                                         sub._cca_score(filtered[i], freq))
        return all_scores

    def _bandpass(self, data: np.ndarray, low: float, high: float
                  ) -> np.ndarray:
        from scipy.signal import butter, filtfilt
        nyq = 0.5 * self.fs
        b, a = butter(4, [low / nyq, high / nyq], btype='band')  # type: ignore[assignment]  # scipy butter stub returns complex union
        return filtfilt(b, a, data, axis=-1)
