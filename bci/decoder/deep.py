"""
Deep Learning Decoder
=====================
Simple 2D CNN for EEG classification using PyTorch.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from bci.decoder.base import Decoder


class _EEGCNN(nn.Module):
    """Lightweight 2D CNN: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> FC."""

    def __init__(self, n_channels: int, n_samples: int, n_classes: int,
                 dropout: float = 0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(n_channels, 3), padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 3), padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            x = self.bn2(self.conv2(self.bn1(self.conv1(dummy))))
            self._flatten_size = x.view(1, -1).shape[1]

        self.fc = nn.Linear(self._flatten_size, n_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class CNNDecoder(Decoder):
    """PyTorch 2D-CNN classifier for EEG epochs.

    Expects (n_epochs, n_channels, n_samples).
    """

    def __init__(self, epochs: int = 30, lr: float = 1e-3,
                 dropout: float = 0.25, device: str = 'cpu'):
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.device = device
        self.model: nn.Module | None = None
        self._n_classes = 0
        self.classes_: np.ndarray = np.array([])
        self._input_shape: tuple = ()

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CNNDecoder':
        n_epochs, n_channels, n_times = X.shape
        self._input_shape = (n_channels, n_times)
        classes, y_idx = np.unique(y, return_inverse=True)
        self._n_classes = len(classes)
        self.classes_ = classes

        self.model = _EEGCNN(n_channels, n_times, self._n_classes,
                             self.dropout).to(self.device)
        assert self.model is not None
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        Xt = torch.tensor(X[:, None, :, :], dtype=torch.float32,
                          device=self.device)
        yt = torch.tensor(y_idx, dtype=torch.long, device=self.device)

        self.model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = criterion(self.model(Xt), yt)
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
            raise RuntimeError("Model not fitted")
        Xt = torch.tensor(X[:, None, :, :], dtype=torch.float32,
                          device=self.device)
        with torch.no_grad():
            logits = self.model(Xt)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Must call fit() before save()")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            'model_state': self.model.state_dict(),
            'input_shape': self._input_shape,
            'n_classes': self._n_classes,
            'classes_': self.classes_,
            'epochs': self.epochs,
            'lr': self.lr,
            'dropout': self.dropout,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path) -> 'CNNDecoder':
        state = torch.load(path, map_location='cpu', weights_only=False)
        obj = cls(epochs=state['epochs'], lr=state['lr'],
                  dropout=state['dropout'])
        obj._input_shape = state['input_shape']
        obj._n_classes = state['n_classes']
        obj.classes_ = state['classes_']
        n_ch, n_t = obj._input_shape
        obj.model = _EEGCNN(n_ch, n_t, obj._n_classes, obj.dropout)
        assert obj.model is not None
        obj.model.load_state_dict(state['model_state'])
        obj.model.eval()
        return obj
