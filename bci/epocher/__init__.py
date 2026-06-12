"""
Epocher Module
==============
Event detection and epoch extraction
"""

from __future__ import annotations
from typing import Optional, Dict, Tuple, List, TYPE_CHECKING
import logging
import numpy as np
from dataclasses import dataclass

if TYPE_CHECKING:
    import mne
    from bci.config import EpochConfig

logger = logging.getLogger(__name__)


@dataclass
class EpochStats:
    """Epoch extraction statistics"""
    n_epochs: int
    n_rejected: int
    rejection_rate: float
    duration: float


class Epocher:
    """Event/Epoch processor

    Examples:
        >>> from bci.epocher import Epocher
        >>> from bci.config import EpochConfig
        >>> epocher = Epocher(raw, EpochConfig())
        >>> events = epocher.find_events()
        Found 45 events
        >>> epochs = epocher.extract_epochs(events, {'left': 1, 'right': 2})
        >>> stats = epocher.get_stats()
        >>> print(f"Epochs: {stats.n_epochs}, rejected: {stats.rejection_rate:.1%}")
        Epochs: 40, rejected: 11.1%

        # Using convenience function
        >>> from bci.epocher import create_epochs
        >>> epochs = create_epochs(raw, events, {'left': 1, 'right': 2}, tmin=-0.2, tmax=0.5)
    """

    def __init__(self, raw: mne.io.Raw, config: Optional[EpochConfig] = None):
        self.raw = raw
        self.config = config
        self.events: Optional[np.ndarray] = None
        self.epochs: Optional[mne.Epochs] = None

    def find_events(self, stim_channel: Optional[str] = None,
                    min_duration: int = 1,
                    prefer_annotations: bool = False) -> np.ndarray:
        """Find events in the data.

        Args:
            stim_channel: Stimulus channel name. If None, uses MNE default.
            min_duration: Minimum event duration (samples).
            prefer_annotations: If True, use annotations first instead of
                trying stim channel.

        Returns:
            Events array (n_events, 3)
        """
        import mne

        if prefer_annotations:
            self.events, event_id = mne.events_from_annotations(self.raw)
            logger.info(f"Found event IDs from annotations: {event_id}")
        else:
            try:
                self.events = mne.find_events(
                    self.raw,
                    stim_channel=stim_channel,
                    min_duration=min_duration,
                )
            except ValueError:
                logger.info("No stim channel, trying annotations...")
                self.events, event_id = mne.events_from_annotations(self.raw)
                logger.info(f"Found event IDs from annotations: {event_id}")

        if self.events is None or len(self.events) == 0:
            raise RuntimeError("No events found in data")
        logger.info(f"Found {len(self.events)} events")
        return self.events

    def extract_epochs(self, events: Optional[np.ndarray] = None,
                      event_id: Optional[Dict[str, int]] = None,
                      tmin: float = -0.2, tmax: float = 0.5,
                      baseline: Tuple[Optional[float], Optional[float]] = (None, 0),
                      preload: bool = True) -> mne.Epochs:
        """Extract epochs around events

        Args:
            events: Events array. If None, uses self.events
            event_id: Event ID mapping (e.g., {'left': 1, 'right': 2})
            tmin: Start time relative to event (s)
            tmax: End time relative to event (s)
            baseline: Baseline correction window
            preload: Whether to load data into memory

        Returns:
            MNE Epochs object
        """
        if events is None:
            events = self.events
        if events is None:
            raise ValueError("No events found, call find_events() first")

        if event_id is None:
            unique_events = np.unique(events[:, 2])
            event_id = {f'event_{int(e)}': int(e) for e in unique_events}

        logger.info(f"Extracting epochs: {tmin}s to {tmax}s, baseline={baseline}")

        from mne import Epochs

        reject = self.config.reject_threshold if self.config else None
        if reject:
            valid_types = set(self.raw.get_channel_types())
            reject = {k: v for k, v in reject.items() if k in valid_types}

        self.epochs = Epochs(
            self.raw, events, event_id,
            tmin=tmin, tmax=tmax,
            baseline=baseline,
            preload=preload,
            reject=reject
        )

        n_rejected = len(self.epochs.drop_log) if self.epochs is not None and self.epochs.drop_log is not None else 0
        n_total = len(self.epochs) if self.epochs is not None else 0
        pct = (n_rejected / n_total * 100) if n_total else 0
        logger.info(f"Extracted {n_total} epochs, "
                   f"{n_rejected} rejected ({pct:.1f}%)")

        return self.epochs

    def get_stats(self) -> EpochStats:
        """Get epoch extraction statistics"""
        if self.epochs is None:
            raise RuntimeError("No epochs extracted")
        n_rejected = len(self.epochs.drop_log)
        return EpochStats(
            n_epochs=len(self.epochs),
            n_rejected=n_rejected,
            rejection_rate=n_rejected / len(self.epochs) if len(self.epochs) > 0 else 0,
            duration=float(self.epochs.tmax - self.epochs.tmin)
        )

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get epochs data and labels

        Returns:
            (data, labels) tuple
        """
        if self.epochs is None:
            raise RuntimeError("No epochs extracted")
        data = self.epochs.get_data()
        labels = self.epochs.events[:, 2]
        return data, labels


def create_epochs(raw: mne.io.Raw, events: np.ndarray,
                 event_id: Dict[str, int], **kwargs) -> mne.Epochs:
    """Convenience function to create epochs"""
    epocher = Epocher(raw)
    return epocher.extract_epochs(events, event_id, **kwargs)