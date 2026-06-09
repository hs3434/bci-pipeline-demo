"""
Pipeline Module
================
BCI Pipeline Orchestrator
"""

from __future__ import annotations
from typing import Optional, Dict, List, TYPE_CHECKING
from pathlib import Path
import logging
from dataclasses import dataclass, field

if TYPE_CHECKING:
    import numpy as np
    import mne
    from bci.config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Pipeline execution result

    Attributes:
        success: Whether pipeline completed successfully
        accuracy: Classification accuracy (0-1), None if failed
        std: Standard deviation across CV folds
        steps_completed: List of pipeline steps that succeeded
        errors: List of error messages if failed
        output_files: Paths to saved output files

    Examples:
        >>> result = run_pipeline(config, 'data.edf')
        >>> if result.success:
        ...     print(f"Accuracy: {result.accuracy:.3f} +/- {result.std:.3f}")
        ...     print(f"Steps: {result.steps_completed}")
        ... else:
        ...     print(f"Failed: {result.errors}")
    """
    success: bool
    accuracy: Optional[float] = None
    std: Optional[float] = None
    cv_scores: List[float] = field(default_factory=list)
    steps_completed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    output_files: List[Path] = field(default_factory=list)


class BCIPipeline:
    """
    BCI Signal Processing Pipeline
    ================================
    Orchestrates: Load → Preprocess → Epoch → Decode → Report

    Examples:
        # Method 1: Convenience function
        >>> from bci.config import create_default_config
        >>> from bci.pipeline import run_pipeline
        >>> config = create_default_config()
        >>> result = run_pipeline(config, 'data.edf')
        >>> print(f"Accuracy: {result.accuracy:.3f}")

        # Method 2: Step-by-step (for debugging/interactive use)
        >>> from bci.config import create_default_config
        >>> from bci.pipeline import BCIPipeline
        >>> config = create_default_config()
        >>> pipeline = BCIPipeline(config)
        >>> pipeline.load('data.edf').preprocess().create_epochs().decode()
        >>> print(f"Accuracy: {pipeline.result.accuracy:.3f}")

        # Method 3: With custom events and event_id
        >>> import numpy as np
        >>> from bci.config import create_default_config
        >>> from bci.pipeline import BCIPipeline
        >>> config = create_default_config()
        >>> events = np.array([[0, 0, 1], [100, 0, 2], [200, 0, 1]])
        >>> event_id = {'left': 1, 'right': 2}
        >>> pipeline = BCIPipeline(config)
        >>> result = pipeline.run('data.edf', events=events, event_id=event_id)
    """

    def __init__(self, config: 'PipelineConfig'):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.raw: Optional['mne.io.Raw'] = None
        self.epochs: Optional['mne.Epochs'] = None
        self.events: Optional['np.ndarray'] = None
        self.result: Optional[PipelineResult] = None

        self._steps: List[str] = []

    def load(self, filepath: Path | str) -> 'BCIPipeline':
        """Load EEG data"""
        from bci.source import FileSource

        self.logger.info(f"Loading data: {filepath}")
        try:
            raw_data = FileSource.load_raw(filepath)
            self.raw = raw_data
            self._steps.append('load')
            self.logger.info(f"Loaded: {len(self.raw.ch_names)} channels")
            return self
        except Exception as e:
            self.logger.error(f"Load failed: {e}")
            raise

    def preprocess(self) -> 'BCIPipeline':
        """Preprocess data"""
        from bci.preprocessor import preprocess

        self.logger.info("Preprocessing")
        try:
            if self.raw is None:
                raise RuntimeError("No raw data loaded, call load() first")
            self.raw = preprocess(self.raw, self.config.filter)
            self._steps.append('preprocess')
            self.logger.info("Preprocessing done")
            return self
        except Exception as e:
            self.logger.error(f"Preprocess failed: {e}")
            raise

    def create_epochs(self, events: Optional['np.ndarray'] = None,
                      event_id: Optional[Dict[str, int]] = None) -> 'BCIPipeline':
        """Create epochs"""
        from bci.epocher import Epocher

        self.logger.info("Creating epochs")
        try:
            if self.raw is None:
                raise RuntimeError("No raw data preprocessed, call preprocess() first")
            epocher = Epocher(self.raw, self.config.epoch)

            if events is None:
                events = epocher.find_events()
                self.events = events

            self.epochs = epocher.extract_epochs(
                events, event_id,
                tmin=self.config.epoch.tmin,
                tmax=self.config.epoch.tmax,
                baseline=self.config.epoch.baseline
            )
            self._steps.append('create_epochs')
            self.logger.info(f"Created {len(self.epochs)} epochs")
            return self
        except Exception as e:
            self.logger.error(f"Epoch creation failed: {e}")
            raise

    def decode(self) -> 'BCIPipeline':
        """Decode epochs"""
        from bci.decoder import decode as decode_fn

        self.logger.info("Decoding")
        try:
            if self.epochs is None:
                raise RuntimeError("No epochs available for decoding")
            data = self.epochs.get_data()
            labels = self.epochs.events[:, 2]

            n_epochs = len(labels)
            if n_epochs < 2:
                raise RuntimeError(
                    f"Need at least 2 epochs for decoding, got {n_epochs}. "
                    "Try a longer recording or adjust event detection parameters."
                )
            cv_folds = min(self.config.decode.cv_folds, n_epochs)
            if cv_folds < self.config.decode.cv_folds:
                self.logger.warning(
                    f"Reducing cv_folds from {self.config.decode.cv_folds} "
                    f"to {cv_folds} (only {n_epochs} epochs)"
                )

            sfreq = self.epochs.info['sfreq']
            decoder_kwargs: dict = {}
            if self.config.decode.method in ('ssvep', 'fbcca'):
                decoder_kwargs['target_freqs'] = sorted(set(labels))
                decoder_kwargs['fs'] = sfreq
            result = decode_fn(data, labels, method=self.config.decode.method,
                               cv_folds=cv_folds, **decoder_kwargs)

            model_path = Path(self.config.output_dir) / 'model.pkl'
            self._save_model(data, labels, model_path)

            self.result = PipelineResult(
                success=True,
                accuracy=result.accuracy,
                std=result.std,
                cv_scores=result.cv_scores,
                steps_completed=self._steps.copy()
            )
            self._steps.append('decode')
            self.logger.info(f"Decoding done: accuracy={result.accuracy:.3f}")
            return self
        except Exception as e:
            self.logger.error(f"Decode failed: {e}")
            raise

    def _save_model(self, data: 'np.ndarray', labels: 'np.ndarray',
                    model_path: Path):
        """Train final decoder on all data and save."""
        try:
            from bci.decoder import create_decoder
            kwargs: dict = {}
            if self.config.decode.method in ('ssvep', 'fbcca'):
                kwargs['fs'] = self.epochs.info.get('sfreq', 256)
                kwargs['target_freqs'] = sorted(set(labels))
            decoder = create_decoder(self.config.decode.method, **kwargs)
            decoder.fit(data, labels)
            decoder.save(model_path)
            self.logger.info(f"Model saved to {model_path}")
        except Exception as e:
            self.logger.warning(f"Model save skipped: {e}")

    def run(self, filepath: Path | str,
            events: Optional['np.ndarray'] = None,
            event_id: Optional[Dict[str, int]] = None) -> PipelineResult:
        """
        Run complete pipeline

        Args:
            filepath: Path to EEG file
            events: Events array (optional)
            event_id: Event ID dict (optional)

        Returns:
            PipelineResult
        """
        self.logger.info("=" * 50)
        self.logger.info("Starting BCI Pipeline")
        self.logger.info("=" * 50)

        self._steps = []

        try:
            self.load(filepath)
            self.preprocess()
            self.create_epochs(events, event_id)
            self.decode()

            self.logger.info("Pipeline completed successfully")
            if self.result is None:
                raise RuntimeError("No result after decode step")
            return self.result

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return PipelineResult(
                success=False,
                errors=[str(e)],
                steps_completed=self._steps
            )

    def save_results(self, output_dir: Optional[Path] = None) -> List[Path]:
        """Save pipeline results"""
        if output_dir is None:
            output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = []

        if self.epochs is not None:
            epochs_path = output_dir / 'epochs.fif'
            self.epochs.save(epochs_path, overwrite=True)
            saved.append(epochs_path)

        import json
        results_path = output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'accuracy': self.result.accuracy if self.result else None,
                'std': self.result.std if self.result else None,
                'steps': self._steps
            }, f, indent=2)
        saved.append(results_path)

        self.logger.info(f"Saved {len(saved)} files to {output_dir}")
        return saved


def run_pipeline(config: 'PipelineConfig', filepath: Path | str) -> PipelineResult:
    """Convenience function to run pipeline"""
    pipeline = BCIPipeline(config)
    return pipeline.run(filepath)