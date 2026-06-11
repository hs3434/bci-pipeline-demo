"""
Pipeline Module
================
BCI Pipeline Orchestrator with incremental state tracking.

When ``run()`` is called and only downstream parameters have changed,
previously-computed upstream steps are reused automatically.

    load → preprocess → create_epochs → decode
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

# Linear execution order — steps must match method names.
_STEP_ORDER = ('load', 'preprocess', 'create_epochs', 'decode')


@dataclass
class PipelineResult:
    """Pipeline execution result

    Attributes:
        success: Whether pipeline completed successfully
        accuracy: Classification accuracy (0-1), None if failed
        std: Standard deviation across CV folds
        steps_completed: List of pipeline steps that succeeded
        steps_skipped: Steps reused from a previous run
        errors: List of error messages if failed
        output_files: Paths to saved output files
    """
    success: bool
    accuracy: Optional[float] = None
    std: Optional[float] = None
    cv_scores: List[float] = field(default_factory=list)
    steps_completed: List[str] = field(default_factory=list)
    steps_skipped: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    output_files: List[Path] = field(default_factory=list)


@dataclass(frozen=True)
class _StepState:
    """Immutable snapshot of a completed step's parameters."""
    params: tuple


class BCIPipeline:
    """BCI Signal Processing Pipeline with incremental re-run.

    Orchestrates: Load → Preprocess → Epoch → Decode → Report.

    Examples:
        # Full run
        >>> config = create_default_config()
        >>> pipeline = BCIPipeline(config)
        >>> pipeline.load_raw(raw)
        >>> result = pipeline.run()
        >>> print(f"Accuracy: {result.accuracy:.3f}")

        # Change params, re-run — only affected steps execute
        >>> pipeline.config.epoch.tmin = -0.5
        >>> result = pipeline.run()
        >>> print(f"Skipped: {result.steps_skipped}")

        # Step-by-step (bypasses state tracking)
        >>> pipeline = BCIPipeline(config)
        >>> pipeline.load_raw(raw).preprocess().create_epochs().decode()
    """

    def __init__(self, config: 'PipelineConfig'):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.raw: Optional['mne.io.Raw'] = None
        self.epochs: Optional['mne.Epochs'] = None
        self.events: Optional['np.ndarray'] = None
        self.result: Optional[PipelineResult] = None

        self._raw_original: Optional['mne.io.Raw'] = None
        self._steps: List[str] = []
        self._states: Dict[str, _StepState] = {}

    # ------------------------------------------------------------------
    # Step methods — each returns self for fluent chaining
    # ------------------------------------------------------------------

    def load_raw(self, raw) -> 'BCIPipeline':
        """Load from an already-loaded MNE Raw object."""
        self._raw_original = raw
        self.raw = raw
        self._steps.append('load')
        self.logger.info(f"Loaded: {len(self.raw.ch_names)} channels")
        return self

    def preprocess(self) -> 'BCIPipeline':
        """Preprocess data (always starts from original unfiltered raw)."""
        from bci.preprocessor import preprocess

        self.logger.info("Preprocessing")
        try:
            if self._raw_original is None:
                raise RuntimeError("No raw data loaded, call load() first")
            self.raw = self._raw_original.copy()
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

            self._steps.append('decode')
            self.result = PipelineResult(
                success=True,
                accuracy=result.accuracy,
                std=result.std,
                cv_scores=result.cv_scores,
                steps_completed=self._steps.copy()
            )
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

    # ------------------------------------------------------------------
    # Parameter snapshots
    # ------------------------------------------------------------------

    def _param_snapshot(self, step: str) -> tuple:
        """Build a hashable, comparable tuple of the current config for *step*."""
        if step == 'load':
            return ()
        elif step == 'preprocess':
            cfg = self.config.filter
            return (cfg.l_freq, cfg.h_freq, tuple(cfg.notch_freqs))
        elif step == 'create_epochs':
            cfg = self.config.epoch
            return (cfg.tmin, cfg.tmax, cfg.baseline,
                    tuple(sorted(cfg.reject_threshold.items())))
        elif step == 'decode':
            cfg = self.config.decode
            return (cfg.method, cfg.cv_folds)
        return ()

    # ------------------------------------------------------------------
    # Main entry: run() with incremental re-run logic
    # ------------------------------------------------------------------

    def run(self,
            events: Optional['np.ndarray'] = None,
            event_id: Optional[Dict[str, int]] = None) -> PipelineResult:
        """Run pipeline, reusing unchanged upstream steps.

        Data must already be loaded via load_raw() before calling run().

        Args:
            events: Events array (optional).
            event_id: Event ID dict (optional).

        Returns:
            PipelineResult
        """
        # Determine which step is the first one that needs re-running
        skipped: List[str] = []
        start_idx = self._find_invalid_from()

        for i in range(start_idx):
            skipped.append(_STEP_ORDER[i])

        # Reset steps tracking to only the still-valid prefix
        self._steps = list(_STEP_ORDER[:start_idx])

        self.logger.info("=" * 50)
        if skipped:
            self.logger.info("Starting BCI Pipeline "
                             "(reusing: %s)", ' → '.join(skipped))
        else:
            self.logger.info("Starting BCI Pipeline")
        self.logger.info("=" * 50)

        try:
            for idx in range(start_idx, len(_STEP_ORDER)):
                step = _STEP_ORDER[idx]
                if step == 'load':
                    if 'load' not in self._steps:
                        raise RuntimeError("No data loaded, call load_raw() first")
                elif step == 'preprocess':
                    self.preprocess()
                elif step == 'create_epochs':
                    self.create_epochs(events, event_id)
                elif step == 'decode':
                    self.decode()
                self._states[step] = _StepState(
                    params=self._param_snapshot(step))

            self.logger.info("Pipeline completed successfully")
            if self.result is None:
                raise RuntimeError("No result after decode step")
            self.result.steps_skipped = skipped
            self.result.steps_completed = list(self._steps)
            return self.result

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return PipelineResult(
                success=False,
                errors=[str(e)],
                steps_completed=self._steps,
                steps_skipped=skipped,
            )

    def _find_invalid_from(self) -> int:
        """Return the first step index whose parameters have changed.

        If ``load()`` was never called or the filepath changed, index 0
        is returned.  All downstream state is pruned when a change is
        detected.
        """
        current = {}
        for step in _STEP_ORDER:
            current[step] = self._param_snapshot(step)

        for i, step in enumerate(_STEP_ORDER):
            if step not in self._states:
                return i
            if self._states[step].params != current[step]:
                for s in _STEP_ORDER[i:]:
                    self._states.pop(s, None)
                return i
        return len(_STEP_ORDER)  # all valid

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

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
    from bci.source import FileSource
    pipeline = BCIPipeline(config)
    raw = FileSource.load(filepath)
    pipeline.load_raw(raw)
    return pipeline.run()
