"""
BCI Pipeline Package
=====================
BCI Signal Processing Pipeline - Engineering and Integration

Modules:
    config   - Configuration management
    source   - Data loading (FileSource / StreamSource / EEGReader)
    preprocessor - Signal preprocessing (MNE-based)
    epocher  - Event/epoch extraction
    decoder  - BCI decoding (LDA, CSP, CNN, SSVEP, Transformer, ...)
    pipeline - Pipeline orchestrator
    processor - Online streaming signal processor
    streaming - Sliding window for real-time prediction
    gui      - Qt GUI (batch + streaming tabs)

Usage:
    from bci.config import PipelineConfig
    from bci.pipeline import BCIPipeline

    config = PipelineConfig()
    pipeline = BCIPipeline(config)
    result = pipeline.run('data.edf')
"""

from __future__ import annotations

__version__ = '1.0.0'
__author__ = 'BCI Learning Journey'

# Import main components for easy access
from bci.config import (
    PipelineConfig,
    FilterConfig,
    EpochConfig,
    DecodeConfig,
    create_default_config,
)

from bci.pipeline import BCIPipeline, PipelineResult, run_pipeline

__all__ = [
    'PipelineConfig',
    'FilterConfig',
    'EpochConfig',
    'DecodeConfig',
    'create_default_config',
    'BCIPipeline',
    'PipelineResult',
    'run_pipeline',
]