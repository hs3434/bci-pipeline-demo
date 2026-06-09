"""
Configuration Management
========================
BCI Pipeline Configuration - YAML/dataclass/validation
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict, Any, TYPE_CHECKING
from pathlib import Path

import yaml
import logging

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """Filtering configuration"""
    l_freq: float = 0.5
    h_freq: float = 40.0
    notch_freqs: List[int] = field(default_factory=lambda: [50, 100])

    def validate(self) -> bool:
        """Validate filter parameters"""
        if self.l_freq < 0:
            raise ValueError(f"l_freq must be non-negative, got {self.l_freq}")
        if self.h_freq <= self.l_freq:
            raise ValueError(f"h_freq must be > l_freq, got {self.h_freq} <= {self.l_freq}")
        return True


@dataclass
class EpochConfig:
    """Epoch extraction configuration"""
    tmin: float = -0.2
    tmax: float = 0.5
    baseline: Tuple[Optional[float], Optional[float]] = (None, 0)
    reject_threshold: Dict[str, float] = field(default_factory=lambda: {
        'eeg': 300e-6
    })

    def validate(self) -> bool:
        if self.tmin >= self.tmax:
            raise ValueError(f"tmin must be < tmax, got {self.tmin} >= {self.tmax}")
        return True


@dataclass
class DecodeConfig:
    """Decoding configuration"""
    method: str = 'lda'
    cv_folds: int = 5

    def validate(self) -> bool:
        from bci.decoder import list_methods as list_decoder_methods
        valid_methods = list_decoder_methods()
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {self.method}")
        if self.cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2, got {self.cv_folds}")
        return True


@dataclass
class PipelineConfig:
    """Complete BCI Pipeline configuration"""
    filter: FilterConfig = field(default_factory=FilterConfig)
    epoch: EpochConfig = field(default_factory=EpochConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    output_dir: str = './output'
    subjects_dir: Optional[str] = None
    preload: bool = True

    def validate(self) -> bool:
        """Validate all sub-configurations"""
        self.filter.validate()
        self.epoch.validate()
        self.decode.validate()
        return True

    @classmethod
    def from_yaml(cls, path: Path | str) -> 'PipelineConfig':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: Path | str) -> None:
        """Save configuration to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


def create_default_config() -> PipelineConfig:
    """Create default configuration"""
    return PipelineConfig()