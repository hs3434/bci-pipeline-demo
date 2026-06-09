"""
Widgets Module
===============
Reusable visualization components for BCI data.
"""
from .waveform import EEGWaveformWidget
from .spectrum import SpectrumWidget
from .topomap import TopomapWidget
from .result_panel import ResultPanel
from .info_panel import EEGInfoPanel
from .step_strip import StepStrip, StepStatus
from .main_page import MainPage
from .preprocess_page import PreprocessPage
from .epoch_page import EpochPage
from .decode_page import DecodePage

__all__ = ['EEGWaveformWidget', 'SpectrumWidget', 'TopomapWidget', 'ResultPanel',
           'EEGInfoPanel', 'StepStrip', 'StepStatus',
           'MainPage', 'PreprocessPage', 'EpochPage', 'DecodePage']
