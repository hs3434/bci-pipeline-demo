# BCI Pipeline — EEG Signal Processing & Decoding Toolkit

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)]()
[![MNE](https://img.shields.io/badge/MNE-Python-425AE8.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg)]()
[![PyQt6](https://img.shields.io/badge/PyQt6-41CD52.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

End-to-end EEG analysis pipeline with **offline batch** and **real-time streaming** dual-mode GUI. Built from scratch — from raw signal loading through preprocessing, epoch extraction, and 6 decoder backends.

```
bci/
├── loader/          Data loading (EDF/FIF/EEGLAB/BrainVision)
├── preprocessor/    Signal preprocessing (filtering/ICA/referencing)
├── processor/       Dual engine (offline zero-phase / online causal)
├── epocher/         Event detection & epoch extraction
├── decoder/         6 decoders: LDA, SSVEP, FBCCA, CSP, CNN, Transformer (GPT/BERT)
├── pipeline/        Pipeline orchestrator (fluent builder chain)
├── streaming/       Sliding window for real-time inference
├── source/          Data source abstraction (File/Stream/Session)
├── gui/             PyQt6 dual-tab GUI (offline analysis + streaming)
├── tests/           ~1400 LOC pytest suite
└── main.py          CLI entry point
```

## Quick Start

```bash
# Install
pip install .

# CLI mode — run a full pipeline
bci sample_data.edf --method lda

# GUI mode — explore, filter, decode interactively
bci --gui
```

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  GUI (PyQt6)                      │
│  ┌────────────┐     ┌────────────────────────┐   │
│  │  BatchTab   │     │  StreamTab             │   │
│  │  离线分析    │     │  实时流式               │   │
│  └─────┬──────┘     └─────┬──────────────────┘   │
│  ┌─────┴──────┐     ┌─────┴──────────────┐       │
│  │ BatchWorker │     │ StreamWorker        │       │
│  │ (QObject)   │     │ (QObject+QTimer)    │       │
│  └─────┬──────┘     └─────┬──────────────┘       │
└────────┼──────────────────┼───────────────────────┘
         │                  │
         ▼                  ▼
   ┌───────────┐    ┌──────────────────┐
   │ BCIPipeline│    │ SlidingWindow    │
   │ 流式链式调用│    │ + OnlineProcessor│
   │            │    │ + StreamSource   │
   └─────┬─────┘    + Decoder          │
         │          └────┬─────────────┘
    ┌────┼────┬─────┬────┼─────┬────────┐
    ▼    ▼    ▼     ▼    ▼     ▼        ▼
  Loader Preproc Epocher Source Processor Decoder
```

Two processing paths sharing the same core modules:

| | Batch | Streaming |
|---|---|---|
| Data access | Full file, random access | Chunk-by-chunk, simulated live |
| Filtering | `filtfilt` (zero-phase, offline) | `lfilter` (causal, state maintained) |
| Normalization | Global z-score | EMA (exponential moving average) |
| Decoding | Full dataset → CV → results | Sliding window → model → prediction |

## Decoders

All decoders share a common `fit/predict/predict_proba/save/load` interface and auto-register via `@register()` decorator.

| Method | Type | Notes |
|--------|------|-------|
| `lda` | sklearn | StandardScaler + PCA + LDA pipeline |
| `ssvep` | CCA | Single-band CCA with harmonic templates |
| `fbcca` | CCA | Filter-bank CCA, weighted sub-band aggregation |
| `csp` | CSP + LDA | MNE CSP → log-var → StandardScaler → LDA |
| `cnn` | PyTorch | 2D Conv → BN → ReLU → Conv → BN → FC |
| `transformer` | PyTorch (GPT) | Causal attention + RoPE + last-token head |
| `transformer_bert` | PyTorch (BERT) | Bidirectional + [CLS] head |

### GPT vs BERT ablation (MNE Sample ERP, 288 trials)

| Model | Accuracy | Δ vs CNN |
|-------|----------|----------|
| Transformer baseline | 0.806 | -13.8pp |
| + pyramid augmentation | 0.844 | -10.0pp |
| → bidirectional + CLS | 0.865 | -7.9pp |
| → optimal window L=85 | 0.878 | -6.6pp |
| CNN baseline | 0.944 | — |

## GUI

**Batch tab**: 4-step pipeline with visual progress indicators, background execution via `BaseWorker.moveToThread` pattern.

**Stream tab**: Real-time playback with speed control (0.25×–100×), live waveform/spectrum/topomap visualization, sliding-window prediction with loaded models.

## Testing

```bash
pip install ".[dev]"
pytest
```

~1400 LOC test suite covering decoders, processors, streaming, pipeline, worker threads, and GUI components.

## License

MIT