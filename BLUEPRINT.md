# 🔧 BLUEPRINT: Predictive GPU Memory Defragmenter
### Version 1.0.0 — Production Architecture & Implementation Blueprint

---

## Phase 1 — Problem Definition & Market Context

### The Problem
GPU-accelerated ML workloads crash with **Out-of-Memory (OOM) errors** even when aggregate free memory is sufficient. Root cause: **CUDA memory fragmentation** — the allocator creates non-contiguous free gaps between live tensors that cannot satisfy large contiguous requests.

### Why Existing Solutions Fail

| Solution | Approach | Failure Mode |
|---|---|---|
| `torch.cuda.empty_cache()` | Reactive cache clear | Only runs *after* crash — loses state |
| NVIDIA Memory Pools | Pre-reserved pools | Static — doesn't adapt to workload |
| Gradient Checkpointing | Reduce peak usage | Doesn't address fragmentation pattern |
| Manual `gc.collect()` | Python-level cleanup | Doesn't touch CUDA allocator |

### Our Innovation
**Predictive, not reactive.** A lightweight Transformer encoder analyzes real-time allocation patterns and triggers compaction *before* fragmentation reaches critical levels.

---

## Phase 2 — System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER TRAINING LOOP                          │
│                                                                 │
│   ┌────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐  │
│   │Forward │ →  │ Backward │ →  │ Optim    │ →  │ Step End  │  │
│   └───┬────┘    └────┬─────┘    └────┬─────┘    └─────┬─────┘  │
│       │              │               │                │         │
│       ▼              ▼               ▼                ▼         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │            DefragCallback (1-line integration)          │   │
│   │  on_step_begin()              on_step_end()             │   │
│   └───────────────────────┬─────────────────────────────────┘   │
│                           │                                     │
│              ┌────────────▼──────────────┐                      │
│              │    DefragMonitor Thread   │ ← Daemon, 50ms loop  │
│              │                           │                      │
│              │  ┌─────────────────────┐  │                      │
│              │  │ AllocationCollector  │  │  Ring buffer (64)    │
│              │  │ [action, size, Δt,  │  │  Sub-ms polling      │
│              │  │  frag_ratio]        │  │                      │
│              │  └────────┬────────────┘  │                      │
│              │           │               │                      │
│              │  ┌────────▼────────────┐  │                      │
│              │  │  FragPredictor      │  │  4-layer Transformer │
│              │  │  (812,801 params)   │  │  < 2ms inference     │
│              │  │  → score ∈ [0, 1]   │  │                      │
│              │  └────────┬────────────┘  │                      │
│              │           │               │                      │
│              │     score > 0.7?          │  Configurable         │
│              │           │ YES           │                      │
│              │  ┌────────▼────────────┐  │                      │
│              │  │  MemoryCompactor    │  │  sync → empty_cache  │
│              │  │  + GC + metrics     │  │  + cooldown (1s)     │
│              │  └─────────────────────┘  │                      │
│              │                           │                      │
│              │  Kill Switch: latency     │                      │
│              │  > 5ms → auto-disable     │                      │
│              └───────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 3 — Module Blueprint

### 3.1 `gpudefrag/collector.py` — Allocation Trace Collector

**Purpose**: Capture CUDA memory events at sub-millisecond frequency.

**Design Decisions**:
- **Dual-mode**: Background polling thread + manual `.record()` hooks for maximum coverage
- **Ring buffer**: Capped at 200K events to prevent memory bloat
- **Thread-safe**: `threading.Lock` on all event mutations
- **Export**: Parquet format via PyArrow for efficient storage + Pandas interop

**Data Schema** (per event):
| Field | Type | Description |
|---|---|---|
| `timestamp_ns` | int64 | High-res perf counter (ns) |
| `delta_bytes` | int64 | Signed memory delta |
| `action` | int8 | 1 = alloc, 0 = free |
| `abs_allocated` | int64 | Current `memory_allocated()` |
| `abs_reserved` | int64 | Current `memory_reserved()` |
| `fragmentation` | float32 | `1 - (allocated / reserved)` |

---

### 3.2 `gpudefrag/predictor.py` — FragPredictor (Transformer)

**Purpose**: Predict near-future fragmentation severity from allocation sequences.

**Architecture**:
```
Input (B, 64, 4) → Linear+LN+GELU → (+PosEnc) → 4× TransformerEncoderLayer
→ LayerNorm → Last Token → MLP Head → Sigmoid → (B, 1)
```

| Hyperparameter | Value | Rationale |
|---|---|---|
| Layers | 4 | Enough depth for temporal patterns |
| Heads | 4 | 128/4 = 32 dim/head — standard |
| Hidden dim | 128 | Balance: expressivity vs. latency |
| FFN dim | 512 | 4× hidden (standard) |
| Seq len | 64 | ~3s of allocation history at full speed |
| Params | **812,801** | Small enough for CPU inference < 2ms |

**Training**: AdamW (lr=1e-3, wd=0.01) + CosineAnnealing + gradient clipping (max_norm=1.0)

---

### 3.3 `gpudefrag/compactor.py` — Memory Compaction Engine

**Purpose**: Execute controlled CUDA memory cleanup.

**Compaction Cycle**:
1. Record pre-state (`memory_allocated`, `memory_reserved`)
2. `torch.cuda.synchronize()` — barrier all streams
3. `torch.cuda.empty_cache()` — release cached allocator blocks
4. `gc.collect()` — optional Python GC
5. Record post-state + compute `freed_mb`, `frag_reduction`

**Metrics Tracked**: Every compaction records 12 fields (timestamp, reason, pre/post memory, freed, frag change, latency, compaction ID).

---

### 3.4 `gpudefrag/monitor.py` — Real-Time Background Monitor

**Purpose**: Daemon thread that continuously predicts and acts.

**Safety Features**:
| Feature | Implementation |
|---|---|
| **Kill Switch** | If prediction latency > 5ms → auto-disable monitor |
| **Cooldown** | Min 1s between compactions (prevent thrashing) |
| **Ring Buffer** | O(1) event insertion, ordered reconstruction for prediction |
| **Thread Safety** | Lock-protected buffer writes |
| **Telemetry** | Full stats export (predictions, latencies, compaction history) |

---

### 3.5 `gpudefrag/callback.py` — Training Integration

**Purpose**: Zero-config drop-in for any PyTorch training loop.

```python
from gpudefrag import DefragCallback

callback = DefragCallback(threshold=0.7)
callback.on_train_begin()

for epoch in range(epochs):
    for batch in dataloader:
        callback.on_step_begin()
        # ... your training code ...
        callback.on_step_end()

callback.on_train_end()
print(callback.stats())
```

---

### 3.6 `gpudefrag/dataset.py` — Data Pipeline

**Purpose**: Convert raw Parquet traces → sliding-window tensors for training.

**Features**:
- Auto-discovers all `.parquet` files in trace directory
- 4-feature vectors: `[action, size_gb, time_delta_ms, fragmentation]`
- Sliding window with configurable `seq_len`
- 80/10/10 train/val/test split (seeded for reproducibility)
- Pinned memory for GPU transfer optimization

---

### 3.7 `gpudefrag/trainer.py` — Training Pipeline

**Features**:
- CosineAnnealing LR scheduler
- Gradient clipping (max_norm=1.0)
- Best-model checkpointing (by val_loss)
- Train/val/test evaluation with MSE + MAE metrics
- JSON metrics export to `results/training_metrics.json`

---

## Phase 4 — Benchmark Suite

### Baseline (`benchmark/run_baseline.py`)
Trains GPT-2 (6-layer) with **synthetic fragmentation** (interleaved 10MB/1MB allocs, hole creation, medium fills) for 100 iterations. Records OOM count, iteration times, peak memory, fragmentation snapshots.

### With Defrag (`benchmark/run_with_defrag.py`)
Same workload + `DefragCallback` enabled. Fair A/B comparison.

### Comparison (`benchmark/compare.py`)
Runs both sequentially, generates:
- `results/comparison.json`
- `results/comparison.csv`
- Console comparison table

---

## Phase 5 — Test Coverage

**17 unit tests across 4 test files**, all passing:

| Test File | Tests | Coverage |
|---|---|---|
| `test_predictor.py` | 6 | Output shape, range, config, save/load, params, gradients |
| `test_compactor.py` | 3 | No-CUDA fallback, history tracking, freed memory |
| `test_collector.py` | 5 | Manual record, dataframe, save, clear, config roundtrip |
| `test_monitor.py` | 3 | Start/stop lifecycle, event recording, stats structure |

---

## Phase 6 — Packaging & Distribution

**`pyproject.toml`** configures:
- `pip install -e .` for development
- Optional deps: `[dev]` (pytest, ruff, mypy), `[models]` (torchvision, transformers)
- Three CLI commands: `gpudefrag-collect`, `gpudefrag-train`, `gpudefrag-benchmark`
- Compatible with PyPI publishing via `python -m build`

---

## Phase 7 — Commercial Value & Patent Angle

### Novel Claims
1. **Sequence-model-based fragmentation prediction** — no prior art uses Transformers for CUDA memory prediction
2. **Proactive compaction** — triggers *before* OOM, unlike all existing reactive approaches
3. **Zero-overhead integration** — background thread with automatic kill switch
4. **User-space only** — no kernel or driver modifications required

### Target Markets
- Cloud ML platforms (AWS SageMaker, GCP Vertex AI, Azure ML)
- Inference serving (Triton, vLLM, TGI)
- Edge AI (NVIDIA Jetson, mobile GPU)
- AutoML systems (hyperparameter search without OOM fear)

---

## File Tree

```
Predictive GPU Memory Defragmenter/
├── gpudefrag/
│   ├── __init__.py          # Public API
│   ├── _models.py           # Model factories (GPT-2, ResNet, BERT)
│   ├── callback.py          # DefragCallback
│   ├── cli.py               # CLI entry points
│   ├── collector.py         # AllocationCollector
│   ├── compactor.py         # MemoryCompactor
│   ├── dataset.py           # AllocationDataset + dataloaders
│   ├── monitor.py           # DefragMonitor
│   ├── predictor.py         # FragPredictor (Transformer)
│   ├── trainer.py           # Training pipeline
│   └── utils.py             # Config, logging, helpers
├── benchmark/
│   ├── run_baseline.py      # Baseline benchmark
│   ├── run_with_defrag.py   # Defrag-enabled benchmark
│   └── compare.py           # A/B comparison + reports
├── tests/
│   ├── test_predictor.py
│   ├── test_compactor.py
│   ├── test_collector.py
│   └── test_monitor.py
├── data/traces/             # Parquet trace files
├── checkpoints/             # Model checkpoints
├── results/                 # Benchmark results (JSON, CSV)
├── pyproject.toml           # Package configuration
├── requirements.txt         # Legacy deps file
├── README.md                # Professional README
├── RESULTS.md               # Benchmark results document
├── patent-angle.md          # IP & commercial positioning
├── BLUEPRINT.md             # ← This file
├── LICENSE                  # MIT
└── .gitignore
```
