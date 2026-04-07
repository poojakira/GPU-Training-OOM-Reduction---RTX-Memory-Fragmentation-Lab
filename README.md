# RTX-OOM-Guard (apex-aegis): Predictive GPU Memory Defragmenter

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)](https://pytorch.org) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE) [![CI](https://github.com/poojakira/RTX-OOM-Guard/actions/workflows/ci.yml/badge.svg)](https://github.com/poojakira/RTX-OOM-Guard/actions)

---

## One-Line Description

A transformer-driven proactive CUDA memory optimizer for PyTorch that predicts and prevents GPU out-of-memory (OOM) crashes by actively defragmenting VRAM during training.

---

## Problem Statement

PyTorch's CachingAllocator leaves memory fragmented during long training runs, causing OOM crashes even when total free VRAM appears sufficient. Existing solutions (gradient checkpointing, reduced batch size) sacrifice throughput. RTX-OOM-Guard solves this by predicting fragmentation before it causes a crash and proactively compacting live tensors into contiguous blocks.

---

## Key Features

- **`GPUMemoryDefragmenter`** — Actively repacks scattered live PyTorch tensors into a single contiguous VRAM allocation; physically replaces `.data` pointers so autograd and optimizer states continue flawlessly
- **Triton Compaction Kernels** — High-bandwidth custom Triton copy kernel (`compaction_kernels.py`) used when available; falls back to pure PyTorch
- **`FragPredictor`** — ML model that predicts fragmentation score from allocation event stream
- **`DefragMonitor`** — Background daemon thread running at 50ms intervals; auto-triggers compaction when predicted fragmentation exceeds configurable threshold (default 0.7)
- **`OOMRiskModel`** — Risk model that scores OOM probability from memory traces
- **`AllocationCollector`** — Hooks into PyTorch allocator to log per-step allocation/free events
- **`AllocatorLogger`** — Structured logging of VRAM allocation traces to Parquet files
- **`auto_instrument`** — Zero-code-change wrapper: `model, optimizer = auto_instrument(model, optimizer)`
- **`DDPSyncManager`** — Distributed Data Parallel (DDP) integration for multi-GPU training
- **FastAPI REST API** — `src/apex_aegis/api/main.py` exposes defrag endpoints for remote monitoring
- **React Dashboard** — Vite+React frontend (`dashboard/`) with 13 panels: VRAM map, fragmentation chart, DDP choreography, Triton trace, latency graphs, KPI grid, and more
- **Extensive Benchmarks** — BERT-base, BERT-large, GPT-2, GPT-2-medium, ResNet-50, ResNet-101, EfficientNet-B4, ViT-Large across multiple batch sizes and VRAM configs (6GB, 8GB, 12GB)
- **KV Cache Manager** — LLM-specific KV cache memory optimization

---

## Tech Stack

| Layer | Technology |
|---|---|
| Core Library | Python 3.10+, PyTorch 2.0+ |
| Compaction Kernels | Triton (optional), PyTorch fallback |
| ML Predictor | Custom FragPredictor model |
| API | FastAPI |
| Dashboard | React 18, Vite, Recharts |
| Data Format | Parquet (PyArrow), JSON |
| Infrastructure | Docker, GitHub Actions CI, pytest |
| Config | YAML (`configs/config.yaml`) |

---

## Architecture

```
src/apex_aegis/
├── __init__.py                    # Public API: auto_instrument, DefragMonitor, etc.
├── defrag_engine/
│   ├── defragmenter.py            # GPUMemoryDefragmenter: tensor compaction
│   ├── compactor.py               # Compactor: orchestrates defrag runs
│   ├── policy.py                  # MitigationPolicy: when/how to defrag
│   └── benchmark_triton.py        # Triton vs PyTorch copy benchmark
├── defrag/compaction_kernels.py   # Custom Triton copy kernel
├── scheduler/
│   ├── monitor.py                 # DefragMonitor: background daemon
│   ├── risk_model.py              # OOMRiskModel: fragmentation risk scoring
│   └── dataset.py                 # Dataset loader for trace parquet files
├── predictor/model.py             # FragPredictor: ML fragmentation predictor
├── profiler/
│   ├── collector.py               # AllocationCollector: allocator event hooks
│   └── allocator_logger.py        # AllocatorLogger: structured Parquet logging
├── trainer/
│   ├── auto_instrument.py         # auto_instrument(): zero-code-change wrapper
│   ├── callback.py                # DefragCallback: training loop callback
│   ├── ddp.py                     # DDPSyncManager: multi-GPU support
│   ├── trainer.py                 # Custom trainer with defrag integration
│   └── training_hook.py           # TrainingHook: per-step instrumentation
├── llm_system/kv_cache_manager.py # LLM KV cache optimization
├── optimization/quantization.py   # INT8/FP16 quantization helpers
├── api/main.py                    # FastAPI REST API
└── utils.py                       # DefragConfig, parse_memory_snapshot, logging

dashboard/                         # React + Vite frontend (13 monitoring panels)
benchmarks/                        # OOM benchmarks, model fragmentation tests
scripts/                           # Data collection, DDP training, stress tests
data/traces/senior_v1/             # 100+ Parquet trace files (BERT/GPT2/ResNet/ViT)
```

---

## Installation

### Python Package

```bash
git clone https://github.com/poojakira/RTX-OOM-Guard.git
cd RTX-OOM-Guard
pip install -e .
```

### Docker

```bash
docker build -t rtx-oom-guard .
docker run --gpus all rtx-oom-guard
```

### Dashboard (React)

```bash
cd dashboard
npm install
npm run dev
# Dashboard at http://localhost:5173
```

---

## Usage

### Zero-Code-Change Integration

```python
from apex_aegis import auto_instrument

model, optimizer = auto_instrument(model, optimizer)

# ... your standard training loop, no other changes needed
```

### Manual Monitor Control

```python
from apex_aegis import DefragMonitor

monitor = DefragMonitor(threshold=0.7)
monitor.start()

for batch in dataloader:
    monitor.record_alloc(tensor.numel() * tensor.element_size())
    output = model(batch)
    loss.backward()
    optimizer.step()

monitor.stop()
print(monitor.stats())
```

### Run Benchmarks

```bash
python run_benchmark.py
# or
python benchmarks/unified_benchmark.py
```

### Collect Real Traces

```bash
python scripts/collect_real_traces.py
```

---

## Configuration

Edit `configs/config.yaml`:

```yaml
defrag:
  threshold: 0.7          # Fragmentation score to trigger compaction
  interval_ms: 50         # Monitor polling interval
  cooldown_steps: 10      # Steps between compaction runs
  use_triton: true        # Use Triton kernels if available
logging:
  results_dir: results    # Output directory for traces and logs
```

---

## Benchmark Results

All benchmarks run across BERT-base, BERT-large, GPT-2, GPT-2-medium, ResNet-50, ResNet-101, EfficientNet-B4, ViT-Large with batch sizes 2–16 and VRAM configs 6GB/8GB/12GB (simulated).

| Metric | Baseline | With apex-aegis |
|---|---|---|
| OOM crashes (100-step run) | 23 | 0 |
| Peak VRAM utilization | 94% | 87% |
| Iteration time overhead | — | < 2% |
| Fragmentation ratio | 0.61 avg | 0.18 avg |

See `RESULTS.md`, `benchmarks.md`, and `results/` for full data.

---

## Tests

```bash
pytest tests/ -v
# 50+ test files covering all modules
```

Key test files:
- `tests/test_100pct_coverage.py` — Comprehensive 100% coverage suite
- `tests/test_defragmenter.py` — GPUMemoryDefragmenter unit tests
- `tests/test_monitor.py` — DefragMonitor daemon tests
- `tests/test_api.py` — FastAPI endpoint tests
- `tests/test_simulator.py` — OOM simulation tests

---

## Project Structure

```
.
├── src/apex_aegis/          # Core Python library
├── dashboard/               # React monitoring dashboard
├── benchmarks/              # Benchmark scripts
├── scripts/                 # Data collection and training scripts
├── data/traces/             # Parquet memory traces
├── results/                 # Benchmark results and plots
├── tests/                   # Test suite (50+ files)
├── configs/config.yaml      # Configuration
├── Dockerfile
├── Makefile
├── pyproject.toml
└── run_benchmark.py
```

---

## Roadmap

- [ ] Automatic Triton kernel tuning per GPU model
- [ ] Integration with HuggingFace Trainer as a callback
- [ ] Support for FSDP (Fully Sharded Data Parallel)
- [ ] Live memory visualization in React dashboard via WebSocket
- [ ] PyPI package release

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Run `pytest tests/` before submitting a PR. Code style enforced via `.editorconfig`.

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Author

Built by [Pooja Kiran](https://github.com/poojakira) — Master's in Information Technology, Arizona State University.
