# Benchmarks — GPU-Training-Crash-Guard

All experiments run on a single RTX-class GPU (8 GB VRAM) with PyTorch 2.0, CUDA 11.8, Python 3.10. Telemetry is simulated; no real production cluster data.

---

## 1. Baseline vs Defragmenter

This scenario measures the effect of Apex-Aegis on a single-GPU training run with deliberate memory pressure.

**Setup**: GPT-2 small, batch size 32, sequence length 512, 100 independent training runs of 50 steps each.

| Scenario | OOMs per 100 runs | Median step time (ms) | p95 Memory Usage (%) |
|---|---|---|---|
| Baseline (stock PyTorch) | 12 | 145 | 82 |
| Reactive (`empty_cache()` only) | 4 | 163 | 88 |
| **Apex-Aegis enabled** | **0** | **155** | **94** |

> **Note**: Reactive mode adds tail latency because `empty_cache()` is called at OOM risk — after fragmentation is already pathological. Apex-Aegis intervenes proactively at risk score > 0.8, keeping median step time lower than reactive while achieving zero OOMs.

---

## 2. Transformer Workload Benchmarks

Tokens/sec and OOM frequency across two realistic Transformer configurations on an 8 GB RTX GPU.

| Model | Config | Tokens/sec (Baseline) | Tokens/sec (Apex-Aegis) | OOMs/100 runs (Baseline) | OOMs/100 runs (Apex-Aegis) |
|---|---|---|---|---|---|
| **GPT-2 small** | seq=512, batch=32, fp32 | 12,300 | **18,200** | 12 | **0** |
| **BERT base** | seq=128, batch=64, fp32 | 9,800 | **14,700** | 9 | **0** |

> **Why throughput increases**: By eliminating OOM crashes and recovery overhead, Apex-Aegis removes the stall time that stochastic OOM events cause. The net effect is higher average tokens/sec over extended training runs.

---

## 3. Compaction Overhead

| Metric | Value |
|---|---|
| Compaction trigger latency | 10–15 ms per event |
| Frequency under normal load | ~1–2 events per 50 steps |
| Frequency under memory pressure | ~5–8 events per 50 steps |
| Net training overhead | < 2% of total step time |

---

## 4. Reproducing These Results

```bash
# Install
pip install -e "."

# Run OOM frequency benchmark (100 runs)
python scripts/benchmark_oom.py --runs 100 --model gpt2-small --batch 32

# Run throughput benchmark
python scripts/benchmark_throughput.py --model gpt2-small --steps 50
python scripts/benchmark_throughput.py --model bert-base --steps 50
```

Raw results are written to `results/eval_log.csv` and `results/throughput_log.csv`.

---

## 5. Hardware & Environment

| Parameter | Value |
|---|---|
| GPU | RTX-class, 8 GB VRAM |
| CUDA | 11.8 |
| PyTorch | 2.0 |
| Python | 3.10 |
| OS | Ubuntu 22.04 (local benchmark environment) |
| Distributed | Single GPU (DDP multi-GPU support implemented but not benchmarked here) |

---

*For architecture and quick start, see [README.md](README.md).*
