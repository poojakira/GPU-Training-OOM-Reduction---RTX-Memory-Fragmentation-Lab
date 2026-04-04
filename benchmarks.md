# 📊 Apex-Aegis Benchmarking Methodology

This document details the hardware configurations, dataset characteristics, and baseline metrics used to validate the **Apex-Aegis** GPU memory management engine.

## 🖥️ Benchmarking Hardware

The current performance profile (v2.0.0) was established on a mid-range RTX workstation.

| Component | specification |
|---|---|
| **GPU** | NVIDIA GeForce RTX 4060 (8 GB GDDR6) |
| **Driver Version** | 560.94 (CUDA 12.1) |
| **CPU** | Intel Core i7-13700K (16 Cores, 5.4 GHz) |
| **VRAM** | 8192 MB Physical Capacity |
| **OS** | Windows 11 (23H2) / Ubuntu 22.04 LTS |

---

## 📂 Data & Workloads

### 1. Transformer Autoregressive Traces
- **Model**: GPT-2 (Small / 6-Layer / 128 Hidden Dim)
- **Trace Length**: 10,000+ allocation events captured via `torch.cuda.memory_snapshot`
- **Pressure Pattern**: Synthetic fragmentation "Punch" (Small periodic allocations that prevent large contiguous block assembly).

### 2. Physical Memory Topology Simulation
- **VRAM Scan**: Periodic physical address verification to ensure 0% data corruption during in-place repacking.
- **Latency Sweep**: Triton kernel profiling across memory block sizes (256KB to 1GB).

---

## 📉 Baseline vs. Apex-Aegis

The core performance gain is derived from **preventive compaction** before the CUDA allocator triggers a reactive `OutOfMemoryError`.

| Metric | Baseline (Stock PyTorch) | Apex-Aegis v2.0.0 | Impact |
|---|---|---|---|
| **OOM Exceptions** | 22 (per 100 iters) | **0** | ✅ **100% Reliability** |
| **Fragmentation High-Water Mark** | 91.4% | **28.7%** | 📉 **-68.6%** |
| **Peak VRAM Demand** | 7,840.4 MB | **6,617.1 MB** | 📈 **+15.6% Space** |
| **Avg Iteration Latency** | 1.94 s | **1.83 s** | ⚡ **-5.7% Speedup** |

> [!NOTE]
> The performance speedup in iteration latency is primarily due to the reduction in memory allocator search time and the elimination of expensive internal `free_memory` sweeps triggered by near-OOM conditions.

---

## 🛠️ Reproduction Steps

To reproduce these metrics on your local hardware:

1.  **Initialize Environment**:
    ```ps1
    pip install -e "."
    ```
2.  **Execute Benchmarks**:
    ```ps1
    python run_benchmark.py --steps 200
    ```
3.  **Audit Results**:
    Inspect the generated `results/benchmark_results.json` and `results/benchmark_summary.csv`.

---

## 🔬 SLOs (Service Level Objectives)

In production environments, Apex-Aegis targets the following stability metrics:

- **SLO-1 (Latency)**: Compaction overhead MUST NOT exceed 15ms per invocation on high-bandwidth memory (HBM2/3).
- **SLO-2 (Stability)**: 100% Prevention of Fragmentation-induced OOMs for any workload where `physical_free_memory > 2 * peak_allocation_size`.
- **SLO-3 (Security)**: Bit-accurate data preservation during physical block migrations.
