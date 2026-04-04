"""
benchmarks/simulate_oom_behavior.py
====================================
High-fidelity simulation of GPU fragmentation and OOM behavior.
Models different hardware tiers (RTX 3060, 4070, 4090) and calculates
the impact of the Predictive Defragmenter workflow.
"""

import os
import sys
import json
import argparse
import statistics
from datetime import datetime
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.workload_simulator import GPUWorkload, TransformerSpec, CNNSpec

# ── Hardware Profiles ────────────────────────────────────────────────────────

HARDWARE_PROFILES = {
    "RTX_3060_12GB": 12288,
    "RTX_3060_TIGHT": 6000, # Tuned for GPT2-Medium Fragmentation demo
    "RTX_4070_12GB": 12288,
    "RTX_4080_16GB": 16384,
    "RTX_4090_24GB": 24576,
    "Enterprise_H100_80GB": 81920,
}

# ── Metrics Helper ───────────────────────────────────────────────────────────

def compute_metrics(events: List[Dict[str, Any]], total_vram_mb: float, overhead_ns: int) -> Dict[str, Any]:
    """Extract requested metrics from a simulation trace."""
    ooms = [e for e in events if e["oom"]]
    oom_hit = len(ooms) > 0
    oom_step = ooms[0]["step"] if oom_hit else -1
    
    utils = [e["utilization"] * 100 for e in events]
    frags = [e["fragmentation"] for e in events]
    
    # Time-series for "over time" tracking
    timeseries = []
    for e in events:
        timeseries.append({
            "step": e["step"],
            "frag_index": e["fragmentation"],
            "util_pct": e["utilization"] * 100,
            "abs_allocated": e["abs_allocated"]
        })
    
    return {
        "oom": oom_hit,
        "oom_step": oom_step,
        "avg_util_pct": statistics.mean(utils) if utils else 0.0,
        "max_util_pct": max(utils) if utils else 0.0,
        "avg_frag_index": statistics.mean(frags) if frags else 0.0,
        "max_frag_index": max(frags) if frags else 0.0,
        "total_overhead_ms": overhead_ns / 1e6,
        "timeseries": timeseries
    }

# ── Simulation Runner ────────────────────────────────────────────────────────

def run_simulation_set(name: str, spec: Any, vram_mb: float, use_defrag: bool, n_trials: int = 20, steps: int = 100) -> Dict[str, Any]:
    """Run N trials for a specific configuration."""
    results = []
    strategy = "predictive" if use_defrag else None
    
    print(f"Running {name} ({'Defrag' if use_defrag else 'Baseline'}) | VRAM: {vram_mb}MB | Trials: {n_trials}")
    
    for i in range(n_trials):
        wl = GPUWorkload(
            spec, 
            vram_mb=vram_mb, 
            defrag_strategy=strategy,
            defrag_threshold=0.6,
            defrag_overhead_ms=15.0
        )
        events = wl.run(steps=steps, seed=i)
        metrics = compute_metrics(events, vram_mb, wl._total_defrag_overhead_ns)
        results.append(metrics)
        
    oom_count = sum(1 for r in results if r["oom"])
    runs_to_oom = [r["oom_step"] for r in results if r["oom"]]
    avg_runs_to_oom = statistics.mean(runs_to_oom) if runs_to_oom else steps
    
    return {
        "oom_rate_pct": oom_count / n_trials * 100,
        "oom_count": oom_count,
        "avg_runs_to_oom": avg_runs_to_oom,
        "avg_util_pct": statistics.mean(r["avg_util_pct"] for r in results),
        "avg_frag_index": statistics.mean(r["avg_frag_index"] for r in results),
        "avg_overhead_ms_per_step": statistics.mean(r["total_overhead_ms"] for r in results) / steps,
        "representative_timeseries": results[0]["timeseries"] if results else [],
        "raw_results": results
    }

# ── Stable Batch Size Search ─────────────────────────────────────────────────

def find_stable_batch_size(spec_factory: Any, vram_mb: float, use_defrag: bool, k: int = 5) -> int:
    """Find max batch size that passes K trials."""
    batch_size = 1
    max_stable = 0
    
    while batch_size < 128:
        spec = spec_factory(batch_size=batch_size)
        strategy = "predictive" if use_defrag else None
        
        failures = 0
        for i in range(k):
            wl = GPUWorkload(spec, vram_mb=vram_mb, defrag_strategy=strategy)
            events = wl.run(steps=50, seed=i*100)
            if any(e["oom"] for e in events):
                failures += 1
                break
        
        if failures == 0:
            max_stable = batch_size
            batch_size += 2 if batch_size >= 8 else 1
        else:
            break
            
    return max_stable

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--gpu", default="RTX_3060_12GB")
    ap.add_argument("--out", default="results/simulated_modeling.json")
    args = ap.parse_args()

    vram_mb = HARDWARE_PROFILES.get(args.gpu, 12288)
    print(f"Targeting Profile: {args.gpu} ({vram_mb} MB)")

    # 1. Model Specific Workload (e.g. GPT-2 Medium)
    spec_factory = lambda batch_size=4: TransformerSpec.gpt2_medium(batch_size=batch_size)
    
    baseline = run_simulation_set("Baseline", spec_factory(), vram_mb, False, args.trials, args.steps)
    defrag   = run_simulation_set("Workflow", spec_factory(), vram_mb, True,  args.trials, args.steps)
    
    # 2. Batch Size Stability
    print("\nCalculating Max Stable Batch Size...")
    stable_b_base = find_stable_batch_size(spec_factory, vram_mb, False)
    stable_b_defrag = find_stable_batch_size(spec_factory, vram_mb, True)

    # 3. Compile Report
    report = {
        "timestamp": datetime.now().isoformat(),
        "hardware_profile": args.gpu,
        "vram_mb": vram_mb,
        "metrics": {
            "oom_reduction_pct": (baseline["oom_count"] - defrag["oom_count"]) / max(baseline["oom_count"], 1) * 100 if baseline["oom_rate_pct"] > 0 else 100.0,
            "util_gain_pp": defrag["avg_util_pct"] - baseline["avg_util_pct"],
            "overhead_pct": (defrag["avg_overhead_ms_per_step"] / 200.0) * 100 # assuming 200ms step
        },
        "baseline": {
            "oom_rate": baseline["oom_rate_pct"],
            "runs_to_oom": baseline["avg_runs_to_oom"],
            "stable_batch_size": stable_b_base,
            "avg_frag": baseline["avg_frag_index"]
        },
        "workflow": {
            "oom_rate": defrag["oom_rate_pct"],
            "runs_to_oom": defrag["avg_runs_to_oom"],
            "stable_batch_size": stable_b_defrag,
            "avg_frag": defrag["avg_frag_index"]
        }
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*50)
    print("SIMULATION RESULTS SUMMARY")
    print("="*50)
    print(f"OOM Rate: {baseline['oom_rate_pct']:.1f}% -> {defrag['oom_rate_pct']:.1f}%")
    print(f"Runs to OOM: {baseline['avg_runs_to_oom']:.1f} -> {defrag['avg_runs_to_oom']:.1f}")
    print(f"Stable Batch Size: {stable_b_base} -> {stable_b_defrag}")
    print(f"Avg Frag Index: {baseline['avg_frag_index']:.3f} -> {defrag['avg_frag_index']:.3f}")
    print("="*50)
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
