import React from 'react';
import { AlertTriangle, Zap, ShieldAlert, Layers, BarChart3, Clock, TrendingDown } from 'lucide-react';
import { FragmentationChart } from './FragmentationChart';

// =============================================================================
// Panel13_Benchmarking.jsx — High-Fidelity Stats Comparison Grid
// =============================================================================
// Visualizes the mathematical "Values" of the defragmentation system.
// =============================================================================

export function Panel13_Benchmarking({ data }) {
  if (!data) return null;

  const { baseline, workflow, metrics, hardware_profile } = data;

  // Transform time-series for the Chart
  const chartData = baseline.representative_timeseries.map((b, i) => {
    const w = workflow.representative_timeseries[i] || {};
    return {
      iteration: b.step,
      baselineFrag: b.frag_index * 100,
      defragFrag: w.frag_index * 100
    };
  });

  const Card = ({ title, value, unit, icon: Icon, color, subValue, subLabel }) => (
    <div className="bg-glass-card border border-glass-border p-6 rounded-lg relative overflow-hidden backdrop-blur-md">
       <div className={`absolute top-0 right-0 p-4 opacity-10 text-${color}`}>
          <Icon size={48} />
       </div>
       <div className="flex items-center gap-3 mb-4">
          <Icon size={16} className={`text-${color}`} />
          <span className="text-[10px] font-bold uppercase tracking-widest text-dim">{title}</span>
       </div>
       <div className="flex items-baseline gap-1">
          <span className="text-3xl font-bold tracking-tighter text-white">{value}</span>
          <span className="text-xs font-medium text-dim">{unit}</span>
       </div>
       <div className="mt-4 flex gap-4 border-t border-glass-border pt-4">
          <div className="flex flex-col">
             <span className="text-[8px] text-dim uppercase font-bold tracking-widest">Baseline</span>
             <span className="text-xs font-mono text-red-400">{baseline[subValue]}{subLabel}</span>
          </div>
          <div className="flex flex-col">
             <span className="text-[8px] text-dim uppercase font-bold tracking-widest">Efficiency</span>
             <span className="text-xs font-mono text-green-400">+{metrics.oom_reduction_pct}%</span>
          </div>
       </div>
    </div>
  );

  return (
    <div className="w-full h-full flex flex-col gap-6">
       {/* ── HEADER ── */}
       <div className="flex justify-between items-end">
          <div>
            <h3 className="text-sm font-bold text-white uppercase tracking-widest mb-1 flex items-center gap-2">
               <ShieldAlert size={14} className="text-amber" />
               Simulated Performance Modeling
            </h3>
            <p className="text-[10px] text-dim uppercase mono-metric tracking-tight">
               TARGET_HARDWARE: <span className="text-amber">{hardware_profile}</span> | MODEL: GPT2-MEDIUM (TIGHT)
            </p>
          </div>
          <div className="px-3 py-1 bg-green/10 border border-green/20 rounded text-[9px] text-green font-bold uppercase tracking-widest">
             CONFIDENCE: HIGH_FIDELITY
          </div>
       </div>

       {/* ── KPI GRID ── */}
       <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card 
            title="OOM Risk Frequency" 
            value={workflow.oom_rate.toFixed(1)} 
            unit="%" 
            icon={AlertTriangle} 
            color="amber"
            subValue="oom_rate"
            subLabel="%"
          />
          <Card 
            title="Max Stable Batch Size" 
            value={workflow.stable_batch_size} 
            unit="items" 
            icon={Layers} 
            color="green"
            subValue="stable_batch_size"
            subLabel=" B"
          />
          <Card 
            title="Fragmentation Index" 
            value={workflow.avg_frag.toFixed(3)} 
            unit="SCORE" 
            icon={BarChart3} 
            color="blue"
            subValue="avg_frag"
            subLabel=""
          />
          <Card 
            title="System Overhead" 
            value={(metrics.overhead_pct || 0).toFixed(2)} 
            unit="%" 
            icon={Zap} 
            color="purple"
            subValue="avg_overhead_ms_per_step"
            subLabel="ms"
          />
       </div>

       {/* ── COMPARISON TABLE ── */}
       <div className="flex-1 bg-glass-card border border-glass-border rounded-lg overflow-hidden flex flex-col backdrop-blur-md">
          <div className="bg-white/5 px-6 py-3 border-b border-glass-border flex justify-between items-center">
             <span className="text-[10px] font-bold uppercase tracking-widest text-white flex items-center gap-2">
                <Clock size={12} className="text-amber" /> Performance Projection Matrix
             </span>
             <span className="text-[9px] text-dim mono-metric">DATASET_ID: GPU_FRAG_2026_04_03</span>
          </div>
          
          <div className="flex-1 flex overflow-hidden">
             {/* ── DATA GRID ── */}
             <div className="flex-1 overflow-y-auto w-1/2">
                <table className="w-full text-left border-collapse">
                   <thead>
                      <tr className="border-b border-glass-border font-bold text-[9px] uppercase tracking-widest text-dim bg-black/20">
                         <th className="px-6 py-4">Benchmarking Dimension</th>
                         <th className="px-6 py-4">Baseline</th>
                         <th className="px-6 py-4">Workflow</th>
                         <th className="px-6 py-4">Variance</th>
                      </tr>
                   </thead>
                   <tbody className="text-xs mono-metric">
                      <tr className="border-b border-glass-border hover:bg-white/5 transition-colors">
                         <td className="px-6 py-4 text-dim">OOM Rate (%)</td>
                         <td className="px-6 py-4 text-red-400">{baseline.oom_rate.toFixed(1)}%</td>
                         <td className="px-6 py-4 text-green-400">{workflow.oom_rate.toFixed(1)}%</td>
                         <td className="px-6 py-4 text-amber font-bold">-{baseline.oom_rate - workflow.oom_rate}%</td>
                      </tr>
                      <tr className="border-b border-glass-border hover:bg-white/5 transition-colors">
                         <td className="px-6 py-4 text-dim">Steps to OOM</td>
                         <td className="px-6 py-4 text-red-400">{baseline.runs_to_oom.toFixed(1)}</td>
                         <td className="px-6 py-4 text-green-400">{workflow.runs_to_oom.toFixed(1)}</td>
                         <td className="px-6 py-4 text-green font-bold">+{workflow.runs_to_oom - baseline.runs_to_oom}</td>
                      </tr>
                      <tr className="border-b border-glass-border hover:bg-white/5 transition-colors">
                         <td className="px-6 py-4 text-dim">Max Batch</td>
                         <td className="px-6 py-4 text-dim">{baseline.stable_batch_size}</td>
                         <td className="px-6 py-4 text-green">{workflow.stable_batch_size}</td>
                         <td className="px-6 py-4 text-green font-bold">+{workflow.stable_batch_size - baseline.stable_batch_size}x</td>
                      </tr>
                      <tr className="hover:bg-white/5 transition-colors">
                         <td className="px-6 py-4 text-dim">Avg Frag</td>
                         <td className="px-6 py-4 text-red-300">{baseline.avg_frag.toFixed(3)}</td>
                         <td className="px-6 py-4 text-green-300">{workflow.avg_frag.toFixed(3)}</td>
                         <td className="px-6 py-4 text-green font-bold">{(100 * (1 - workflow.avg_frag/baseline.avg_frag)).toFixed(1)}%</td>
                      </tr>
                   </tbody>
                </table>
             </div>

             {/* ── CHART SIDEBAR ── */}
             <div className="flex-1 border-l border-glass-border p-4 flex flex-col bg-black/10 w-1/2">
                <div className="flex items-center justify-between mb-4">
                   <span className="text-[10px] font-bold uppercase tracking-widest text-dim flex items-center gap-2">
                      <TrendingDown size={12} className="text-green" /> Fragmentation Stability Trend
                   </span>
                   <div className="flex gap-4 text-[9px] uppercase font-bold mono-metric">
                      <div className="flex items-center gap-1"><div className="w-2 h-2 bg-red-400"></div> Baseline</div>
                      <div className="flex items-center gap-1"><div className="w-2 h-2 bg-green-400"></div> Workflow</div>
                   </div>
                </div>
                <div className="flex-1 min-h-[200px]">
                   <FragmentationChart data={chartData} timeline={[]} />
                </div>
             </div>
          </div>
       </div>
    </div>
  );
}
