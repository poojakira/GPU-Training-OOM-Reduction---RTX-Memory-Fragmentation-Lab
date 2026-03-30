import React, { useState, useEffect } from 'react';
import { fetchLiveTelemetry, fetchBenchmarkStats } from './utils/dataLoader';
import { MemoryMap } from './components/MemoryMap';
import { FragmentationChart } from './components/FragmentationChart';
import { IterationTimeChart } from './components/IterationTimeChart';
import { 
  Activity, 
  Cpu, 
  Zap, 
  LayoutDashboard, 
  Database, 
  Settings, 
  Terminal,
  Eye,
  History,
  ShieldCheck,
  Server
} from 'lucide-react';

// Simulated Hardware Sparkline component
const Sparkline = ({ data, color }) => (
  <svg viewBox="0 0 100 20" className="w-full h-8 opacity-70">
    <polyline 
      fill="none" 
      stroke={color} 
      strokeWidth="2" 
      points={data.map((v, i) => `${i * (100 / (data.length - 1))},${20 - (v / 100) * 20}`).join(' ')} 
    />
  </svg>
);

function App() {
  const [viewMode, setViewMode] = useState('live');
  const [activeSection, setActiveSection] = useState('dashboard');
  const [connected, setConnected] = useState(false);
  const [data, setData] = useState({ benchmark: null, live: null, liveHistory: [] });
  const [dummyHw, setDummyHw] = useState({
    smActive: Array.from({length: 20}, () => Math.random() * 100),
    tensorCore: Array.from({length: 20}, () => Math.random() * 100),
    pcieRx: Array.from({length: 20}, () => Math.random() * 100),
    pcieTx: Array.from({length: 20}, () => Math.random() * 100)
  });

  useEffect(() => {
    fetchBenchmarkStats().then(benchmark => setData(prev => ({ ...prev, benchmark })));
    
    const interval = setInterval(() => {
      // Simulate hardware noise
      setDummyHw(prev => ({
        smActive: [...prev.smActive.slice(1), 70 + Math.random() * 30],
        tensorCore: [...prev.tensorCore.slice(1), 40 + Math.random() * 60],
        pcieRx: [...prev.pcieRx.slice(1), Math.random() * 100],
        pcieTx: [...prev.pcieTx.slice(1), Math.random() * 100]
      }));

      fetchLiveTelemetry().then(live => {
        if (live) {
          setConnected(true);
          setData(prev => {
            const newHistory = [...prev.liveHistory, {
              iteration: prev.liveHistory.length,
              defragFrag: parseFloat(live.currentFrag),
              baselineFrag: prev.benchmark ? parseFloat(prev.benchmark.baseline.chart[prev.liveHistory.length % prev.benchmark.baseline.chart.length]?.frag || 60) : 60,
              defragTime: parseFloat(live.avgTime),
              baselineTime: prev.benchmark ? parseFloat(prev.benchmark.baseline.avgTime) : 0
            }].slice(-100);

            return { ...prev, live, liveHistory: newHistory };
          });
        } else {
          setConnected(false);
        }
      });
    }, 1000);
    
    return () => clearInterval(interval);
  }, []);

  const metrics = viewMode === 'live' && data.live ? {
    currentAlloc: data.live.currentAllocated,
    currentRes: data.live.currentReserved,
    frag: data.live.currentFrag,
    compactions: data.live.totalCompactions,
    freed: data.live.totalFreed
  } : {
    currentAlloc: data.benchmark?.defrag.peakMem || "6314",
    currentRes: data.benchmark?.defrag.peakMem || "6314",
    frag: "58.7",
    compactions: 14,
    freed: 40498
  };

  const chartData = viewMode === 'live' ? data.liveHistory : (data.benchmark?.defrag.chart || []);

  if (!data.benchmark && !data.live) {
    return <div className="h-screen w-screen flex items-center justify-center font-bold animate-pulse text-brand" style={{background: '#000'}}>NSIGHT PROFILER INITIALIZING...</div>;
  }

  return (
    <>
      <aside className="sidebar">
        <div className="logo-section">
          <div className="logo-icon">
            <Cpu size={20} color="#000" />
          </div>
          <span className="text-xl font-bold tracking-tighter text-white">SYSTEMS <span className="text-brand">NSIGHT</span></span>
        </div>

        <nav className="flex-1">
          <button onClick={() => setActiveSection('dashboard')} className={`w-full nav-link ${activeSection === 'dashboard' ? 'active' : ''}`}>
             <LayoutDashboard size={18} /><span>Mission Control</span>
          </button>
          <button onClick={() => setActiveSection('telemetry')} className={`w-full nav-link ${activeSection === 'telemetry' ? 'active' : ''}`}>
            <Activity size={18} /><span>Hardware Telemetry</span>
          </button>
          <button onClick={() => setActiveSection('memory')} className={`w-full nav-link ${activeSection === 'memory' ? 'active' : ''}`}>
            <Database size={18} /><span>VRAM Topology</span>
          </button>
          <button onClick={() => setActiveSection('console')} className={`w-full nav-link ${activeSection === 'console' ? 'active' : ''}`}>
            <Terminal size={18} /><span>Execution Trace</span>
          </button>
          <div className="h-[1px] bg-glass-border my-4" />
          <div className="px-3 mb-2 text-10px uppercase tracking-widest text-brand font-bold">Data Capture</div>
          <button onClick={() => setViewMode('live')} className={`w-full nav-link ${viewMode === 'live' ? 'active-mode' : ''}`}>
            <Eye size={16} /><span>Live Monitor</span>
          </button>
          <button onClick={() => setViewMode('benchmark')} className={`w-full nav-link ${viewMode === 'benchmark' ? 'active-mode' : ''}`}>
            <History size={16} /><span>Static Analysis</span>
          </button>
        </nav>

        <div className="glass-card mt-auto p-4 border-brand/20">
          <div className="flex items-center gap-2 mb-2">
            <ShieldCheck size={14} className="text-brand" />
            <span className="text-[10px] uppercase font-bold text-secondary">Allocator Hook: ACTIVE</span>
          </div>
          <p className="text-[10px] text-brand font-mono">DDP SYNC: DETECTED</p>
        </div>
      </aside>

      <main className="main-content">
        <header className="top-nav">
          <div className="flex items-center gap-4">
            <div className={`live-indicator ${connected ? '' : 'disconnected'}`} style={{ backgroundColor: connected ? 'var(--brand)' : 'var(--accent-red)', boxShadow: connected ? '0 0 10px var(--brand)' : '0 0 10px var(--accent-red)' }} />
            <span className="text-sm font-medium text-white">LINK: {connected ? 'ESTABLISHED' : 'DISCONNECTED'}</span>
            <span className="text-xs text-brand font-mono">/ NVIDIA RTX 4090 / CUDA 12.1</span>
          </div>

          <div className="flex items-center gap-4">
            <span className="text-xs font-mono text-secondary">{new Date().toLocaleTimeString()}</span>
            <div className="flex items-center gap-2 px-3 py-1.5 border border-brand/40 rounded-full text-brand text-[10px] font-bold uppercase tracking-widest" style={{background: 'rgba(118, 185, 0, 0.1)'}}>
              <Server size={12} fill="currentColor" />
              Node 00
            </div>
          </div>
        </header>

        <div className="container">
          {activeSection === 'dashboard' && (
            <div className="flex flex-col gap-4 h-full">
              {/* Top Dense Row */}
              <div className="grid grid-cols-5 gap-4">
                <div className="glass-card p-4">
                  <div className="kpi-title">Active SMs</div>
                  <div className="text-2xl font-mono text-white mb-2">{dummyHw.smActive[19].toFixed(1)}%</div>
                  <Sparkline data={dummyHw.smActive} color="#00ffcc" />
                </div>
                <div className="glass-card p-4">
                  <div className="kpi-title">Tensor Cores</div>
                  <div className="text-2xl font-mono text-white mb-2">{dummyHw.tensorCore[19].toFixed(1)}%</div>
                  <Sparkline data={dummyHw.tensorCore} color="#76b900" />
                </div>
                <div className="glass-card p-4">
                  <div className="kpi-title">PCIe RX Bandwidth</div>
                  <div className="text-2xl font-mono text-white mb-2">{(dummyHw.pcieRx[19] / 4).toFixed(2)} GB/s</div>
                  <Sparkline data={dummyHw.pcieRx} color="#00ffcc" />
                </div>
                <div className="glass-card p-4">
                  <div className="kpi-title">Total Allocated</div>
                  <div className="text-2xl font-mono text-white mb-2">{metrics.currentAlloc} MB</div>
                  <div className="text-[10px] text-secondary mt-2">VRAM ADDRESS BOUNDS</div>
                </div>
                <div className="glass-card p-4" style={{border: '1px solid #ff3333', background: 'rgba(255, 51, 51, 0.05)'}}>
                  <div className="kpi-title text-accent-red">Address Gap (FRAG)</div>
                  <div className="text-2xl font-mono text-accent-red mb-2">{metrics.frag}%</div>
                  <div className="text-[10px] text-accent-red mt-2">DEFRAGMENTER TRIGGER PROBABILITY</div>
                </div>
              </div>

              {/* Main Visualization Grid */}
              <div className="grid grid-cols-12 gap-4 flex-1 min-h-[500px]">
                {/* Left Side: Fragmentation Stream */}
                <div className="col-span-8 glass-card border-brand/20 flex flex-col">
                  <div className="flex justify-between items-center mb-4">
                    <span className="kpi-title text-brand">VRAM Address Fragmentation Stream</span>
                    <div className="flex gap-4">
                      <span className="text-[10px] uppercase font-mono text-white flex items-center gap-1"><div className="w-2 h-2 rounded bg-accent-red"/> Unmanaged (Cache Release)</span>
                      <span className="text-[10px] uppercase font-mono text-white flex items-center gap-1"><div className="w-2 h-2 rounded bg-brand"/> gpudefrag (Triton Repack)</span>
                    </div>
                  </div>
                  <div className="flex-1 w-full min-h-[400px]">
                    <FragmentationChart data={chartData} timeline={viewMode === 'live' && data.live ? data.live.history : []} />
                  </div>
                </div>
                
                {/* Right Side: Topology & Compactions */}
                <div className="col-span-4 flex flex-col gap-4">
                  <div className="glass-card flex-1 flex flex-col">
                    <div className="flex justify-between items-center mb-4">
                      <h3 className="kpi-title">VRAM Block Topology</h3>
                      <span className="text-[10px] font-mono text-secondary">PAGE OFFSET: 0x000</span>
                    </div>
                    <div className="flex-1 overflow-hidden" style={{transform: 'scale(1.1)', transformOrigin: 'top left', marginLeft: '10px'}}>
                      <MemoryMap fragPercent={metrics.frag} />
                    </div>
                  </div>
                  <div className="glass-card h-32 flex justify-between items-center px-8 border-brand/30" style={{background: 'rgba(118, 185, 0, 0.05)'}}>
                    <div>
                      <div className="kpi-title text-brand">Physical Compactions</div>
                      <div className="text-4xl font-black font-mono text-white">{metrics.compactions}</div>
                    </div>
                    <div>
                      <div className="kpi-title text-brand">Reclaimed Vol</div>
                      <div className="text-4xl font-black font-mono text-brand">+{metrics.freed} <span className="text-xl">MB</span></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Tab Rendering Logic */}
          {activeSection !== 'dashboard' && activeSection !== 'telemetry' && activeSection !== 'console' && activeSection !== 'memory' && (
             <div className="glass-card h-full w-full flex items-center justify-center">
                 <div className="text-center font-mono opacity-50">
                     <Database size={48} className="mx-auto mb-4" />
                     <h3>COMPONENT STREAMING</h3>
                     <p className="text-[10px] mt-2 text-brand">AWAITING BACKEND ALLOCATOR...</p>
                 </div>
             </div>
          )}

          {activeSection === 'telemetry' && (
            <div className="w-full h-full">
               <div className="glass-card h-full flex flex-col p-6">
                 <span className="kpi-title text-brand mb-4">Extended Telemetry Feed (Iteration Times)</span>
                 <div className="w-full h-[500px]">
                    <IterationTimeChart 
                      data={viewMode === 'live' ? data.liveHistory : (data.benchmark?.defrag.chart || [])} 
                    />
                 </div>
               </div>
            </div>
          )}

          {activeSection === 'console' && (
            <div className="w-full h-[600px] flex">
              <div className="glass-card flex-1 flex flex-col p-6">
                <div className="flex justify-between items-center mb-6 border-b border-brand/20 pb-4">
                  <span className="kpi-title text-brand">Active Execution Trace Log</span>
                  <div className="text-[10px] text-brand font-mono px-2 py-1 bg-brand/10 rounded">CHANNEL: 01_FRAG_SENSE</div>
                </div>
                <div className="flex-1 overflow-y-auto">
                  <table className="w-full text-left font-mono text-sm border-collapse">
                    <thead className="text-[10px] text-secondary uppercase border-b border-white/10">
                      <tr>
                        <th className="py-2">Timestamp</th>
                        <th className="py-2">Event</th>
                        <th className="py-2">Impact</th>
                        <th className="py-2">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.live?.history?.length > 0 ? (
                        data.live.history.map(h => (
                          <tr key={h.id} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                            <td className="py-2 text-dim">{h.timestamp}</td>
                            <td className="py-2 font-bold text-white">COMPACTION_EVENT_{h.id}</td>
                            <td className="py-2 text-brand">+{h.freed}MB</td>
                            <td className="py-2 text-brand">COMPLETE</td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan="4" className="text-center py-20 opacity-30 italic">Streaming from Allocation Hooks... Awaiting first trigger.</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {activeSection === 'memory' && (
            <div className="w-full glass-card h-[600px] flex flex-col p-6">
                <div className="flex justify-between items-center mb-6">
                  <span className="kpi-title text-brand">Physical VRAM Topology (Address View)</span>
                  <div className="text-[10px] text-brand font-mono">ADDR RANGE: 0x0000 - 0xFFFF</div>
                </div>
                <div className="flex-1 w-full overflow-hidden flex items-center justify-center">
                    <div style={{transform: 'scale(1.5)', transformOrigin: 'center'}}>
                      <MemoryMap fragPercent={metrics.frag} />
                    </div>
                </div>
            </div>
          )}
        </div>
      </main>
    </>
  );
}

export default App;
