"use client";

import React, { useState, useEffect } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer,
  LineChart, Line
} from 'recharts';
import { ShieldAlert, Zap, Target, Activity, Cpu, Loader2 } from 'lucide-react';
import { DashboardDataSchema, DashboardData } from '@/lib/schemas';

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<'overview' | 'training'>('overview');
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Fetch dữ liệu từ FastAPI Backend
    const fetchData = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/metrics');
        if (!response.ok) throw new Error('Không thể kết nối đến Backend FastAPI (Cổng 8000)');
        
        const json = await response.json();
        // Validation nghiêm ngặt bằng Zod trước khi đưa vào State
        const validatedData = DashboardDataSchema.parse(json);
        setData(validatedData);
      } catch (err: any) {
        console.error(err);
        setError(err.message || 'Lỗi khi parse dữ liệu từ Backend');
      }
    };
    
    fetchData();
  }, []);

  if (error) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center text-rose-400 p-8 text-center">
        <div className="bg-slate-900/80 p-8 rounded-2xl border border-rose-900/50">
          <ShieldAlert className="w-12 h-12 mx-auto mb-4" />
          <h2 className="text-xl font-bold mb-2">Lỗi Kết Nối</h2>
          <p className="text-slate-400">{error}</p>
          <p className="text-sm mt-4 text-slate-500">Hãy chắc chắn rằng bạn đã chạy `python api.py`</p>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center text-slate-300">
        <Loader2 className="w-10 h-10 animate-spin text-indigo-500 mb-4" />
        <p className="animate-pulse">Đang tải dữ liệu từ Mô hình Deep Learning...</p>
      </div>
    );
  }

  const getMetricColor = (value: number, type: 'high' | 'low' = 'high') => {
    if (type === 'high') {
      if (value >= 0.8) return 'text-emerald-400';
      if (value >= 0.5) return 'text-amber-400';
      return 'text-rose-400';
    } else {
      if (value <= 50) return 'text-emerald-400';
      if (value <= 100) return 'text-amber-400';
      return 'text-rose-400';
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-8 font-sans selection:bg-indigo-500/30">
      <div className="max-w-7xl mx-auto space-y-8">
        
        {/* Header */}
        <header className="flex items-center justify-between border-b border-slate-800 pb-6">
          <div>
            <h1 className="text-4xl font-extrabold tracking-tight bg-gradient-to-r from-indigo-400 via-cyan-400 to-emerald-400 bg-clip-text text-transparent">
              Deep Learning Benchmark
            </h1>
            <p className="text-slate-400 mt-2 text-lg">Multi-Scale CNN vs Standard CNN vs Baseline KNN</p>
          </div>
          <div className="flex gap-2 bg-slate-900/50 p-1.5 rounded-lg border border-slate-800">
            <button 
              onClick={() => setActiveTab('overview')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${activeTab === 'overview' ? 'bg-indigo-500/20 text-indigo-300' : 'text-slate-400 hover:text-slate-200'}`}
            >
              System Overview
            </button>
            <button 
              onClick={() => setActiveTab('training')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${activeTab === 'training' ? 'bg-indigo-500/20 text-indigo-300' : 'text-slate-400 hover:text-slate-200'}`}
            >
              Training Dynamics
            </button>
          </div>
        </header>

        {activeTab === 'overview' ? (
          <div className="space-y-8 animate-[fadeIn_0.5s_ease-in-out]">
            {/* Model Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {data.metrics.map((model) => (
                <div key={model.id} className="relative group bg-slate-900/40 rounded-2xl p-6 border border-slate-800 hover:border-indigo-500/50 transition-all duration-300 hover:shadow-[0_0_30px_-5px_rgba(99,102,241,0.15)] overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                  
                  <h3 className="text-xl font-bold text-slate-100 flex items-center gap-2 mb-2">
                    {model.id === 'multi-scale-cnn' ? <ShieldAlert className="w-5 h-5 text-emerald-400" /> : <Activity className="w-5 h-5 text-slate-500" />}
                    {model.name}
                  </h3>
                  <p className="text-sm text-slate-400 mb-6 min-h-[40px]">{model.description}</p>
                  
                  <div className="space-y-4">
                    <div className="flex justify-between items-center bg-slate-950/50 p-3 rounded-lg">
                      <span className="text-slate-400 text-sm flex items-center gap-2"><Target className="w-4 h-4" /> mAP Score</span>
                      <span className={`font-mono font-bold text-lg ${getMetricColor(model.mAP)}`}>{(model.mAP * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between items-center bg-slate-950/50 p-3 rounded-lg">
                      <span className="text-slate-400 text-sm flex items-center gap-2"><Zap className="w-4 h-4" /> Inference</span>
                      <span className={`font-mono font-bold text-lg ${getMetricColor(model.inferenceTimeMs, 'low')}`}>{model.inferenceTimeMs}ms</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Charts Section */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* mAP Comparison Chart */}
              <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6">
                <h3 className="text-lg font-semibold mb-6 flex items-center gap-2"><Target className="w-5 h-5 text-indigo-400" /> Mean Average Precision (mAP)</h3>
                <div className="h-[300px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data.metrics} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                      <XAxis dataKey="name" stroke="#64748b" tick={{ fill: '#64748b' }} axisLine={false} tickLine={false} />
                      <YAxis stroke="#64748b" tick={{ fill: '#64748b' }} axisLine={false} tickLine={false} domain={[0, 1]} tickFormatter={(val) => `${val * 100}%`} />
                      <RechartsTooltip 
                        cursor={{ fill: '#1e293b', opacity: 0.4 }}
                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px', color: '#f1f5f9' }}
                        formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'mAP']}
                      />
                      <Bar dataKey="mAP" fill="#6366f1" radius={[4, 4, 0, 0]} maxBarSize={60}>
                        {data.metrics.map((entry, index) => (
                          <cell key={`cell-${index}`} fill={entry.id === 'multi-scale-cnn' ? '#34d399' : entry.id === 'standard-cnn' ? '#818cf8' : '#475569'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Inference Time Comparison Chart */}
              <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6">
                <h3 className="text-lg font-semibold mb-6 flex items-center gap-2"><Zap className="w-5 h-5 text-amber-400" /> Inference Time (Lower is Better)</h3>
                <div className="h-[300px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data.metrics} layout="vertical" margin={{ top: 10, right: 30, left: 20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                      <XAxis type="number" stroke="#64748b" tick={{ fill: '#64748b' }} axisLine={false} tickLine={false} unit="ms" />
                      <YAxis dataKey="name" type="category" stroke="#64748b" tick={{ fill: '#64748b' }} axisLine={false} tickLine={false} width={100} />
                      <RechartsTooltip 
                        cursor={{ fill: '#1e293b', opacity: 0.4 }}
                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px', color: '#f1f5f9' }}
                        formatter={(value: number) => [`${value} ms`, 'Inference Time']}
                      />
                      <Bar dataKey="inferenceTimeMs" fill="#fbbf24" radius={[0, 4, 4, 0]} maxBarSize={40}>
                         {data.metrics.map((entry, index) => (
                          <cell key={`cell-${index}`} fill={entry.id === 'multi-scale-cnn' ? '#fbbf24' : entry.id === 'standard-cnn' ? '#34d399' : '#ef4444'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-8 animate-[fadeIn_0.5s_ease-in-out]">
             {/* Training Dynamics */}
             <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6">
                <h3 className="text-lg font-semibold mb-6 flex items-center gap-2"><Activity className="w-5 h-5 text-cyan-400" /> Training Loss Convergence</h3>
                <div className="h-[400px] w-full mt-4">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data.trainingHistory} margin={{ top: 10, right: 30, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                      <XAxis dataKey="epoch" stroke="#64748b" tick={{ fill: '#64748b' }} axisLine={false} tickLine={false} tickFormatter={(val) => `Ep ${val}`} />
                      <YAxis stroke="#64748b" tick={{ fill: '#64748b' }} axisLine={false} tickLine={false} />
                      <RechartsTooltip 
                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px', color: '#f1f5f9' }}
                      />
                      <Legend wrapperStyle={{ paddingTop: '20px' }} />
                      <Line type="monotone" name="Multi-Scale CNN Loss" dataKey="multiScaleLoss" stroke="#34d399" strokeWidth={3} dot={{ r: 4, fill: '#34d399', strokeWidth: 0 }} activeDot={{ r: 6 }} />
                      <Line type="monotone" name="Standard CNN Loss" dataKey="standardLoss" stroke="#818cf8" strokeWidth={3} dot={{ r: 4, fill: '#818cf8', strokeWidth: 0 }} activeDot={{ r: 6 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
             </div>
          </div>
        )}

      </div>
    </div>
  );
}
