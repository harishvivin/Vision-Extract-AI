import React from 'react';
import { Loader2, Sparkles } from 'lucide-react';

export default function ProgressBar({ progress, statusText }) {
  return (
    <div className="w-full max-w-3xl mx-auto my-6 px-4">
      <div className="glass-panel p-6 rounded-3xl border border-slate-800 shadow-2xl space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Loader2 className="w-5 h-5 text-emerald-400 animate-spin" />
            <span className="text-sm font-semibold text-white">{statusText || 'Processing PDF...'}</span>
          </div>
          <span className="text-sm font-bold text-emerald-400">{progress}%</span>
        </div>

        <div className="w-full h-3 bg-slate-900 rounded-full overflow-hidden p-0.5 border border-slate-800">
          <div
            className="h-full bg-gradient-to-r from-emerald-500 via-teal-400 to-cyan-400 rounded-full transition-all duration-500 shadow-lg shadow-emerald-500/50"
            style={{ width: `${progress}%` }}
          />
        </div>

        <div className="flex justify-between items-center text-xs text-slate-400 pt-1">
          <span className="flex items-center gap-1">
            <Sparkles className="w-3.5 h-3.5 text-amber-400" /> Grounding DINO Zero-Shot Detection
          </span>
          <span>SAM 2 Instance Masking</span>
        </div>
      </div>
    </div>
  );
}
