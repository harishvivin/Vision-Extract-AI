import React, { useState } from 'react';
import { Download, Sparkles, Target, Layers, Tag, MapPin, CheckCircle, Clock } from 'lucide-react';

export default function PageCard({ page }) {
  const [viewMode, setViewMode] = useState('cropped'); // 'cropped' | 'overlay'

  const {
    page_number,
    raw_question,
    parsed_question,
    detection_prompt,
    confidence,
    bounding_box,
    spatial_score,
    sam2_used,
    processing_time_ms,
    output_filename,
    output_url,
    preview_url
  } = page;

  return (
    <div className="glass-panel rounded-3xl p-6 border border-slate-800/80 hover:border-slate-700/80 transition-all duration-300 shadow-xl flex flex-col justify-between">
      <div>
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-slate-800">
          <div className="flex items-center space-x-3">
            <span className="px-3 py-1 rounded-xl bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 text-xs font-bold uppercase tracking-wider">
              Page {page_number.toString().padStart(2, '0')}
            </span>
            <span className="text-xs font-mono text-slate-400 bg-slate-800 px-2.5 py-1 rounded-lg border border-slate-700">
              {output_filename}
            </span>
          </div>

          <div className="flex items-center gap-2 text-xs text-slate-400">
            <Clock className="w-3.5 h-3.5 text-slate-500" />
            <span>{processing_time_ms} ms</span>
          </div>
        </div>

        {/* Question & Parsed Pills */}
        <div className="my-4">
          <p className="text-sm text-slate-300 font-medium line-clamp-2 mb-3 italic">
            "{raw_question}"
          </p>

          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-slate-900/80 border border-slate-800 p-2 rounded-xl flex items-center gap-2">
              <Tag className="w-3.5 h-3.5 text-amber-400 shrink-0" />
              <span className="text-slate-400">Object:</span>
              <span className="font-bold text-white truncate">{parsed_question.object}</span>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 p-2 rounded-xl flex items-center gap-2">
              <Sparkles className="w-3.5 h-3.5 text-sky-400 shrink-0" />
              <span className="text-slate-400">Color:</span>
              <span className="font-bold text-sky-300 capitalize">{parsed_question.color || 'N/A'}</span>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 p-2 rounded-xl flex items-center gap-2">
              <MapPin className="w-3.5 h-3.5 text-rose-400 shrink-0" />
              <span className="text-slate-400">Position:</span>
              <span className="font-bold text-rose-300 capitalize">{parsed_question.position || 'Auto'}</span>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 p-2 rounded-xl flex items-center gap-2">
              <Target className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
              <span className="text-slate-400">Conf:</span>
              <span className="font-bold text-emerald-300">{(confidence * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>

        {/* Detection Metadata Banner */}
        <div className="my-3 p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 text-xs space-y-1.5">
          <div className="flex justify-between items-center text-slate-400">
            <span>Grounding DINO Prompt:</span>
            <span className="font-mono text-emerald-400 bg-emerald-950/40 px-2 py-0.5 rounded border border-emerald-800/50">
              "{detection_prompt}"
            </span>
          </div>

          <div className="flex justify-between items-center text-slate-400">
            <span>Segmentation Method:</span>
            <span className={`font-semibold flex items-center gap-1 ${sam2_used ? 'text-teal-400' : 'text-amber-400'}`}>
              <Layers className="w-3 h-3" /> {sam2_used ? 'SAM 2 Mask' : 'Bounding Box Crop'}
            </span>
          </div>
        </div>

        {/* Preview View Mode Switcher */}
        <div className="flex items-center justify-between bg-slate-900/90 p-1 rounded-xl border border-slate-800 my-3">
          <button
            onClick={() => setViewMode('cropped')}
            className={`flex-1 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              viewMode === 'cropped'
                ? 'bg-emerald-500 text-slate-950 shadow-md'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Cropped Object PNG
          </button>
          <button
            onClick={() => setViewMode('overlay')}
            className={`flex-1 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              viewMode === 'overlay'
                ? 'bg-emerald-500 text-slate-950 shadow-md'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Detection BBox Overlay
          </button>
        </div>

        {/* Display Image Container */}
        <div className="relative rounded-2xl overflow-hidden bg-slate-950 border border-slate-800 h-64 flex items-center justify-center group">
          <img
            src={viewMode === 'cropped' ? output_url : preview_url}
            alt={output_filename}
            className="max-h-full max-w-full object-contain p-2 transition-transform duration-300 group-hover:scale-105"
          />

          <a
            href={output_url}
            download={output_filename}
            className="absolute bottom-3 right-3 p-2.5 rounded-xl bg-slate-900/80 hover:bg-emerald-500 text-white hover:text-slate-950 border border-slate-700 transition-all shadow-lg backdrop-blur-md"
            title={`Download ${output_filename}`}
          >
            <Download className="w-4 h-4" />
          </a>
        </div>
      </div>
    </div>
  );
}
