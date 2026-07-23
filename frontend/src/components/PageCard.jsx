import React, { useState } from 'react';
import { Download, Sparkles, Target, Layers, Tag, MapPin, CheckCircle, Clock } from 'lucide-react';

export default function PageCard({ page, darkMode }) {
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
    <div className={`glass-panel rounded-3xl p-6 border ${darkMode ? 'border-slate-800 bg-slate-900/80 text-slate-100' : 'border-slate-200 bg-white/90 text-slate-900'} transition-all duration-300 shadow-xl flex flex-col justify-between`}>
      <div>
        {/* Header */}
        <div className={`flex items-center justify-between pb-4 border-b ${darkMode ? 'border-slate-800' : 'border-slate-200'}`}>
          <div className="flex items-center space-x-3">
            <span className="px-3 py-1 rounded-xl bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30 text-xs font-bold uppercase tracking-wider">
              Page {page_number.toString().padStart(2, '0')}
            </span>
            <span className={`text-xs font-mono px-2.5 py-1 rounded-lg border ${darkMode ? 'text-slate-300 bg-slate-800 border-slate-700' : 'text-slate-700 bg-slate-100 border-slate-200'}`}>
              {output_filename}
            </span>
          </div>

          <div className={`flex items-center gap-2 text-xs ${darkMode ? 'text-slate-400' : 'text-slate-500'}`}>
            <Clock className={`w-3.5 h-3.5 ${darkMode ? 'text-slate-500' : 'text-slate-400'}`} />
            <span>{processing_time_ms} ms</span>
          </div>
        </div>

        {/* Question & Parsed Pills */}
        <div className="my-4">
          <p className={`text-sm ${darkMode ? 'text-slate-300' : 'text-slate-700'} font-medium line-clamp-2 mb-3 italic`}>
            "{raw_question}"
          </p>

          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className={`p-2 rounded-xl border flex items-center gap-2 ${darkMode ? 'bg-slate-950/80 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
              <Tag className="w-3.5 h-3.5 text-amber-500 dark:text-amber-400 shrink-0" />
              <span className={darkMode ? 'text-slate-400' : 'text-slate-500'}>Object:</span>
              <span className={`font-bold ${darkMode ? 'text-white' : 'text-slate-900'} truncate`}>{parsed_question.object}</span>
            </div>

            <div className={`p-2 rounded-xl border flex items-center gap-2 ${darkMode ? 'bg-slate-950/80 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
              <Sparkles className="w-3.5 h-3.5 text-sky-500 dark:text-sky-400 shrink-0" />
              <span className={darkMode ? 'text-slate-400' : 'text-slate-500'}>Color:</span>
              <span className={`font-bold ${darkMode ? 'text-sky-300' : 'text-sky-600'} capitalize`}>{parsed_question.color || 'N/A'}</span>
            </div>

            <div className={`p-2 rounded-xl border flex items-center gap-2 ${darkMode ? 'bg-slate-950/80 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
              <MapPin className="w-3.5 h-3.5 text-rose-500 dark:text-rose-400 shrink-0" />
              <span className={darkMode ? 'text-slate-400' : 'text-slate-500'}>Position:</span>
              <span className={`font-bold ${darkMode ? 'text-rose-300' : 'text-rose-600'} capitalize`}>{parsed_question.position || 'Auto'}</span>
            </div>

            <div className={`p-2 rounded-xl border flex items-center gap-2 ${darkMode ? 'bg-slate-950/80 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
              <Target className="w-3.5 h-3.5 text-emerald-500 dark:text-emerald-400 shrink-0" />
              <span className={darkMode ? 'text-slate-400' : 'text-slate-500'}>Conf:</span>
              <span className={`font-bold ${darkMode ? 'text-emerald-300' : 'text-emerald-600'}`}>{(confidence * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>

        {/* Detection Metadata Banner */}
        <div className={`my-3 p-3 rounded-xl border text-xs space-y-1.5 ${darkMode ? 'bg-slate-950/60 border-slate-800/80 text-slate-400' : 'bg-slate-50 border-slate-200 text-slate-600'}`}>
          <div className="flex justify-between items-center">
            <span>Grounding DINO Prompt:</span>
            <span className={`font-mono px-2 py-0.5 rounded border ${darkMode ? 'text-emerald-400 bg-emerald-950/60 border-emerald-800/50' : 'text-emerald-700 bg-emerald-100/80 border-emerald-300'}`}>
              "{detection_prompt}"
            </span>
          </div>

          <div className="flex justify-between items-center">
            <span>Segmentation Method:</span>
            <span className={`font-semibold flex items-center gap-1 ${sam2_used ? (darkMode ? 'text-teal-400' : 'text-teal-600') : (darkMode ? 'text-amber-400' : 'text-amber-600')}`}>
              <Layers className="w-3 h-3" /> {sam2_used ? 'SAM 2 Mask' : 'Bounding Box Crop'}
            </span>
          </div>
        </div>

        {/* Preview View Mode Switcher */}
        <div className={`flex items-center justify-between p-1 rounded-xl border my-3 ${darkMode ? 'bg-slate-950/90 border-slate-800' : 'bg-slate-100 border-slate-200'}`}>
          <button
            onClick={() => setViewMode('cropped')}
            className={`flex-1 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              viewMode === 'cropped'
                ? 'bg-emerald-500 text-slate-950 shadow-md'
                : darkMode ? 'text-slate-400 hover:text-white' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Cropped Object PNG
          </button>
          <button
            onClick={() => setViewMode('overlay')}
            className={`flex-1 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              viewMode === 'overlay'
                ? 'bg-emerald-500 text-slate-950 shadow-md'
                : darkMode ? 'text-slate-400 hover:text-white' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Detection BBox Overlay
          </button>
        </div>

        {/* Display Image Container */}
        <div className={`relative rounded-2xl overflow-hidden border h-64 flex items-center justify-center group ${darkMode ? 'bg-slate-950 border-slate-800' : 'bg-slate-100 border-slate-200'}`}>
          <img
            src={viewMode === 'cropped' ? output_url : preview_url}
            alt={output_filename}
            className="max-h-full max-w-full object-contain p-2 transition-transform duration-300 group-hover:scale-105"
          />

          <a
            href={output_url}
            download={output_filename}
            className={`absolute bottom-3 right-3 p-2.5 rounded-xl border transition-all shadow-lg backdrop-blur-md ${darkMode ? 'bg-slate-900/90 text-white border-slate-700 hover:bg-emerald-500 hover:text-slate-950' : 'bg-white/90 text-slate-800 border-slate-300 hover:bg-emerald-500 hover:text-slate-950'}`}
            title={`Download ${output_filename}`}
          >
            <Download className="w-4 h-4" />
          </a>
        </div>
      </div>
    </div>
  );
}
