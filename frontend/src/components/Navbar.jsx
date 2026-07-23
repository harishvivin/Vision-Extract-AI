import React from 'react';
import { Eye, Download, FileText, Moon, Sun, Cpu } from 'lucide-react';

export default function Navbar({ onDownloadAll, isProcessing, totalPages, darkMode, setDarkMode, onOpenLogs }) {
  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-slate-200 dark:border-slate-800 px-6 py-4 flex items-center justify-between">
      <div className="flex items-center space-x-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-500 to-teal-400 flex items-center justify-center shadow-lg shadow-emerald-500/20">
          <Eye className="w-6 h-6 text-slate-950 font-bold" />
        </div>
        <div>
          <h1 className={`text-xl font-bold ${darkMode ? 'bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent' : 'text-slate-900'}`}>
            Vision Extract AI
          </h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 flex items-center gap-1">
            <Cpu className="w-3 h-3 text-emerald-500 dark:text-emerald-400" /> Grounding DINO + SAM 2 PDF Object Extractor
          </p>
        </div>
      </div>

      <div className="flex items-center space-x-4">
        <button
          onClick={onOpenLogs}
          className="px-3 py-2 text-xs font-medium text-slate-700 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white bg-slate-100 dark:bg-slate-800/80 hover:bg-slate-200 dark:hover:bg-slate-700 border border-slate-200 dark:border-slate-700/60 rounded-lg flex items-center gap-1.5 transition-all"
        >
          <FileText className="w-4 h-4 text-sky-500 dark:text-sky-400" /> Logs & Telemetry
        </button>

        {totalPages > 0 && (
          <button
            onClick={onDownloadAll}
            disabled={isProcessing}
            className="px-4 py-2 text-sm font-semibold text-slate-950 bg-gradient-to-r from-emerald-400 to-teal-300 hover:from-emerald-300 hover:to-teal-200 active:scale-95 disabled:opacity-50 rounded-xl shadow-lg shadow-emerald-500/25 flex items-center gap-2 transition-all"
            title="Download single PDF document containing all extracted object images"
          >
            <Download className="w-4 h-4" /> Download All Extracted Images (PDF)
          </button>
        )}

        <button
          onClick={() => setDarkMode(!darkMode)}
          className="p-2 rounded-xl bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 transition-colors"
          title="Toggle Theme"
        >
          {darkMode ? <Sun className="w-5 h-5 text-amber-400" /> : <Moon className="w-5 h-5 text-indigo-600" />}
        </button>
      </div>
    </header>
  );
}
