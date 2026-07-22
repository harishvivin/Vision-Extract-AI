import React from 'react';
import { X, Terminal, FileCode, CheckCircle2 } from 'lucide-react';

export default function LogsModal({ isOpen, onClose, logs }) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-md">
      <div className="glass-panel w-full max-w-4xl max-h-[85vh] rounded-3xl border border-slate-800 flex flex-col overflow-hidden shadow-2xl">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/80">
          <div className="flex items-center space-x-2 text-emerald-400 font-bold text-base">
            <Terminal className="w-5 h-5" />
            <h3 className="text-white">Pipeline Execution Telemetry & Logs</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded-xl transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto font-mono text-xs text-slate-300 space-y-4 bg-slate-950/90">
          {logs && logs.length > 0 ? (
            logs.map((log, idx) => (
              <div key={idx} className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
                <div className="flex items-center justify-between text-emerald-400 font-bold border-b border-slate-800 pb-2">
                  <span className="flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" /> Page {log.page_number} ({log.output_filename})
                  </span>
                  <span className="text-slate-400 font-normal">{log.processing_time_ms} ms</span>
                </div>

                <div className="grid grid-cols-2 gap-2 text-slate-400 text-[11px] pt-1">
                  <div><span className="text-slate-500">Question:</span> {log.raw_question}</div>
                  <div><span className="text-slate-500">Detection Prompt:</span> "{log.detection_prompt}"</div>
                  <div><span className="text-slate-500">Confidence:</span> {(log.confidence * 100).toFixed(1)}%</div>
                  <div><span className="text-slate-500">Bounding Box:</span> [{log.bounding_box.map(b => b.toFixed(1)).join(', ')}]</div>
                  <div><span className="text-slate-500">Spatial Match Score:</span> {(log.spatial_score * 100).toFixed(1)}%</div>
                  <div><span className="text-slate-500">SAM2 Status:</span> {log.sam2_used ? 'Enabled & Executed' : 'Box Fallback'}</div>
                </div>

                {log.attempts_log && log.attempts_log.length > 0 && (
                  <div className="mt-2 pt-2 border-t border-slate-800 text-[10px] text-slate-500">
                    <span className="font-semibold text-slate-400">Retry Cascade Attempts ({log.attempts_log.length}):</span>
                    <pre className="mt-1 p-2 rounded bg-slate-950 text-slate-400 overflow-x-auto">
                      {JSON.stringify(log.attempts_log, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            ))
          ) : (
            <div className="text-center text-slate-500 py-12">
              <FileCode className="w-10 h-10 mx-auto mb-2 opacity-50" />
              No execution telemetry logs available yet. Run the extraction pipeline to populate logs.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
