import React, { useState } from 'react';
import { UploadCloud, FileUp, Sparkles, CheckCircle2, FileText, ArrowRight } from 'lucide-react';

export default function UploadZone({ onFileUpload, onUseDefaultBenchmark, isProcessing }) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.name.toLowerCase().endsWith('.pdf')) {
        setSelectedFile(file);
        onFileUpload(file);
      } else {
        alert('Please drop a valid PDF file.');
      }
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      onFileUpload(file);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto my-8 px-4">
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className={`relative border-2 border-dashed rounded-3xl p-10 text-center transition-all duration-300 glass-panel ${dragActive
            ? 'border-emerald-400 bg-emerald-500/10 scale-[1.01]'
            : 'border-slate-300 dark:border-slate-700/80 hover:border-emerald-500/50 bg-white/80 dark:bg-slate-900/60'
          }`}
      >
        <input
          type="file"
          id="pdf-input"
          accept=".pdf"
          onChange={handleChange}
          className="hidden"
          disabled={isProcessing}
        />

        <div className="flex flex-col items-center justify-center space-y-4">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-emerald-500/20 to-teal-500/20 border border-emerald-500/30 flex items-center justify-center text-emerald-500 dark:text-emerald-400 shadow-inner">
            <UploadCloud className="w-8 h-8 animate-bounce" />
          </div>

          <div>
            <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-1">
              Upload Medical Report PDF File
            </h3>
            <p className="text-sm text-slate-600 dark:text-slate-400 max-w-md mx-auto">
              Drag & drop your PDF medical report or lab result file here. The AI will analyze and index the pages to answer your questions accurately.
            </p>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-3 pt-2">
            <label
              htmlFor="pdf-input"
              className={`px-6 py-3 rounded-xl font-bold text-sm cursor-pointer transition-all flex items-center gap-2 shadow-lg ${isProcessing
                  ? 'bg-slate-200 dark:bg-slate-800 text-slate-400 dark:text-slate-500 cursor-not-allowed'
                  : 'bg-emerald-500 hover:bg-emerald-600 text-slate-950 shadow-emerald-500/25'
                }`}
            >
              <FileUp className="w-4 h-4" /> Browse & Upload PDF
            </label>

            <button
              onClick={onUseDefaultBenchmark}
              disabled={isProcessing}
              className="px-6 py-3 rounded-xl font-semibold text-sm bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-800 dark:text-slate-200 border border-slate-300 dark:border-slate-700 flex items-center gap-2 transition-all"
            >
              <FileText className="w-4 h-4 text-indigo-400" /> Analyze Sample Medical Report (20 Pages) <ArrowRight className="w-4 h-4" />
            </button>
          </div>

          {selectedFile && (
            <div className="mt-3 inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-700 dark:text-emerald-300 text-xs font-medium">
              <CheckCircle2 className="w-4 h-4 text-emerald-500 dark:text-emerald-400" />
              Uploaded PDF: <span className="font-semibold text-slate-900 dark:text-white">{selectedFile.name}</span> ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
