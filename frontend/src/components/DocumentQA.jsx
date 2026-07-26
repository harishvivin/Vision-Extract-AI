import React, { useState } from 'react';
import { CheckCircle2, ExternalLink, FileText, HelpCircle, Image as ImageIcon, Loader2, Search, Sparkles, X } from 'lucide-react';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'https://vision-extract-ai.onrender.com').replace(/\/$/, '');

function backendUrl(path) {
  return path?.startsWith('/') ? `${API_BASE_URL}${path}` : path;
}

export default function DocumentQA({ darkMode }) {
  const [question, setQuestion] = useState('');
  const [isAsking, setIsAsking] = useState(false);
  const [qaResult, setQaResult] = useState(null);
  const [qaError, setQaError] = useState('');
  const [zoomImage, setZoomImage] = useState(null);

  const sampleQuestions = [
    'What is the patient\'s name?', 'What is the haemoglobin level?',
    'What is the HbA1c percentage?', 'What is the creatinine level?',
    'What is the HIV test result?', 'Show ECG interpretation.', 'Summarize this report.',
  ];

  const handleAsk = async (queryText) => {
    const query = (queryText || question).trim();
    if (!query) return;
    setQuestion(query);
    setIsAsking(true);
    setQaResult(null);
    setQaError('');
    try {
      const response = await fetch(`${API_BASE_URL}/api/qa/ask`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: query }),
      });
      const data = await response.json().catch(() => null);
      if (!response.ok || !data?.success) {
        throw new Error(data?.detail || 'The document QA service did not return an answer.');
      }
      setQaResult({ ...data, snippet_url: backendUrl(data.snippet_url), preview_url: backendUrl(data.preview_url) });
    } catch (error) {
      console.error('Document QA request failed:', error);
      setQaError(error.message || 'Unable to reach the document QA service.');
    } finally {
      setIsAsking(false);
    }
  };

  return (
    <div className={`mt-8 mb-12 p-6 md:p-8 rounded-3xl ${darkMode ? 'bg-slate-900/80 border-slate-800' : 'bg-white border-slate-200'} border shadow-2xl`}>
      <div className="space-y-1 mb-6">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-semibold">
          <Sparkles className="w-3.5 h-3.5" /> Document-grounded medical report QA
        </div>
        <h3 className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-slate-900'}`}>Ask Document & Retrieve Screenshot Evidence</h3>
        <p className={`${darkMode ? 'text-slate-400' : 'text-slate-600'} text-xs`}>Answers and crops are returned only by the uploaded-document QA service.</p>
      </div>

      <div className={`flex items-center rounded-2xl border ${darkMode ? 'bg-slate-950/60 border-slate-700/60 text-white' : 'bg-slate-50 border-slate-300 text-slate-900'} p-2`}>
        <Search className="w-5 h-5 ml-3 text-slate-400 shrink-0" />
        <input value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => event.key === 'Enter' && handleAsk()} placeholder="Ask about the uploaded report..." className="w-full bg-transparent px-3 py-2 text-sm focus:outline-none" />
        <button onClick={() => handleAsk()} disabled={isAsking || !question.trim()} className="px-5 py-2.5 rounded-xl font-bold text-xs bg-emerald-500 hover:bg-emerald-600 text-slate-950 flex items-center gap-2 disabled:opacity-50">
          {isAsking ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />} Ask Document
        </button>
      </div>

      <div className="flex flex-wrap items-center gap-2 mt-4 mb-6">
        <span className="text-xs font-semibold text-slate-400 mr-1 flex items-center gap-1"><HelpCircle className="w-3.5 h-3.5" /> Sample questions:</span>
        {sampleQuestions.map((sample) => <button key={sample} onClick={() => handleAsk(sample)} className={`px-3 py-1.5 rounded-xl text-xs font-medium border ${darkMode ? 'bg-slate-800 border-slate-700 text-slate-300' : 'bg-slate-100 border-slate-300 text-slate-700'}`}>{sample}</button>)}
      </div>

      {isAsking && <div className="p-8 text-center rounded-2xl bg-emerald-500/5 border border-emerald-500/20"><Loader2 className="w-8 h-8 text-emerald-400 animate-spin mx-auto mb-2" /><p className="text-sm font-semibold text-emerald-400">Finding document evidence…</p></div>}
      {qaError && <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm">{qaError} Upload the report again after the backend is available; no client-side answer is shown because it could be inaccurate.</div>}

      {qaResult && !isAsking && <div className={`mt-4 p-6 rounded-2xl border ${darkMode ? 'bg-slate-950/80 border-slate-800' : 'bg-slate-50 border-slate-200'} space-y-5`}>
        <div className="flex items-start justify-between gap-4 pb-4 border-b border-slate-800/60">
          <div><div className="flex items-center gap-2"><span className="px-2.5 py-0.5 rounded-md bg-emerald-500/20 border border-emerald-500/40 text-emerald-400 text-xs font-extrabold uppercase">Q&A Result</span><span className="text-xs text-slate-400">Confidence: {(qaResult.confidence * 100).toFixed(0)}%</span></div><h4 className="text-lg font-extrabold text-slate-900 dark:text-white mt-1">“{qaResult.question}”</h4></div>
          <span className="px-3 py-1 rounded-xl bg-slate-800 border border-slate-700 text-slate-300 text-xs font-semibold flex items-center gap-1.5"><FileText className="w-3.5 h-3.5 text-emerald-400" /> Page {qaResult.page_number ?? '—'}</span>
        </div>
        <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 space-y-2"><div className="flex items-center gap-2 text-xs font-bold text-emerald-400"><CheckCircle2 className="w-4 h-4" /> Extracted Answer Text:</div><p className="text-sm font-medium leading-relaxed whitespace-pre-line text-slate-100">{qaResult.answer}</p></div>
        {qaResult.snippet_url && <div className="space-y-3"><div className="flex justify-between text-xs font-bold text-slate-300"><span className="flex items-center gap-2"><ImageIcon className="w-4 h-4 text-indigo-400" /> Visual evidence</span><span>{qaResult.section_title}</span></div><button onClick={() => setZoomImage(qaResult.snippet_url)} className="block w-full relative rounded-2xl overflow-hidden border-2 border-emerald-500/40 bg-slate-950"><img src={qaResult.snippet_url} alt={`Evidence for page ${qaResult.page_number}`} className="w-full max-h-[420px] object-contain mx-auto" /><span className="absolute bottom-3 right-3 px-3 py-1.5 rounded-lg bg-emerald-500 text-slate-950 text-xs font-bold flex items-center gap-1"><ExternalLink className="w-3.5 h-3.5" /> Fullscreen</span></button></div>}
      </div>}

      {zoomImage && <div className="fixed inset-0 z-50 bg-slate-950/90 flex items-center justify-center p-4"><div className="relative max-w-5xl w-full bg-slate-900 border border-slate-800 rounded-3xl p-4"><button onClick={() => setZoomImage(null)} className="absolute top-5 right-5 w-8 h-8 rounded-full bg-slate-800 text-white flex items-center justify-center"><X className="w-4 h-4" /></button><img src={zoomImage} alt="Fullscreen evidence" className="w-full max-h-[80vh] object-contain rounded-2xl" /></div></div>}
    </div>
  );
}
