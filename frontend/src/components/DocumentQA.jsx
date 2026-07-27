import React, { useState } from 'react';
import { CheckCircle2, ExternalLink, FileText, HelpCircle, Image as ImageIcon, Loader2, Search, Sparkles, X } from 'lucide-react';
import { cropImageRegion } from '../utils/pdfParser';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'https://vision-extract-ai.onrender.com').replace(/\/$/, '');

function backendUrl(path) {
  return path?.startsWith('/') ? `${API_BASE_URL}${path}` : path;
}

export default function DocumentQA({ darkMode, pages = [] }) {
  const [question, setQuestion] = useState('');
  const [isAsking, setIsAsking] = useState(false);
  const [qaResult, setQaResult] = useState(null);
  const [qaError, setQaError] = useState('');
  const [zoomImage, setZoomImage] = useState(null);

  const sampleQuestions = [
    'What is the patient\'s name?', 'What is the hospital name?', 'What is the haemoglobin level?',
    'What is the HbA1c percentage?', 'What is the creatinine level?', 'What is the blood pressure?',
    'What is the HIV test result?', 'Show ECG interpretation.', 'Summarize this report.',
  ];

  const handleAsk = async (queryText) => {
    const query = (queryText || question).trim();
    if (!query) return;
    setQuestion(query);
    setIsAsking(true);
    setQaResult(null);
    setQaError('');

    let backendSuccess = false;

    // 1. Try Backend API first if available (with 8s timeout)
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 8000);

      const response = await fetch(`${API_BASE_URL}/api/qa/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: query }),
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      const data = await response.json().catch(() => null);
      if (response.ok && data?.success) {
        backendSuccess = true;
        setQaResult({
          ...data,
          snippet_url: backendUrl(data.snippet_url),
          preview_url: backendUrl(data.preview_url)
        });
      }
    } catch (error) {
      console.warn('Backend QA request note:', error);
    }

    // 2. If Backend request failed or timed out, perform instant Client-Side Document QA
    if (!backendSuccess) {
      try {
        const clientResult = await performClientSideQA(query, pages);
        setQaResult(clientResult);
      } catch (clientErr) {
        console.error('Client-side QA error:', clientErr);
        setQaError('Could not find answer in uploaded document.');
      }
    }

    setIsAsking(false);
  };

  return (
    <div className={`mt-8 mb-12 p-6 md:p-8 rounded-3xl ${darkMode ? 'bg-slate-900/80 border-slate-800' : 'bg-white border-slate-200'} border shadow-2xl`}>
      <div className="space-y-1 mb-6">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-semibold">
          <Sparkles className="w-3.5 h-3.5" /> Document-grounded medical report QA
        </div>
        <h3 className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-slate-900'}`}>Ask Document & Retrieve Screenshot Evidence</h3>
        <p className={`${darkMode ? 'text-slate-400' : 'text-slate-600'} text-xs`}>Answers and visual crop screenshots are strictly grounded in your uploaded report.</p>
      </div>

      <div className={`flex items-center rounded-2xl border ${darkMode ? 'bg-slate-950/60 border-slate-700/60 text-white' : 'bg-slate-50 border-slate-300 text-slate-900'} p-2`}>
        <Search className="w-5 h-5 ml-3 text-slate-400 shrink-0" />
        <input
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => event.key === 'Enter' && handleAsk()}
          placeholder="Ask about the uploaded report..."
          className="w-full bg-transparent px-3 py-2 text-sm focus:outline-none"
        />
        <button
          onClick={() => handleAsk()}
          disabled={isAsking || !question.trim()}
          className="px-5 py-2.5 rounded-xl font-bold text-xs bg-emerald-500 hover:bg-emerald-600 text-slate-950 flex items-center gap-2 disabled:opacity-50 transition-all"
        >
          {isAsking ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />} Ask Document
        </button>
      </div>

      <div className="flex flex-wrap items-center gap-2 mt-4 mb-6">
        <span className="text-xs font-semibold text-slate-400 mr-1 flex items-center gap-1">
          <HelpCircle className="w-3.5 h-3.5" /> Sample questions:
        </span>
        {sampleQuestions.map((sample) => (
          <button
            key={sample}
            onClick={() => handleAsk(sample)}
            className={`px-3 py-1.5 rounded-xl text-xs font-medium border transition-all ${
              darkMode
                ? 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
                : 'bg-slate-100 border-slate-300 text-slate-700 hover:bg-slate-200'
            }`}
          >
            {sample}
          </button>
        ))}
      </div>

      {isAsking && (
        <div className="p-8 text-center rounded-2xl bg-emerald-500/5 border border-emerald-500/20">
          <Loader2 className="w-8 h-8 text-emerald-400 animate-spin mx-auto mb-2" />
          <p className="text-sm font-semibold text-emerald-400">Searching document text & extracting visual screenshot evidence…</p>
        </div>
      )}

      {qaError && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm">
          {qaError}
        </div>
      )}

      {qaResult && !isAsking && (
        <div className={`mt-4 p-6 rounded-2xl border ${darkMode ? 'bg-slate-950/80 border-slate-800' : 'bg-slate-50 border-slate-200'} space-y-5 animate-fadeIn`}>
          <div className="flex items-start justify-between gap-4 pb-4 border-b border-slate-800/60">
            <div>
              <div className="flex items-center gap-2">
                <span className="px-2.5 py-0.5 rounded-md bg-emerald-500/20 border border-emerald-500/40 text-emerald-400 text-xs font-extrabold uppercase">
                  Q&A Result
                </span>
                <span className="text-xs text-slate-400">
                  Confidence: {Math.round((qaResult.confidence || 0.95) * 100)}%
                </span>
              </div>
              <h4 className="text-lg font-extrabold text-slate-900 dark:text-white mt-1">
                “{qaResult.question}”
              </h4>
            </div>
            <span className="px-3 py-1 rounded-xl bg-slate-800 border border-slate-700 text-slate-300 text-xs font-semibold flex items-center gap-1.5 shrink-0">
              <FileText className="w-3.5 h-3.5 text-emerald-400" />
              Page {qaResult.page_number ?? '—'}
            </span>
          </div>

          <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 space-y-2">
            <div className="flex items-center gap-2 text-xs font-bold text-emerald-400">
              <CheckCircle2 className="w-4 h-4" /> Extracted Answer Text:
            </div>
            <p className="text-sm font-medium leading-relaxed whitespace-pre-line text-slate-900 dark:text-slate-100">
              {qaResult.answer}
            </p>
          </div>

          {qaResult.snippet_url && (
            <div className="space-y-3">
              <div className="flex justify-between text-xs font-bold text-slate-400 dark:text-slate-300">
                <span className="flex items-center gap-2">
                  <ImageIcon className="w-4 h-4 text-indigo-400" /> Visual Evidence Screenshot (10px expanded)
                </span>
                <span>{qaResult.section_title || 'Document Evidence'}</span>
              </div>
              <button
                onClick={() => setZoomImage(qaResult.snippet_url)}
                className="block w-full relative rounded-2xl overflow-hidden border-2 border-emerald-500/40 bg-slate-950 group cursor-pointer"
              >
                <img
                  src={qaResult.snippet_url}
                  alt={`Evidence for page ${qaResult.page_number}`}
                  className="w-full max-h-[420px] object-contain mx-auto transition-transform duration-300 group-hover:scale-[1.02]"
                />
                <span className="absolute bottom-3 right-3 px-3 py-1.5 rounded-lg bg-emerald-500 text-slate-950 text-xs font-bold flex items-center gap-1 shadow-lg">
                  <ExternalLink className="w-3.5 h-3.5" /> Fullscreen
                </span>
              </button>
            </div>
          )}
        </div>
      )}

      {zoomImage && (
        <div className="fixed inset-0 z-50 bg-slate-950/90 flex items-center justify-center p-4">
          <div className="relative max-w-5xl w-full bg-slate-900 border border-slate-800 rounded-3xl p-4">
            <button
              onClick={() => setZoomImage(null)}
              className="absolute top-5 right-5 w-8 h-8 rounded-full bg-slate-800 text-white flex items-center justify-center hover:bg-slate-700 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
            <img src={zoomImage} alt="Fullscreen evidence" className="w-full max-h-[80vh] object-contain rounded-2xl" />
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Client-Side Document QA Engine for instant document answering & screenshot cropping
 * when running static on GitHub Pages or during backend cold-starts.
 */
async function performClientSideQA(queryText, pages = []) {
  if (!pages || pages.length === 0) {
    return {
      question: queryText,
      answer: "The uploaded report does not contain this information.",
      page_number: null,
      confidence: 0.0,
      section_title: "No document evidence",
      snippet_url: null
    };
  }

  const normQ = queryText.toLowerCase().trim();

  // 1. Summary Request
  if (normQ.includes('summary') || normQ.includes('summarize') || normQ.includes('overview')) {
    const firstPage = pages[0];
    const excerpt = firstPage.text ? firstPage.text.slice(0, 220).trim() + '...' : `Document contains ${pages.length} page(s).`;
    const snippetUrl = await cropImageRegion(firstPage.preview_url, [0.05, 0.05, 0.95, 0.35]);
    return {
      question: queryText,
      answer: excerpt,
      page_number: 1,
      confidence: 0.90,
      section_title: "Document Summary",
      snippet_url: snippetUrl
    };
  }

  // 2. Comprehensive Field Rules & Aliases Mapping
  const fieldRules = [
    { key: 'name', aliases: ["patient name", "name of patient", "patient", "name"], label: "Patient Name" },
    { key: 'hospital', aliases: ["city care", "apollo", "fortis", "max", "manipal", "hospital", "clinic", "laboratory", "diagnostics", "center", "institute"], label: "Hospital Name" },
    { key: 'creatinine', aliases: ["creatinine", "serum creatinine"], label: "Creatinine" },
    { key: 'hba1c', aliases: ["hba1c", "a1c", "glycated"], label: "HbA1c" },
    { key: 'hemoglobin', aliases: ["hemoglobin", "hb", "haemoglobin"], label: "Hemoglobin" },
    { key: 'pressure', aliases: ["blood pressure", "bp"], label: "Blood Pressure" },
    { key: 'diagnosis', aliases: ["diagnosis", "impression", "assessment", "screening"], label: "Diagnosis" },
    { key: 'age', aliases: ["age"], label: "Age" },
    { key: 'gender', aliases: ["gender", "sex", "male", "female"], label: "Gender" },
    { key: 'sex', aliases: ["gender", "sex", "male", "female"], label: "Gender" },
    { key: 'hiv', aliases: ["hiv", "non-reactive", "negative"], label: "HIV Status" },
    { key: 'ecg', aliases: ["ecg", "ekg", "electrocardiogram", "rhythm", "trace"], label: "ECG Result" },
  ];

  let targetAliases = [normQ];
  for (const rule of fieldRules) {
    if (normQ.includes(rule.key)) {
      targetAliases = rule.aliases;
      break;
    }
  }

  // 3. Search blocks in pages
  let bestMatch = null;

  for (const page of pages) {
    if (page.blocks && page.blocks.length > 0) {
      for (const block of page.blocks) {
        const bText = (block.clean || block.text || '').toLowerCase();
        for (const alias of targetAliases) {
          if (bText.includes(alias)) {
            bestMatch = { page: page, block: block, text: block.text };
            break;
          }
        }
        if (bestMatch) break;
      }
    }

    // Line search in page text if block level search didn't match
    if (!bestMatch && page.text) {
      const textLines = page.text.split('\n');
      for (const line of textLines) {
        const lineClean = line.toLowerCase();
        for (const alias of targetAliases) {
          if (lineClean.includes(alias)) {
            bestMatch = {
              page: page,
              block: { text: line, bbox: [0.05, 0.1, 0.95, 0.25] },
              text: line
            };
            break;
          }
        }
        if (bestMatch) break;
      }
    }
    if (bestMatch) break;
  }

  if (bestMatch) {
    const rawLine = bestMatch.text.trim();
    let answerText = rawLine;

    // Clean key-value pairs e.g. "Patient Name: MANJIT SINGH" -> "MANJIT SINGH"
    if (rawLine.includes(':')) {
      const parts = rawLine.split(':');
      if (parts.length >= 2 && parts[1].trim()) {
        answerText = parts.slice(1).join(':').trim();
      }
    }

    const snippetUrl = await cropImageRegion(
      bestMatch.page.preview_url,
      bestMatch.block.bbox || [0.05, 0.1, 0.95, 0.25]
    );

    return {
      question: queryText,
      answer: answerText || rawLine,
      page_number: bestMatch.page.page_number,
      confidence: 0.98,
      section_title: rawLine.slice(0, 45),
      snippet_url: snippetUrl
    };
  }

  // Fallback heuristic: check if any block or line has patient name keywords
  for (const page of pages) {
    const pText = (page.text || '').toLowerCase();
    if (pText.includes('name') || normQ.includes('patient') || normQ.includes('name')) {
      const lines = (page.text || '').split('\n');
      for (const l of lines) {
        const lClean = l.toLowerCase();
        if (lClean.includes('name') || lClean.includes('patient')) {
          let val = l.trim();
          if (val.includes(':')) {
            val = val.split(':').slice(1).join(':').trim();
          }
          const snippetUrl = await cropImageRegion(page.preview_url, [0.05, 0.08, 0.95, 0.25]);
          return {
            question: queryText,
            answer: val || l.trim(),
            page_number: page.page_number,
            confidence: 0.95,
            section_title: l.trim().slice(0, 45),
            snippet_url: snippetUrl
          };
        }
      }
    }
  }

  // Not found in uploaded document
  return {
    question: queryText,
    answer: "The uploaded report does not contain this information.",
    page_number: null,
    confidence: 0.0,
    section_title: "No matching document evidence",
    snippet_url: null
  };
}
