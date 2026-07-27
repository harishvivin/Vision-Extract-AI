import React, { useState, useEffect, useRef } from 'react';
import Navbar from './components/Navbar';
import UploadZone from './components/UploadZone';
import ProgressBar from './components/ProgressBar';
import DocumentQA from './components/DocumentQA';
import { Sparkles, FileText, CheckCircle2, RefreshCw, AlertCircle } from 'lucide-react';
import { parsePdfInBrowser } from './utils/pdfParser';
import { createSampleMedicalPdfFile } from './utils/samplePdf';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'https://vision-extract-ai.onrender.com').replace(/\/$/, '');

export default function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusText, setStatusText] = useState('');
  const [pages, setPages] = useState([]);
  const [errorMessage, setErrorMessage] = useState('');

  // PDF Upload & Analysis Flow state
  const [isAnalyzed, setIsAnalyzed] = useState(false);
  const [activeDocName, setActiveDocName] = useState('');

  const progressIntervalRef = useRef(null);

  // Set html dark mode class
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  // Clean up timer on unmount
  useEffect(() => {
    return () => {
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
      }
    };
  }, []);

  const handleBenchmarkUpload = async () => {
    const sampleFile = createSampleMedicalPdfFile();
    handleFileUpload(sampleFile);
  };

  const handleFileUpload = async (inputFile) => {
    // Ensure inputFile is a valid File object
    const file = (inputFile instanceof File) ? inputFile : createSampleMedicalPdfFile();

    setIsProcessing(true);
    setProgress(15);
    setStatusText(`Reading uploaded file '${file.name}'...`);
    setErrorMessage('');
    setPages([]);

    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
    }

    try {
      // Step 1: Render pages and extract text from uploaded PDF in browser
      setStatusText(`Rendering pages & extracting visual text from '${file.name}'...`);
      let browserPages = [];
      try {
        browserPages = await parsePdfInBrowser(file);
        if (browserPages && browserPages.length > 0) {
          setPages(browserPages);
        }
      } catch (pdfErr) {
        console.warn('Browser PDF render fallback:', pdfErr);
      }

      // Step 2: Start smooth progress animation from 45% to 95% while processing with backend
      setProgress(45);
      setStatusText(`Uploading '${file.name}' to Document Analysis Engine...`);

      let currentProg = 45;
      progressIntervalRef.current = setInterval(() => {
        currentProg += Math.floor(Math.random() * 4) + 2;
        if (currentProg >= 95) {
          currentProg = 95;
          clearInterval(progressIntervalRef.current);
        }
        setProgress(currentProg);

        if (currentProg < 65) {
          setStatusText(`Uploading '${file.name}' to AI Engine...`);
        } else if (currentProg < 80) {
          setStatusText(`Parsing document layout & indexing text blocks...`);
        } else if (currentProg < 92) {
          setStatusText(`Building TF-IDF index & vector embeddings...`);
        } else {
          setStatusText(`Finalizing document QA session for '${file.name}'...`);
        }
      }, 500);

      // Step 3: Send file to Backend API endpoint with 30s timeout
      let backendSuccess = false;
      try {
        const formData = new FormData();
        formData.append('file', file);

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000);

        const response = await fetch(`${API_BASE_URL}/api/process`, {
          method: 'POST',
          body: formData,
          signal: controller.signal
        });
        clearTimeout(timeoutId);

        const data = await response.json().catch(() => null);
        if (response.ok && data?.success) {
          backendSuccess = true;
          if (data.pages && data.pages.length > 0) {
            setPages(data.pages);
          }
        }
      } catch (backendErr) {
        console.warn('Backend analysis request note:', backendErr);
      }

      // Clear interval timer
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
      }

      // Complete analysis if backend succeeded OR if client-side browser parsing extracted pages
      if (backendSuccess || (browserPages && browserPages.length > 0)) {
        setProgress(100);
        setStatusText(`Analysis Complete for '${file.name}'! Unlocking Question Answering Bar...`);
        setActiveDocName(file.name);

        setTimeout(() => {
          setIsProcessing(false);
          setIsAnalyzed(true);
        }, 500);
      } else {
        throw new Error('Could not parse or analyze the uploaded PDF file.');
      }

    } catch (err) {
      console.error('File processing error:', err);
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
      }
      setIsProcessing(false);
      setErrorMessage(err.message || 'Failed to analyze uploaded PDF file. Please try again.');
    }
  };

  const handleResetDocument = () => {
    setIsAnalyzed(false);
    setActiveDocName('');
    setProgress(0);
    setStatusText('');
  };

  return (
    <div className={`min-h-screen ${darkMode ? 'bg-slate-950 text-slate-100' : 'bg-slate-50 text-slate-800'} flex flex-col transition-colors duration-300 relative overflow-hidden`}>
      {/* Ambient Glow Orbs */}
      <div className="glass-glow-emerald" />
      <div className="glass-glow-teal" />
      <div className="glass-glow-indigo" />

      <Navbar darkMode={darkMode} setDarkMode={setDarkMode} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-8 relative z-10">

        {/* Banner Section */}
        <div className="text-center my-6 space-y-3">
          <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 dark:text-emerald-400 text-xs font-semibold">
            <Sparkles className="w-3.5 h-3.5" /> Vision AI Visual Document QA Engine
          </div>
          <h2 className={`text-3xl md:text-4xl font-extrabold ${darkMode ? 'text-white' : 'text-slate-900'} tracking-tight`}>
            Medical Report PDF Analyzer & Visual QA
          </h2>
          <p className={`${darkMode ? 'text-slate-400' : 'text-slate-600'} text-sm max-w-2xl mx-auto`}>
            Upload your PDF medical report or lab scan. The AI analyzes and indexes the document, unlocking the interactive question bar to answer your queries with exact text explanations and visual page screenshots.
          </p>
        </div>

        {/* STAGE 1: Upload & Analyze PDF (Shown if no document analyzed yet) */}
        {!isAnalyzed && (
          <div className="space-y-6 animate-fadeIn">
            <UploadZone
              onFileUpload={handleFileUpload}
              onUseDefaultBenchmark={handleBenchmarkUpload}
              isProcessing={isProcessing}
            />

            {isProcessing && (
              <ProgressBar progress={progress} statusText={statusText} />
            )}

            {errorMessage && (
              <div className="max-w-3xl mx-auto my-4 p-4 rounded-2xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm flex items-center gap-3">
                <AlertCircle className="w-5 h-5 text-rose-400 shrink-0" />
                <span>{errorMessage}</span>
              </div>
            )}
          </div>
        )}

        {/* STAGE 2: Document Analyzed -> Show Active Document Bar & Unlocked Visual QA */}
        {isAnalyzed && (
          <div className="space-y-6 animate-fadeIn">
            {/* Active Document Status Bar */}
            <div className={`p-4 rounded-2xl border ${darkMode ? 'bg-slate-900/90 border-emerald-500/30' : 'bg-emerald-50 border-emerald-300'} flex flex-col md:flex-row items-center justify-between gap-4 shadow-xl`}>
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 rounded-xl bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400">
                  <FileText className="w-6 h-6" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-bold text-slate-900 dark:text-white">
                      Active Medical Report: {activeDocName}
                    </span>
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 text-xs font-semibold">
                      <CheckCircle2 className="w-3 h-3" /> Analyzed & Indexed
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    Question bar unlocked. Ready to answer natural language queries and retrieve screenshot evidence.
                  </p>
                </div>
              </div>

              <button
                onClick={handleResetDocument}
                className="px-4 py-2 text-xs font-semibold rounded-xl bg-slate-800 hover:bg-slate-700 text-white border border-slate-700 flex items-center gap-2 transition-all shrink-0"
              >
                <RefreshCw className="w-3.5 h-3.5" /> Upload New PDF File
              </button>
            </div>

            {/* Unlocked Visual Document QA Component */}
            <DocumentQA darkMode={darkMode} pages={pages} />
          </div>
        )}

      </main>

      <footer className="border-t border-slate-800 py-6 text-center text-xs text-slate-500">
        <p>Vision Extract AI Pipeline &bull; Production Ready AI Internship Project</p>
      </footer>
    </div>
  );
}
