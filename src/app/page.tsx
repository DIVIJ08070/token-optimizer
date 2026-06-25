'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import {
  Upload,
  FileText,
  Send,
  Loader2,
  Bot,
  User,
  CheckCircle2,
  ClipboardList,
  ChevronRight,
  ThumbsUp,
  ThumbsDown,
  RefreshCw,
  Search,
  AlertCircle,
  MessageCircleQuestion,
  Trash2,
} from 'lucide-react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type AppStep = 'upload' | 'review' | 'chat';

interface FaqPair {
  id: string;
  question: string;
  rephrasings: string[];
  answer: string;
  source: string;
  chunk_ref: string;
  grounded_quote: string;
  status: 'pending' | 'approved' | 'rejected';
  indexed: boolean;
}

interface ChatMessage {
  role: 'user' | 'bot';
  text: string;
  sources?: { pdf: string; chunkRef: string }[];
  matchedQuestion?: string;
  suggestions?: string[];
  isDidYouMean?: boolean;
  isFallback?: boolean;
  apiCalled?: boolean;
  pairId?: string;
  feedback?: 'up' | 'down';
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function StepBadge({ step, current, label }: { step: AppStep; current: AppStep; label: string }) {
  const steps: AppStep[] = ['upload', 'review', 'chat'];
  const stepIdx    = steps.indexOf(step);
  const currentIdx = steps.indexOf(current);
  const done       = currentIdx > stepIdx;
  const active     = stepIdx === currentIdx;

  return (
    <div className={`flex items-center gap-2 text-xs font-semibold ${
      done ? 'text-emerald-400' : active ? 'text-indigo-300' : 'text-slate-600'
    }`}>
      <div className={`w-6 h-6 rounded-full flex items-center justify-center border ${
        done   ? 'bg-emerald-500/20 border-emerald-500/50' :
        active ? 'bg-indigo-600/30 border-indigo-500/60 ring-2 ring-indigo-500/30' :
                 'bg-slate-800 border-slate-700'
      }`}>
        {done ? <CheckCircle2 className="w-3.5 h-3.5" /> : <span>{stepIdx + 1}</span>}
      </div>
      {label}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Page
// ---------------------------------------------------------------------------

export default function Home() {
  const [appStep, setAppStep]     = useState<AppStep>('upload');
  const [files, setFiles]         = useState<File[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<any>(null);
  const [isClearingData, setIsClearingData] = useState(false);
  const [localStats, setLocalStats] = useState<{ total: number; indexed: number; approved: number } | null>(null);

  // Review state
  const [pairs, setPairs]               = useState<FaqPair[]>([]);
  const [isLoadingPairs, setIsLoadingPairs] = useState(false);
  const [isIndexing, setIsIndexing]     = useState(false);
  const [indexResult, setIndexResult]   = useState<any>(null);
  const [filterStatus, setFilterStatus] = useState<'all' | 'pending' | 'approved' | 'rejected'>('all');
  const [searchQuery, setSearchQuery]   = useState('');

  // Chat state
  const [question, setQuestion]     = useState('');
  const [chatState, setChatState]   = useState<'idle' | 'awaiting_lead'>('idle');
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [isAsking, setIsAsking]     = useState(false);

  // Missed Queries state
  const [missedQueries, setMissedQueries] = useState<any[]>([]);
  const [showMissed, setShowMissed] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatEndRef   = useRef<HTMLDivElement>(null);

  // Fetch local storage stats (total/indexed pairs) on mount and after actions
  const fetchLocalStats = useCallback(async () => {
    try {
      const res  = await fetch('/api/review');
      const data = await res.json();
      if (res.ok) {
        const all  = data.pairs ?? [];
        setLocalStats({
          total:    all.length,
          indexed:  all.filter((p: FaqPair) => p.indexed).length,
          approved: all.filter((p: FaqPair) => p.status === 'approved').length,
        });
      }
    } catch { /* silent */ }
  }, []);

  useEffect(() => { fetchLocalStats(); }, [fetchLocalStats]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  // ---------------------------------------------------------------------------
  // Upload
  // ---------------------------------------------------------------------------

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selected = Array.from(e.target.files);
      if (selected.length > 50) { alert('Maximum 50 PDFs allowed'); return; }
      setFiles(selected);
    }
  };

  const handleUpload = async () => {
    if (files.length === 0) return;
    setIsUploading(true);

    const formData = new FormData();
    files.forEach(f => formData.append('files', f));

    try {
      const res  = await fetch('/api/upload', { method: 'POST', body: formData });
      const data = await res.json();
      if (res.ok) {
        setUploadResult(data);
        setAppStep('review');
        await fetchPairs();
      } else {
        alert(data.error || 'Upload failed');
      }
    } catch {
      alert('Error uploading files');
    } finally {
      setIsUploading(false);
    }
  };

  const handleClearData = async () => {
    if (!confirm('This will permanently delete all uploaded PDFs, Q&A pairs, and the chat index. Are you sure?')) return;
    setIsClearingData(true);
    try {
      const res = await fetch('/api/upload/clear', { method: 'DELETE' });
      if (res.ok) {
        // Reset all frontend state
        setAppStep('upload');
        setFiles([]);
        setUploadResult(null);
        setPairs([]);
        setIndexResult(null);
        setChatHistory([]);
        setQuestion('');
        setLocalStats({ total: 0, indexed: 0, approved: 0 });
      } else {
        const data = await res.json();
        alert(data.error || 'Failed to clear data');
      }
    } catch {
      alert('Error clearing data');
    } finally {
      setIsClearingData(false);
    }
  };

  // ---------------------------------------------------------------------------
  // Review
  // ---------------------------------------------------------------------------

  const fetchPairs = useCallback(async () => {
    setIsLoadingPairs(true);
    try {
      const res  = await fetch('/api/review');
      const data = await res.json();
      if (res.ok) setPairs(data.pairs ?? []);
    } catch {
      console.error('Failed to load pairs');
    } finally {
      setIsLoadingPairs(false);
    }
  }, []);

  useEffect(() => {
    if (appStep === 'review') {
      fetchPairs();
      fetchMissedQueries();
    }
  }, [appStep, fetchPairs]);

  const fetchMissedQueries = async () => {
    try {
      const res = await fetch('/api/missed-queries');
      const data = await res.json();
      if (res.ok) setMissedQueries(data.misses ?? []);
    } catch {
      console.error('Failed to load missed queries');
    }
  };

  const handleClearMissed = async () => {
    if (!confirm('Clear all missed queries?')) return;
    await fetch('/api/missed-queries', { method: 'DELETE' });
    setMissedQueries([]);
  };

  const reviewAction = async (action: string, ids: string[], pairUpdate?: Partial<FaqPair>) => {
    const body: any = { action, ids };
    if (pairUpdate) body.pair = pairUpdate;

    const res = await fetch('/api/review', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });
    if (res.ok) await fetchPairs();
  };

  const handleApproveAll = () => reviewAction('approve_all', []);

  const handleIndex = async () => {
    setIsIndexing(true);
    try {
      const res  = await fetch('/api/review/index', { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        setIndexResult(data);
        if (data.chatReady) setAppStep('chat');
      } else {
        alert(data.error || 'Indexing failed');
      }
    } catch {
      alert('Indexing error');
    } finally {
      setIsIndexing(false);
    }
  };

  // ---------------------------------------------------------------------------
  // Chat
  // ---------------------------------------------------------------------------

  const handleAsk = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    const q = question;
    setQuestion('');
    setChatHistory(prev => [...prev, { role: 'user', text: q }]);
    setIsAsking(true);

    try {
      const res  = await fetch('/api/chat', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ question: q, chatState }),
      });
      const data = await res.json();
      
      if (data.nextState) {
        setChatState(data.nextState);
      }

      setChatHistory(prev => [...prev, {
        role: 'bot',
        text: data.answer || data.error || 'Unknown error.',
        sources: data.sources,
        matchedQuestion: data.matchedQuestion,
        suggestions: data.suggestions,
        isDidYouMean: data.isDidYouMean,
        isFallback: data.isFallback,
        apiCalled: data.apiCalled,
        pairId: data.pairId,
      }]);
    } catch {
      setChatHistory(prev => [...prev, { role: 'bot', text: 'Failed to connect to the server.' }]);
    } finally {
      setIsAsking(false);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setQuestion(suggestion);
    // Use setTimeout so the state updates before submission
    setTimeout(() => {
      document.getElementById('chat-send-btn')?.click();
    }, 0);
  };

  const handleFeedback = async (msgIndex: number, feedback: 'up' | 'down') => {
    const msg = chatHistory[msgIndex];
    if (!msg.pairId || msg.feedback) return; // Already voted or no pairId

    // Optimistically update UI
    setChatHistory(prev => {
      const next = [...prev];
      next[msgIndex] = { ...next[msgIndex], feedback };
      return next;
    });

    try {
      await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pairId: msg.pairId, feedback }),
      });
      // Silent success
    } catch (e) {
      console.error('Failed to submit feedback', e);
    }
  };

  // ---------------------------------------------------------------------------
  // Derived
  // ---------------------------------------------------------------------------

  const approvedCount = pairs.filter(p => p.status === 'approved').length;
  const pendingCount  = pairs.filter(p => p.status === 'pending').length;
  const indexedCount  = pairs.filter(p => p.indexed).length;

  const filteredPairs = pairs
    .filter(p => filterStatus === 'all' || p.status === filterStatus)
    .filter(p => !searchQuery ||
      p.question.toLowerCase().includes(searchQuery.toLowerCase()) ||
      p.answer.toLowerCase().includes(searchQuery.toLowerCase())
    );

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-indigo-500/30">
      <div className="max-w-7xl mx-auto p-4 md:p-6 grid grid-cols-1 lg:grid-cols-3 gap-6 min-h-screen pt-8 pb-8">

        {/* ============================================================
            SIDEBAR
        ============================================================ */}
        <div className="lg:col-span-1 space-y-4 flex flex-col">
          <div className="bg-slate-900/50 p-5 rounded-2xl border border-slate-800 shadow-xl backdrop-blur-sm">
            <h1 className="text-2xl font-extrabold bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent mb-1">
              DocuFAQ
            </h1>
            <p className="text-slate-500 text-xs mb-5">
              AI-generated FAQ chatbot. Answers only approved content.
            </p>

            {/* Step indicators */}
            <div className="flex gap-3 mb-5">
              <StepBadge step="upload" current={appStep} label="Upload" />
              <ChevronRight className="w-3 h-3 text-slate-600 mt-1.5" />
              <StepBadge step="review" current={appStep} label="Review" />
              <ChevronRight className="w-3 h-3 text-slate-600 mt-1.5" />
              <StepBadge step="chat"   current={appStep} label="Chat"   />
            </div>

            {/* --- UPLOAD PANEL --- */}
            {appStep === 'upload' && (
              <div className="space-y-4">

                {/* Local Storage Stats Card — always visible */}
                {localStats !== null && (
                  <div className="p-3.5 bg-slate-800/60 rounded-xl border border-slate-700/60 space-y-3">
                    <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Local Storage</p>
                    <div className="grid grid-cols-3 gap-2">
                      {[
                        { label: 'Stored',   value: localStats.total,    color: 'text-slate-200' },
                        { label: 'Approved', value: localStats.approved,  color: 'text-emerald-400' },
                        { label: 'Indexed',  value: localStats.indexed,   color: 'text-indigo-400' },
                      ].map(s => (
                        <div key={s.label} className="bg-slate-900/50 rounded-lg p-2 text-center border border-slate-700/40">
                          <p className={`text-base font-bold ${s.color}`}>{s.value}</p>
                          <p className="text-[10px] text-slate-500 mt-0.5">{s.label}</p>
                        </div>
                      ))}
                    </div>
                    {localStats.total > 0 && (
                      <button
                        id="clear-data-btn-upload"
                        onClick={handleClearData}
                        disabled={isClearingData}
                        className="w-full flex items-center justify-center gap-2 text-xs font-semibold text-red-400/80 hover:text-red-300 hover:bg-red-900/20 border border-red-900/30 hover:border-red-700/50 py-2 rounded-lg transition-all disabled:opacity-50"
                      >
                        {isClearingData
                          ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
                          : <Trash2 className="w-3.5 h-3.5" />}
                        {isClearingData ? 'Clearing…' : 'Clear All Data'}
                      </button>
                    )}
                  </div>
                )}

                <div
                  className="border-2 border-dashed border-slate-700/80 bg-slate-900/30 rounded-xl p-8 text-center hover:border-indigo-500 hover:bg-slate-800/50 transition-all cursor-pointer group"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input
                    type="file"
                    multiple
                    accept="application/pdf"
                    className="hidden"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                  />
                  <div className="bg-slate-800 rounded-full w-14 h-14 flex items-center justify-center mx-auto mb-3 group-hover:scale-110 transition-transform">
                    <Upload className="w-7 h-7 text-indigo-400" />
                  </div>
                  <p className="text-sm text-slate-300 font-medium">Click to select PDFs</p>
                  <p className="text-xs text-slate-500 mt-1">Max 50 documents</p>
                </div>

                {files.length > 0 && (
                  <div className="space-y-2">
                    {files.map((f, i) => (
                      <div key={i} className="flex items-center gap-2 p-2.5 bg-slate-800/50 rounded-lg border border-slate-700/50">
                        <FileText className="w-4 h-4 text-indigo-400 shrink-0" />
                        <span className="text-xs truncate flex-1 text-slate-300">{f.name}</span>
                      </div>
                    ))}

                    <button
                      id="upload-btn"
                      onClick={handleUpload}
                      disabled={isUploading}
                      className="w-full mt-2 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold py-3 rounded-lg flex items-center justify-center gap-2 transition-all shadow-lg shadow-indigo-900/20 disabled:opacity-50"
                    >
                      {isUploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
                      {isUploading ? 'Generating FAQs with AI…' : 'Process & Generate FAQs'}
                    </button>

                    {isUploading && (
                      <p className="text-xs text-slate-400 text-center mt-1 animate-pulse">
                        OpenAI is reading each chunk and writing Q&A pairs…
                      </p>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* --- REVIEW PANEL (sidebar summary) --- */}
            {(appStep === 'review' || appStep === 'chat') && (
              <div className="space-y-3">
                {uploadResult && (
                  <div className="p-3 bg-slate-800/50 rounded-lg border border-slate-700 text-xs text-slate-400 space-y-1">
                    <p><span className="text-slate-200 font-medium">{uploadResult.pairsStored}</span> pairs generated</p>
                    {uploadResult.pairsFailedCheck > 0 && (
                      <p className="text-amber-400/80">{uploadResult.pairsFailedCheck} pairs failed grounding check</p>
                    )}
                  </div>
                )}

                <div className="grid grid-cols-3 gap-2">
                  {[
                    { label: 'Total',    value: pairs.length,   color: 'text-slate-300' },
                    { label: 'Approved', value: approvedCount,  color: 'text-emerald-400' },
                    { label: 'Missed Qs',  value: missedQueries.length,   color: 'text-amber-400' },
                  ].map(s => (
                    <div key={s.label} className="bg-slate-800/50 rounded-lg p-2.5 text-center border border-slate-700/50">
                      <p className={`text-lg font-bold ${s.color}`}>{s.value}</p>
                      <p className="text-xs text-slate-500">{s.label}</p>
                    </div>
                  ))}
                </div>

                {appStep === 'review' && (
                  <>
                    <button
                      id="approve-all-btn"
                      onClick={handleApproveAll}
                      disabled={pendingCount === 0}
                      className="w-full bg-emerald-700/60 hover:bg-emerald-600/60 text-emerald-200 font-semibold py-2.5 rounded-lg flex items-center justify-center gap-2 transition-all text-sm disabled:opacity-40"
                    >
                      <ThumbsUp className="w-4 h-4" />
                      Approve All ({pendingCount} pending)
                    </button>

                    <button
                      id="index-btn"
                      onClick={handleIndex}
                      disabled={isIndexing || approvedCount === 0}
                      className="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-semibold py-2.5 rounded-lg flex items-center justify-center gap-2 transition-all text-sm disabled:opacity-50"
                    >
                      {isIndexing ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
                      {isIndexing ? 'Embedding locally…' : `Index ${approvedCount} Approved Pairs`}
                    </button>

                    {isIndexing && (
                      <p className="text-xs text-slate-400 text-center animate-pulse">
                        Running local bge-small embeddings (no API calls)…
                      </p>
                    )}

                    {missedQueries.length > 0 && (
                      <button
                        onClick={() => setShowMissed(!showMissed)}
                        className={`w-full font-semibold py-2.5 rounded-lg flex items-center justify-center gap-2 transition-all text-sm border ${
                          showMissed 
                            ? 'bg-amber-500/20 text-amber-300 border-amber-500/30' 
                            : 'bg-amber-900/10 hover:bg-amber-900/20 text-amber-500/80 border-amber-900/30'
                        }`}
                      >
                        <MessageCircleQuestion className="w-4 h-4" />
                        {showMissed ? 'Hide Missed Queries' : `View Missed Queries (${missedQueries.length})`}
                      </button>
                    )}
                  </>
                )}

                {appStep === 'chat' && (
                  <div className="p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg text-center">
                    <CheckCircle2 className="w-6 h-6 text-emerald-400 mx-auto mb-1" />
                    <p className="text-sm font-semibold text-emerald-300">Chat Ready</p>
                    <p className="text-xs text-emerald-400/70 mt-0.5">{indexedCount} pairs indexed locally</p>
                  </div>
                )}

                <button
                  onClick={() => { setAppStep('upload'); setFiles([]); setUploadResult(null); }}
                  className="w-full text-xs text-slate-500 hover:text-slate-300 py-1 transition-colors"
                >
                  ← Upload more PDFs
                </button>

                <button
                  id="clear-data-btn"
                  onClick={handleClearData}
                  disabled={isClearingData}
                  className="w-full mt-1 flex items-center justify-center gap-2 text-xs font-semibold text-red-400/80 hover:text-red-300 hover:bg-red-900/20 border border-red-900/30 hover:border-red-700/50 py-2.5 rounded-lg transition-all disabled:opacity-50"
                >
                  {isClearingData
                    ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    : <Trash2 className="w-3.5 h-3.5" />}
                  {isClearingData ? 'Clearing…' : 'Clear All Data'}
                </button>
              </div>
            )}
          </div>
        </div>

        {/* ============================================================
            MAIN PANEL — switches between Review and Chat
        ============================================================ */}
        <div className="lg:col-span-2 flex flex-col">

          {/* ---- REVIEW VIEW ---- */}
          {(appStep === 'upload' || appStep === 'review') && (
            <div className="flex flex-col h-full bg-slate-900/50 border border-slate-800 rounded-2xl shadow-xl backdrop-blur-sm overflow-hidden">

              {/* Header */}
              <div className="p-5 border-b border-slate-800 flex items-center justify-between gap-4 shrink-0">
                <div className="flex items-center gap-2">
                  <ClipboardList className="w-5 h-5 text-indigo-400" />
                  <h2 className="font-semibold text-slate-200">
                    {appStep === 'upload' ? 'Awaiting Upload' : 'Review Q&A Pairs'}
                  </h2>
                </div>
                {appStep === 'review' && (
                  <button onClick={fetchPairs} className="text-slate-500 hover:text-slate-300 transition-colors">
                    <RefreshCw className="w-4 h-4" />
                  </button>
                )}
              </div>

              {appStep === 'upload' && (
                <div className="flex-1 flex flex-col items-center justify-center text-slate-500 p-8 text-center">
                  <div className="bg-slate-800/50 p-5 rounded-full mb-4 border border-slate-700/50">
                    <ClipboardList className="w-10 h-10 text-slate-600" />
                  </div>
                  <h3 className="text-lg text-slate-300 font-medium mb-2">Upload PDFs to begin</h3>
                  <p className="text-sm max-w-sm mb-6">
                    After uploading, AI will generate Q&A pairs from your documents.
                    You'll review and approve them here before the chatbot goes live.
                  </p>

                  {localStats && localStats.total > 0 && (
                    <div className="flex flex-col gap-3 items-center">
                      <div className="text-xs text-emerald-400/80 bg-emerald-500/10 px-4 py-2 rounded-full border border-emerald-500/20">
                        Found {localStats.total} existing pairs in local storage!
                      </div>
                      <div className="flex gap-3">
                        <button
                          onClick={() => { setAppStep('review'); fetchPairs(); }}
                          className="px-5 py-2.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 text-sm font-medium rounded-lg transition-colors shadow-sm"
                        >
                          Proceed to Review
                        </button>
                        {localStats.indexed > 0 && (
                          <button
                            onClick={() => setAppStep('chat')}
                            className="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg transition-colors flex items-center gap-2 shadow-sm shadow-indigo-900/50"
                          >
                            Jump to Chat <ChevronRight className="w-4 h-4" />
                          </button>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {appStep === 'review' && (
                <>
                  {/* Filter + search bar */}
                  <div className="p-4 border-b border-slate-800/50 flex gap-3 shrink-0">
                    <div className="relative flex-1">
                      <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                      <input
                        type="text"
                        placeholder="Search questions or answers…"
                        value={searchQuery}
                        onChange={e => setSearchQuery(e.target.value)}
                        className="w-full bg-slate-950/50 border border-slate-700 rounded-lg py-2 pl-9 pr-3 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500/50 placeholder:text-slate-600"
                      />
                    </div>
                    <select
                      value={filterStatus}
                      onChange={e => setFilterStatus(e.target.value as any)}
                      className="bg-slate-800 border border-slate-700 text-slate-300 rounded-lg px-3 py-2 text-sm focus:outline-none"
                    >
                      <option value="all">All ({pairs.length})</option>
                      <option value="pending">Pending ({pairs.filter(p=>p.status==='pending').length})</option>
                      <option value="approved">Approved ({approvedCount})</option>
                      <option value="rejected">Rejected ({pairs.filter(p=>p.status==='rejected').length})</option>
                    </select>
                  </div>

                  {/* Pairs list */}
                  <div className="flex-1 overflow-y-auto p-4 space-y-3 custom-scrollbar">
                    {isLoadingPairs && (
                      <div className="flex items-center justify-center py-12 text-slate-500 gap-2">
                        <Loader2 className="w-5 h-5 animate-spin" />
                        Loading pairs…
                      </div>
                    )}

                    {!isLoadingPairs && filteredPairs.length === 0 && (
                      <div className="flex flex-col items-center justify-center py-12 text-slate-500 text-center">
                        <AlertCircle className="w-8 h-8 mb-2 text-slate-600" />
                        <p className="text-sm">No pairs match this filter.</p>
                        {pairs.length === 0 && <p className="text-xs mt-1 text-slate-600">Upload PDFs to generate Q&A pairs.</p>}
                      </div>
                    )}

                    {filteredPairs.map(pair => (
                      <PairCard
                        key={pair.id}
                        pair={pair}
                        onApprove={() => reviewAction('approve', [pair.id])}
                        onReject={() => reviewAction('reject', [pair.id])}
                      />
                    ))}
                  </div>

                  {/* Missed Queries Drawer (overlaid if toggled) */}
                  {showMissed && (
                    <div className="absolute inset-0 bg-slate-900 z-10 flex flex-col border-t border-slate-800">
                      <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-800/50">
                        <div className="flex items-center gap-2">
                          <AlertCircle className="w-4 h-4 text-amber-400" />
                          <h3 className="font-semibold text-slate-200 text-sm">Queries That Needed Fallbacks</h3>
                        </div>
                        <button onClick={() => setShowMissed(false)} className="text-xs bg-slate-700 hover:bg-slate-600 px-2 py-1 rounded">Close</button>
                      </div>
                      <div className="flex-1 overflow-y-auto p-4 space-y-3 custom-scrollbar">
                        <p className="text-xs text-slate-400 mb-3">Add these to your FAQs so the bot can answer them next time.</p>
                        {missedQueries.map((m, i) => (
                          <div key={i} className="bg-slate-800/50 border border-amber-900/30 rounded-lg p-3">
                            <p className="text-sm font-medium text-slate-200 mb-1">"{m.query}"</p>
                            <p className="text-xs text-slate-500">Closest: "{m.closestMatch}" (Score: {m.score})</p>
                          </div>
                        ))}
                        <button onClick={handleClearMissed} className="w-full mt-4 text-xs text-red-400 border border-red-900/30 rounded py-2 hover:bg-red-900/10">Clear Log</button>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {/* ---- CHAT VIEW ---- */}
          {appStep === 'chat' && (
            <div className="flex flex-col h-[calc(100vh-8rem)] bg-slate-900/50 border border-slate-800 rounded-2xl shadow-xl backdrop-blur-sm overflow-hidden">

              {/* Header */}
              <div className="p-4 border-b border-slate-800 flex items-center gap-2 shrink-0">
                <Bot className="w-5 h-5 text-indigo-400" />
                <h2 className="font-semibold text-slate-200">FAQ Chat</h2>
                <span className="ml-auto text-xs text-emerald-400/80 bg-emerald-500/10 border border-emerald-500/20 rounded-full px-2.5 py-0.5">
                  Local-only · No API calls
                </span>
              </div>

              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-5 space-y-5 custom-scrollbar">
                {chatHistory.length === 0 && (
                  <div className="h-full flex flex-col items-center justify-center text-slate-500 opacity-60">
                    <MessageCircleQuestion className="w-14 h-14 mb-3 text-slate-600" />
                    <p className="text-base">Ask anything covered in your approved FAQs.</p>
                    <p className="text-xs mt-1">Out-of-scope questions will be declined honestly.</p>
                  </div>
                )}

                {chatHistory.map((msg, i) => (
                  <div key={i} className={`flex gap-3 animate-in fade-in slide-in-from-bottom-2 duration-300 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    {msg.role === 'bot' && (
                      <div className="w-9 h-9 rounded-full bg-indigo-600/20 flex items-center justify-center border border-indigo-500/30 shrink-0">
                        <Bot className="w-4.5 h-4.5 text-indigo-400" />
                      </div>
                    )}

                    <div className={`max-w-[82%] rounded-2xl p-4 shadow-sm ${
                      msg.role === 'user'
                        ? 'bg-indigo-600 text-white rounded-tr-sm'
                        : 'bg-slate-800/80 border border-slate-700 text-slate-200 rounded-tl-sm'
                    }`}>
                      <p className="whitespace-pre-wrap leading-relaxed text-[14px]">{msg.text}</p>

                      {msg.matchedQuestion && (
                        <p className="text-xs text-slate-400 mt-2 pt-2 border-t border-slate-700/50">
                          Matched: <span className="italic text-slate-300">"{msg.matchedQuestion}"</span>
                        </p>
                      )}

                      {msg.apiCalled && (
                        <div className="mt-2 pt-2 border-t border-slate-700/50 flex items-center justify-between">
                          <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md bg-indigo-500/10 border border-indigo-500/20 text-[11px] text-indigo-300 font-medium">
                            Auto-generated by AI
                          </span>
                          {msg.pairId && (
                            <div className="flex items-center gap-1">
                              <button
                                onClick={() => handleFeedback(i, 'up')}
                                disabled={!!msg.feedback}
                                className={`p-1.5 rounded-md transition-colors ${msg.feedback === 'up' ? 'text-emerald-400 bg-emerald-500/20' : 'text-slate-400 hover:text-emerald-400 hover:bg-emerald-500/10 disabled:opacity-50'}`}
                                title="Good answer"
                              >
                                <ThumbsUp className="w-3.5 h-3.5" />
                              </button>
                              <button
                                onClick={() => handleFeedback(i, 'down')}
                                disabled={!!msg.feedback}
                                className={`p-1.5 rounded-md transition-colors ${msg.feedback === 'down' ? 'text-red-400 bg-red-500/20' : 'text-slate-400 hover:text-red-400 hover:bg-red-500/10 disabled:opacity-50'}`}
                                title="Bad answer (needs review)"
                              >
                                <ThumbsDown className="w-3.5 h-3.5" />
                              </button>
                            </div>
                          )}
                        </div>
                      )}

                      {msg.sources && msg.sources.length > 0 && (
                        <div className="mt-3 pt-3 border-t border-slate-700/60">
                          <p className="text-xs font-semibold text-slate-400 mb-2 uppercase tracking-wider flex items-center gap-1.5">
                            <FileText className="w-3 h-3" /> Source
                          </p>
                          <div className="flex flex-wrap gap-1.5">
                            {msg.sources.map((s, idx) => (
                              <span key={idx} className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-slate-950/50 border border-slate-700 text-xs text-indigo-300 font-medium">
                                {s.pdf}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Clickable Suggestions (Did You Mean? / Fallback Menu) */}
                      {msg.suggestions && msg.suggestions.length > 0 && (
                        <div className="mt-3 space-y-2">
                          {msg.suggestions.map((sug, idx) => (
                            <button
                              key={idx}
                              onClick={() => handleSuggestionClick(sug)}
                              className="block w-full text-left bg-slate-900/50 hover:bg-indigo-600/20 border border-slate-700 hover:border-indigo-500/50 rounded-lg p-2.5 text-sm text-indigo-300 transition-colors"
                            >
                              <span className="text-xs text-slate-500 mr-2">↳</span>
                              {sug}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>

                    {msg.role === 'user' && (
                      <div className="w-9 h-9 rounded-full bg-slate-700 flex items-center justify-center shrink-0 border border-slate-600">
                        <User className="w-4 h-4 text-slate-300" />
                      </div>
                    )}
                  </div>
                ))}

                {isAsking && (
                  <div className="flex gap-3 justify-start">
                    <div className="w-9 h-9 rounded-full bg-indigo-600/20 flex items-center justify-center border border-indigo-500/30 shrink-0">
                      <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />
                    </div>
                    <div className="bg-slate-800/80 border border-slate-700 rounded-2xl rounded-tl-sm p-4 flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-slate-500 animate-bounce" />
                      <div className="w-2 h-2 rounded-full bg-slate-500 animate-bounce delay-75" />
                      <div className="w-2 h-2 rounded-full bg-slate-500 animate-bounce delay-150" />
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              {/* Input */}
              <div className="p-4 border-t border-slate-800 bg-slate-900/90 backdrop-blur-md shrink-0">
                <form onSubmit={handleAsk} className="relative flex items-center">
                  <input
                    id="chat-input"
                    type="text"
                    value={question}
                    onChange={e => setQuestion(e.target.value)}
                    placeholder="Ask a question about your approved FAQs…"
                    className="w-full bg-slate-950/50 border border-slate-700 rounded-xl py-3.5 pl-4 pr-12 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition-all placeholder:text-slate-500"
                    disabled={isAsking}
                  />
                  <button
                    type="submit"
                    id="chat-send-btn"
                    disabled={isAsking || !question.trim()}
                    className="absolute right-2 p-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg transition-colors disabled:opacity-40"
                  >
                    <Send className="w-4 h-4" />
                  </button>
                </form>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// PairCard component
// ---------------------------------------------------------------------------

function PairCard({
  pair,
  onApprove,
  onReject,
}: {
  pair: FaqPair;
  onApprove: () => void;
  onReject:  () => void;
}) {
  const [expanded, setExpanded] = useState(false);

  const statusColors: Record<string, string> = {
    pending:  'text-amber-400 bg-amber-400/10 border-amber-400/20',
    approved: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/20',
    rejected: 'text-red-400 bg-red-400/10 border-red-400/20',
  };

  return (
    <div className={`rounded-xl border p-4 transition-all ${
      pair.status === 'approved' ? 'border-emerald-800/40 bg-emerald-950/20' :
      pair.status === 'rejected' ? 'border-red-900/40 bg-red-950/10 opacity-60' :
      'border-slate-700/60 bg-slate-800/30'
    }`}>
      {/* Header row */}
      <div className="flex items-start justify-between gap-3 mb-2">
        <button
          onClick={() => setExpanded(e => !e)}
          className="flex-1 text-left text-sm font-medium text-slate-200 hover:text-white leading-snug"
        >
          {pair.question}
        </button>
        <div className="flex items-center gap-1.5 shrink-0">
          <span className={`text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full border ${statusColors[pair.status]}`}>
            {pair.status}
          </span>
          {pair.status !== 'approved' && (
            <button
              id={`approve-${pair.id}`}
              onClick={onApprove}
              title="Approve"
              className="p-1.5 rounded-lg bg-emerald-700/30 hover:bg-emerald-600/40 text-emerald-400 transition-colors"
            >
              <ThumbsUp className="w-3.5 h-3.5" />
            </button>
          )}
          {pair.status !== 'rejected' && (
            <button
              id={`reject-${pair.id}`}
              onClick={onReject}
              title="Reject"
              className="p-1.5 rounded-lg bg-red-700/20 hover:bg-red-700/30 text-red-400 transition-colors"
            >
              <ThumbsDown className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Answer preview */}
      <p className="text-xs text-slate-400 line-clamp-2">{pair.answer}</p>

      {/* Expanded details */}
      {expanded && (
        <div className="mt-3 space-y-2 border-t border-slate-700/40 pt-3">
          <div>
            <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-1 font-semibold">Full Answer</p>
            <p className="text-xs text-slate-300 leading-relaxed">{pair.answer}</p>
          </div>

          {pair.rephrasings.length > 0 && (
            <div>
              <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-1 font-semibold">Rephrasings ({pair.rephrasings.length})</p>
              <ul className="text-xs text-slate-400 list-disc list-inside space-y-0.5">
                {pair.rephrasings.map((r, i) => <li key={i}>{r}</li>)}
              </ul>
            </div>
          )}

          {pair.grounded_quote && (
            <div>
              <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-1 font-semibold">Grounding Quote</p>
              <blockquote className="text-xs text-slate-400 italic border-l-2 border-slate-600 pl-3">
                "{pair.grounded_quote}"
              </blockquote>
            </div>
          )}

          <p className="text-[10px] text-slate-600">Source: {pair.source} · Chunk {pair.chunk_ref}</p>
        </div>
      )}

      {!expanded && (
        <button
          onClick={() => setExpanded(true)}
          className="text-[10px] text-slate-600 hover:text-slate-400 mt-1 transition-colors"
        >
          Show details ▾
        </button>
      )}
    </div>
  );
}
