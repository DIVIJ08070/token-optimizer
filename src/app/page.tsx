'use client';

import { useState, useRef, useEffect } from 'react';
import { Upload, FileText, Send, Loader2, Bot, User, CheckCircle2 } from 'lucide-react';

export default function Home() {
  const [files, setFiles] = useState<File[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isUploaded, setIsUploaded] = useState(false);

  const [question, setQuestion] = useState('');
  const [chatHistory, setChatHistory] = useState<{ role: 'user' | 'bot', text: string, sources?: any[] }[]>([]);
  const [isAsking, setIsAsking] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selected = Array.from(e.target.files);
      if (selected.length > 2) {
        alert('Maximum 2 PDFs allowed');
        return;
      }
      setFiles(selected);
    }
  };

  const handleUpload = async () => {
    if (files.length === 0) return;
    setIsUploading(true);

    const formData = new FormData();
    files.forEach(f => formData.append('files', f));

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setIsUploaded(true);
      } else {
        alert(data.error || 'Upload failed');
      }
    } catch (err) {
      alert('Error uploading files');
    } finally {
      setIsUploading(false);
    }
  };

  const handleAsk = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    const q = question;
    setQuestion('');
    setChatHistory(prev => [...prev, { role: 'user', text: q }]);
    setIsAsking(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q }),
      });
      const data = await res.json();

      setChatHistory(prev => [...prev, {
        role: 'bot',
        text: data.answer || data.error || 'Unknown error occurred.',
        sources: data.sources
      }]);
    } catch (err) {
      setChatHistory(prev => [...prev, { role: 'bot', text: 'Failed to connect to the server.' }]);
    } finally {
      setIsAsking(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-indigo-500/30">
      <div className="max-w-6xl mx-auto p-4 md:p-8 grid grid-cols-1 lg:grid-cols-3 gap-8 h-screen pt-12 pb-12">

        {/* Sidebar - Upload Section */}
        <div className="lg:col-span-1 space-y-6 flex flex-col">
          <div className="bg-slate-900/50 p-6 rounded-2xl border border-slate-800 shadow-xl backdrop-blur-sm flex-1">
            <h1 className="text-3xl font-extrabold bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent mb-2">DocuMind</h1>
            <p className="text-slate-400 text-sm mb-8">Upload up to 2 PDFs and start interacting with your documents completely locally.</p>

            {!isUploaded ? (
              <div className="space-y-4">
                <div
                  className="border-2 border-dashed border-slate-700/80 bg-slate-900/30 rounded-xl p-10 text-center hover:border-indigo-500 hover:bg-slate-800/50 transition-all cursor-pointer group"
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
                  <div className="bg-slate-800 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform">
                    <Upload className="w-8 h-8 text-indigo-400" />
                  </div>
                  <p className="text-sm text-slate-300 font-medium">Click to select PDFs</p>
                  <p className="text-xs text-slate-500 mt-2">Max 2 documents</p>
                </div>

                {files.length > 0 && (
                  <div className="space-y-3 mt-6">
                    {files.map((f, i) => (
                      <div key={i} className="flex items-center gap-3 p-3 bg-slate-800/50 rounded-lg border border-slate-700/50 shadow-sm">
                        <FileText className="w-5 h-5 text-indigo-400 shrink-0" />
                        <span className="text-sm truncate flex-1 text-slate-300">{f.name}</span>
                      </div>
                    ))}

                    <button
                      onClick={handleUpload}
                      disabled={isUploading}
                      className="w-full mt-4 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold py-3 rounded-lg flex items-center justify-center gap-2 transition-all shadow-lg shadow-indigo-900/20 disabled:opacity-50"
                    >
                      {isUploading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Upload className="w-5 h-5" />}
                      {isUploading ? 'Extracting text...' : 'Process Documents'}
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <div className="p-8 bg-emerald-500/10 border border-emerald-500/20 rounded-xl text-center space-y-4 mt-8 animate-in fade-in zoom-in duration-500">
                <div className="w-16 h-16 bg-emerald-500/20 rounded-full flex items-center justify-center mx-auto">
                  <CheckCircle2 className="w-8 h-8 text-emerald-400" />
                </div>
                <div>
                  <h3 className="font-semibold text-emerald-300 text-lg">Ready to Ask!</h3>
                  <p className="text-sm text-emerald-400/80 mt-1">Your PDFs are embedded in local memory.</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Chat Section */}
        <div className="lg:col-span-2 flex flex-col h-full bg-slate-900/50 border border-slate-800 rounded-2xl shadow-xl backdrop-blur-sm overflow-hidden relative">

          {!isUploaded && (
            <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm z-10 flex flex-col items-center justify-center text-center p-6 transition-all duration-500">
              <div className="bg-slate-800/50 p-6 rounded-full mb-6 border border-slate-700/50">
                <Bot className="w-12 h-12 text-indigo-500/50" />
              </div>
              <h2 className="text-2xl font-medium text-slate-200">Awaiting Knowledge</h2>
              <p className="text-slate-500 mt-2 max-w-sm">Upload documents on the left to inject them into my memory before we chat.</p>
            </div>
          )}

          <div className="flex-1 overflow-y-auto p-6 space-y-6 custom-scrollbar">
            {chatHistory.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-slate-500 opacity-60">
                <Bot className="w-16 h-16 mb-4 text-slate-600" />
                <p className="text-lg">Ask me anything about your documents.</p>
              </div>
            ) : (
              chatHistory.map((msg, i) => (
                <div key={i} className={`flex gap-4 animate-in fade-in slide-in-from-bottom-2 duration-300 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  {msg.role === 'bot' && (
                    <div className="w-10 h-10 rounded-full bg-indigo-600/20 flex items-center justify-center border border-indigo-500/30 shrink-0">
                      <Bot className="w-5 h-5 text-indigo-400" />
                    </div>
                  )}

                  <div className={`max-w-[85%] rounded-2xl p-5 shadow-sm ${msg.role === 'user'
                      ? 'bg-indigo-600 text-white rounded-tr-sm'
                      : 'bg-slate-800/80 border border-slate-700 text-slate-200 rounded-tl-sm'
                    }`}>
                    <p className="whitespace-pre-wrap leading-relaxed text-[15px]">{msg.text}</p>

                    {msg.sources && msg.sources.length > 0 && (
                      <div className="mt-5 pt-4 border-t border-slate-700/60">
                        <p className="text-xs font-semibold text-slate-400 mb-3 uppercase tracking-wider flex items-center gap-2">
                          <FileText className="w-3.5 h-3.5" />
                          Sources Context
                        </p>
                        <div className="flex flex-wrap gap-2">
                          {msg.sources.map((s, idx) => (
                            <span key={idx} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-slate-950/50 border border-slate-700 text-xs text-indigo-300 font-medium hover:bg-slate-900 transition-colors">
                              {s.pdf} <span className="text-slate-500 px-1">•</span> Page {s.page}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {msg.role === 'user' && (
                    <div className="w-10 h-10 rounded-full bg-slate-700 flex items-center justify-center shrink-0 border border-slate-600">
                      <User className="w-5 h-5 text-slate-300" />
                    </div>
                  )}
                </div>
              ))
            )}
            {isAsking && (
              <div className="flex gap-4 justify-start animate-in fade-in duration-300">
                <div className="w-10 h-10 rounded-full bg-indigo-600/20 flex items-center justify-center border border-indigo-500/30 shrink-0">
                  <Loader2 className="w-5 h-5 text-indigo-400 animate-spin" />
                </div>
                <div className="bg-slate-800/80 border border-slate-700 rounded-2xl rounded-tl-sm p-5 flex items-center gap-2">
                  <div className="w-2.5 h-2.5 rounded-full bg-slate-500 animate-bounce" />
                  <div className="w-2.5 h-2.5 rounded-full bg-slate-500 animate-bounce delay-75" />
                  <div className="w-2.5 h-2.5 rounded-full bg-slate-500 animate-bounce delay-150" />
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          <div className="p-5 border-t border-slate-800 bg-slate-900/90 backdrop-blur-md">
            <form onSubmit={handleAsk} className="relative flex items-center">
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask a question about the PDFs..."
                className="w-full bg-slate-950/50 border border-slate-700 rounded-xl py-4 pl-5 pr-14 text-[15px] focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition-all placeholder:text-slate-500"
                disabled={!isUploaded || isAsking}
              />
              <button
                type="submit"
                disabled={!isUploaded || isAsking || !question.trim()}
                className="absolute right-2 p-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg transition-colors disabled:opacity-40 disabled:hover:bg-indigo-600"
              >
                <Send className="w-5 h-5" />
              </button>
            </form>
          </div>

        </div>

      </div>
    </div>
  );
}
