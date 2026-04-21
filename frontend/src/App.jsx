import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileText, CheckCircle2, ChevronDown, Activity, Settings, LayoutDashboard, Microscope, Eye, EyeOff } from 'lucide-react';

import LiveTerminal from './components/LiveTerminal';
import DetectionVisuals from './components/DetectionVisuals';

function App() {
  const [activeTab, setActiveTab] = useState('microscope'); // microscope, arena
  
  const [mode, setMode] = useState('OCR (Tesseract + EasyOCR)');
  const [apiKeys, setApiKeys] = useState({ gemini: '', groq: '' });
  const [provider, setProvider] = useState('gemini');
  const [apiModel, setApiModel] = useState('gemini-2.0-flash');
  
  // Microscope State
  const [handwrittenFile, setHandwrittenFile] = useState(null);
  const [printedFile, setPrintedFile] = useState(null);
  const [results, setResults] = useState(null);
  
  // Arena State
  const [datasetPath, setDatasetPath] = useState('');
  const [arenaStrategies, setArenaStrategies] = useState(['OCR (Tesseract + EasyOCR)']);
  const [arenaStatus, setArenaStatus] = useState('idle');
  const [arenaError, setArenaError] = useState(null);
  const [arenaLogs, setArenaLogs] = useState([]);
  const [arenaProgress, setArenaProgress] = useState(null);
  const [arenaResults, setArenaResults] = useState(null);
  
  // Laboratory State
  const [historyRuns, setHistoryRuns] = useState([]);
  const [historyView, setHistoryView] = useState('arena');
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  
  // API Key Visibility State
  const [showApiKey, setShowApiKey] = useState(false);
  const [showArenaGeminiKey, setShowArenaGeminiKey] = useState(false);
  const [showArenaGroqKey, setShowArenaGroqKey] = useState(false);
  
  // Shared Process State
  const [status, setStatus] = useState('idle'); // idle, processing, done, error
  const [error, setError] = useState(null);
  const [logs, setLogs] = useState([]);
  const [elapsedTime, setElapsedTime] = useState(0);
  const timerRef = useRef(null);

  useEffect(() => {
    return () => clearInterval(timerRef.current);
  }, []);

  useEffect(() => {
    if (activeTab === 'laboratory') {
      fetchHistory(historyView);
    }
  }, [activeTab, historyView]);

  const fetchHistory = async (view) => {
    setIsLoadingHistory(true);
    try {
      const res = await fetch(`http://localhost:8000/api/history?view=${view}`);
      const data = await res.json();
      setHistoryRuns(data.runs || []);
    } catch(e) {
      console.error(e);
    }
    setIsLoadingHistory(false);
  };

  const handleFileDrop = (e, setter) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setter(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e, setter) => {
    if (e.target.files && e.target.files.length > 0) {
      setter(e.target.files[0]);
    }
  };

  const startTimer = () => {
    setElapsedTime(0);
    clearInterval(timerRef.current);
    timerRef.current = setInterval(() => setElapsedTime(prev => prev + 0.1), 100);
  };

  const stopTimer = () => {
    clearInterval(timerRef.current);
  };

  const getTimestamp = () => {
    const d = new Date();
    return `${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}:${d.getSeconds().toString().padStart(2, '0')}.${d.getMilliseconds().toString().padStart(3, '0')}`;
  };

  const analyzeDocuments = async () => {
    if (!handwrittenFile || !printedFile) {
      setError("Please upload both documents.");
      return;
    }
    
    setStatus('processing');
    setError(null);
    setResults(null);
    setLogs([]);
    startTimer();
    
    const formData = new FormData();
    formData.append('handwritten_file', handwrittenFile);
    formData.append('printed_file', printedFile);
    formData.append('mode', mode);
    if (apiKeys[provider]) {
      formData.append('api_key', apiKeys[provider]);
    }
    formData.append('provider', provider);
    formData.append('api_model', apiModel);

    try {
      // Using fetch to read SSE stream
      const response = await fetch('http://localhost:8000/api/analyze/stream', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        
        const events = buffer.split("\n\n");
        buffer = events.pop(); 
        
        for (const ev of events) {
          if (!ev.trim()) continue;
          
          const lines = ev.split("\n");
          const eventLine = lines.find(l => l.startsWith('event:'));
          const dataLine = lines.find(l => l.startsWith('data:'));
          
          if (!eventLine || !dataLine) continue;
          
          const eventType = eventLine.replace("event:", "").trim();
          const dataPayload = JSON.parse(dataLine.replace("data:", "").trim());
          
          if (eventType === "log") {
            setLogs(prev => [...prev, { ...dataPayload, timestamp: getTimestamp() }]);
          } else if (eventType === "result") {
            setResults(dataPayload);
            setStatus('done');
            stopTimer();
            setTimeout(() => {
              document.getElementById('results-section')?.scrollIntoView({ behavior: 'smooth' });
            }, 300);
          } else if (eventType === "error") {
            setError(dataPayload.detail);
            setStatus('error');
            stopTimer();
          }
        }
      }
    } catch (err) {
      setError(err.message || "An error occurred during analysis.");
      setStatus('error');
      stopTimer();
    }
  };

  const runBenchmark = async () => {
    if (!datasetPath || arenaStrategies.length === 0) {
      setArenaError("Please specify a dataset path and select at least one strategy.");
      return;
    }
    
    setArenaStatus('processing');
    setArenaError(null);
    setArenaResults(null);
    setArenaLogs([]);
    setArenaProgress(null);
    startTimer();
    
    const formData = new FormData();
    formData.append('dataset_path', datasetPath);
    formData.append('strategies', JSON.stringify(arenaStrategies));
    formData.append('api_keys', JSON.stringify(apiKeys));

    try {
      const response = await fetch('http://localhost:8000/api/arena/stream', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) throw new Error(`Server returned ${response.status}: ${response.statusText}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        
        const events = buffer.split("\n\n");
        buffer = events.pop(); 
        
        for (const ev of events) {
          if (!ev.trim()) continue;
          
          const lines = ev.split("\n");
          const eventLine = lines.find(l => l.startsWith('event:'));
          const dataLine = lines.find(l => l.startsWith('data:'));
          
          if (!eventLine || !dataLine) continue;
          
          const eventType = eventLine.replace("event:", "").trim();
          const dataPayload = JSON.parse(dataLine.replace("data:", "").trim());
          
          if (eventType === "log") {
            setArenaLogs(prev => [...prev, { ...dataPayload, timestamp: getTimestamp() }]);
          } else if (eventType === "progress") {
            setArenaProgress(dataPayload);
          } else if (eventType === "result") {
            setArenaResults(dataPayload);
            setArenaStatus('done');
            stopTimer();
            setTimeout(() => {
              document.getElementById('arena-results-section')?.scrollIntoView({ behavior: 'smooth' });
            }, 300);
          } else if (eventType === "error") {
            setArenaError(dataPayload.detail);
            setArenaStatus('error');
            stopTimer();
          }
        }
      }
    } catch (err) {
      setArenaError(err.message || "An error occurred during analysis.");
      setArenaStatus('error');
      stopTimer();
    }
  };


  const fadeInUp = {
    hidden: { opacity: 0, y: 40 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.8, ease: "easeOut" } }
  };

  const staggerContainer = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.2 }
    }
  };

  return (
    <>
      <div className="bg-blob blob-1"></div>
      <div className="bg-blob blob-2"></div>
      
      <div className="app-container">
        
        {/* HEADER */}
        <header style={{ display: 'flex', justifyContent: 'center', marginBottom: '2rem' }}>
          <div className="glass-panel" style={{ display: 'flex', gap: '1rem', padding: '0.5rem', borderRadius: '50px' }}>
            <button 
              className={`tab-button ${activeTab === 'microscope' ? 'active' : ''}`}
              onClick={() => setActiveTab('microscope')}
              style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.8rem 2rem', borderRadius: '40px', border: 'none', cursor: 'pointer', background: activeTab === 'microscope' ? 'var(--accent-primary)' : 'transparent', color: activeTab === 'microscope' ? '#fff' : 'var(--text-light)', fontWeight: 600, transition: 'all 0.3s' }}
            >
              <Microscope size={20} /> The Microscope
            </button>
            <button 
              className={`tab-button ${activeTab === 'arena' ? 'active' : ''}`}
              onClick={() => setActiveTab('arena')}
              style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.8rem 2rem', borderRadius: '40px', border: 'none', cursor: 'pointer', background: activeTab === 'arena' ? 'var(--accent-primary)' : 'transparent', color: activeTab === 'arena' ? '#fff' : 'var(--text-light)', fontWeight: 600, transition: 'all 0.3s' }}
            >
              <LayoutDashboard size={20} /> The Arena
            </button>
            <button 
              className={`tab-button ${activeTab === 'laboratory' ? 'active' : ''}`}
              onClick={() => setActiveTab('laboratory')}
              style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.8rem 2rem', borderRadius: '40px', border: 'none', cursor: 'pointer', background: activeTab === 'laboratory' ? 'var(--accent-primary)' : 'transparent', color: activeTab === 'laboratory' ? '#fff' : 'var(--text-light)', fontWeight: 600, transition: 'all 0.3s' }}
            >
              <Activity size={20} /> The Laboratory
            </button>
          </div>
        </header>

        {/* HERO SECTION */}
        <motion.section 
          className="hero"
          initial="hidden"
          animate="visible"
          variants={staggerContainer}
          style={{ display: activeTab === 'laboratory' ? 'none' : 'block', paddingTop: '1rem', paddingBottom: '1rem', marginBottom: '0.5rem' }}
        >
          <motion.h1 variants={fadeInUp}>
            Document <span className="gradient-text">Similarity</span> Analyzer
          </motion.h1>
          <motion.p variants={fadeInUp} style={{ textAlign: 'center', maxWidth: '600px', margin: '0 auto' }}>
            {activeTab === 'microscope' 
              ? "Extract text from handwritten and printed documents, then compute multi-metric similarity scores instantly." 
              : "Race extraction models against a ground-truth dataset in a batch benchmark."}
          </motion.p>
        </motion.section>

        {/* CONFIG SECTION */}
        <motion.section 
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          variants={fadeInUp}
          style={{ marginBottom: '4rem', display: activeTab === 'laboratory' ? 'none' : 'block' }}
        >
          <div className="glass-panel">
            <h2 style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Settings className="gradient-text" /> Configuration
            </h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1.5rem' }}>
              <div>
                <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)' }}>Extraction Mode</label>
                <select 
                  className="glass-input glass-select"
                  value={mode}
                  onChange={(e) => setMode(e.target.value)}
                >
                  <option value="OCR (Tesseract + EasyOCR)">🔍 OCR (Tesseract + EasyOCR)</option>
                  <option value="API (Cloud Vision)">🌐 API (Cloud Vision)</option>
                  {/* Local Model & Custom Model removed as requested */}
                </select>
              </div>
              
              {mode.includes('API') && (
                <>
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)' }}>Provider</label>
                    <select 
                      className="glass-input glass-select"
                      value={provider}
                      onChange={(e) => {
                        const p = e.target.value;
                        setProvider(p);
                        setApiModel(p === 'gemini' ? 'gemini-2.0-flash' : 'meta-llama/llama-4-scout-17b-16e-instruct');
                      }}
                    >
                      <option value="gemini">Google Gemini</option>
                      <option value="groq">Groq</option>
                    </select>
                  </motion.div>
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)' }}>Model</label>
                    <select 
                      className="glass-input glass-select"
                      value={apiModel}
                      onChange={(e) => setApiModel(e.target.value)}
                    >
                      {provider === 'gemini' ? (
                        <>
                          <option value="gemini-2.5-flash">gemini-2.5-flash</option>
                          <option value="gemini-2.5-pro">gemini-2.5-pro</option>
                          <option value="gemini-2.0-flash">gemini-2.0-flash</option>
                          <option value="gemini-1.5-pro">gemini-1.5-pro</option>
                        </>
                      ) : (
                        <option value="meta-llama/llama-4-scout-17b-16e-instruct">llama-4-scout-17b (Vision)</option>
                      )}
                    </select>
                  </motion.div>
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)' }}>API Key</label>
                    <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
                      <input 
                        type={showApiKey ? 'text' : 'password'}
                        className="glass-input" 
                        placeholder={`Enter ${provider === 'gemini' ? 'Gemini' : 'Groq'} API Key`}
                        value={apiKeys[provider] || ''}
                        onChange={(e) => setApiKeys({...apiKeys, [provider]: e.target.value})}
                        style={{ paddingRight: '3rem' }}
                      />
                      <button
                        onClick={() => setShowApiKey(v => !v)}
                        style={{ position: 'absolute', right: '0.75rem', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', padding: '0.25rem' }}
                        title={showApiKey ? 'Hide key' : 'Show key'}
                      >
                        {showApiKey ? <EyeOff size={16} /> : <Eye size={16} />}
                      </button>
                    </div>
                  </motion.div>
                </>
              )}
            </div>
          </div>
        </motion.section>

        <AnimatePresence mode="wait">
          {activeTab === 'microscope' && (
            <motion.div
              key="microscope"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              {/* UPLOAD SECTION */}
              <section style={{ marginBottom: '4rem' }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem' }}>
                  
                  {/* Handwritten Dropzone */}
                  <div className="glass-panel">
                    <h3 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <FileText /> Handwritten Document
                    </h3>
                    <label 
                      className={`dropzone ${handwrittenFile ? 'active' : ''}`}
                      onDragOver={(e) => e.preventDefault()}
                      onDrop={(e) => handleFileDrop(e, setHandwrittenFile)}
                    >
                      {handwrittenFile ? (
                        <>
                          <CheckCircle2 size={48} color="var(--accent-secondary)" />
                          <span style={{ fontWeight: '500' }}>{handwrittenFile.name}</span>
                        </>
                      ) : (
                        <>
                          <Upload size={48} color="var(--accent-primary)" />
                          <span>Drag & drop or click to upload</span>
                        </>
                      )}
                      <input type="file" onChange={(e) => handleFileChange(e, setHandwrittenFile)} accept="image/*,.pdf" />
                    </label>
                  </div>

                  {/* Printed Dropzone */}
                  <div className="glass-panel">
                    <h3 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <FileText /> Printed Document
                    </h3>
                    <label 
                      className={`dropzone ${printedFile ? 'active' : ''}`}
                      onDragOver={(e) => e.preventDefault()}
                      onDrop={(e) => handleFileDrop(e, setPrintedFile)}
                    >
                      {printedFile ? (
                        <>
                          <CheckCircle2 size={48} color="var(--accent-secondary)" />
                          <span style={{ fontWeight: '500' }}>{printedFile.name}</span>
                        </>
                      ) : (
                        <>
                          <Upload size={48} color="var(--accent-primary)" />
                          <span>Drag & drop or click to upload</span>
                        </>
                      )}
                      <input type="file" onChange={(e) => handleFileChange(e, setPrintedFile)} accept="image/*,.pdf" />
                    </label>
                  </div>

                </div>

                <div style={{ textAlign: 'center', marginTop: '3rem' }}>
                  {error && <p style={{ color: 'var(--accent-danger)', marginBottom: '1rem', background: 'rgba(252,92,124,0.1)', padding: '1rem', borderRadius: '12px' }}>{error}</p>}
                  
                  <button 
                    className="btn-primary" 
                    onClick={analyzeDocuments} 
                    disabled={!handwrittenFile || !printedFile || status === 'processing'}
                    style={{ padding: '1.2rem 3rem', fontSize: '1.2rem' }}
                  >
                    {status === 'processing' ? (
                      <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <div className="spinner"></div> Analyzing...
                      </span>
                    ) : (
                      'Analyze & Compare'
                    )}
                  </button>
                </div>
              </section>

              {/* LIVE TERMINAL */}
              {(status !== 'idle' || logs.length > 0) && (
                <LiveTerminal logs={logs} elapsedTime={elapsedTime} status={status} />
              )}

              {/* RESULTS SECTION */}
              {results && (
                <motion.section 
                  id="results-section"
                  initial="hidden"
                  whileInView="visible"
                  viewport={{ once: true }}
                  variants={staggerContainer}
                  className="glass-panel"
                  style={{ marginTop: '4rem', marginBottom: '4rem' }}
                >
                  <motion.h2 variants={fadeInUp} style={{ textAlign: 'center', marginBottom: '3rem', fontSize: '2.5rem' }}>
                    Analysis <span className="gradient-text">Results</span>
                  </motion.h2>

                  <DetectionVisuals 
                    visuals1={results.visuals1} 
                    visuals2={results.visuals2} 
                    heatmap1={results.heatmap1} 
                    heatmap2={results.heatmap2} 
                  />

                  <div className="results-grid" style={{ marginTop: '3rem' }}>
                    
                    {/* Texts */}
                    <motion.div variants={fadeInUp}>
                      <h3 style={{ marginBottom: '1rem', color: 'var(--text-muted)' }}>Handwritten Extracted Text</h3>
                      <div className="text-area-box">{results.text1 || '(No text extracted)'}</div>
                    </motion.div>
                    
                    <motion.div variants={fadeInUp}>
                      <h3 style={{ marginBottom: '1rem', color: 'var(--text-muted)' }}>Printed Extracted Text</h3>
                      <div className="text-area-box">{results.text2 || '(No text extracted)'}</div>
                    </motion.div>

                    {/* Scores */}
                    <motion.div variants={fadeInUp} style={{ gridColumn: '1 / -1', marginTop: '2rem' }}>
                      <h3 style={{ marginBottom: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>Similarity Metrics</h3>
                      
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '2rem' }}>
                        {Object.entries(results.scores).filter(([k]) => k !== 'Final Similarity').map(([key, val], idx) => {
                          const percentage = Math.round(val * 100);
                          return (
                            <div key={idx} className="score-card">
                              <div className="score-header">
                                <span className="score-label">{key}</span>
                                <span className="score-value">{val.toFixed(4)}</span>
                              </div>
                              <div className="progress-container">
                                <motion.div 
                                  className="progress-bar"
                                  initial={{ width: 0 }}
                                  whileInView={{ width: `${percentage}%` }}
                                  transition={{ duration: 1.5, ease: "easeOut" }}
                                  style={{ background: `linear-gradient(90deg, var(--accent-primary), var(--accent-secondary))` }}
                                />
                              </div>
                              <div style={{ textAlign: 'right', fontSize: '0.8rem', color: 'var(--text-muted)' }}>{percentage}%</div>
                            </div>
                          )
                        })}
                      </div>

                      {/* Final Score */}
                      {results.scores['Final Similarity'] !== undefined && (
                        <motion.div 
                          initial={{ scale: 0.9, opacity: 0 }}
                          whileInView={{ scale: 1, opacity: 1 }}
                          transition={{ delay: 0.5, duration: 0.8 }}
                          style={{ 
                            marginTop: '4rem', 
                            textAlign: 'center',
                            padding: '2rem',
                            background: 'rgba(0,0,0,0.2)',
                            borderRadius: '20px',
                            border: '1px solid var(--glass-border)'
                          }}
                        >
                          <div style={{ fontSize: '1.2rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Final Similarity Score</div>
                          <div className="gradient-text" style={{ fontSize: '5rem', fontWeight: '700', lineHeight: '1' }}>
                            {Math.round(results.scores['Final Similarity'] * 100)}%
                          </div>
                          
                          <button 
                            className="btn-primary" 
                            style={{ marginTop: '2rem' }}
                            onClick={async () => {
                              try {
                                const payload = {
                                  view: "microscope",
                                  strategy: mode.includes('API') ? `${mode} (${provider}/${apiModel})` : mode,
                                  file_name: `${handwrittenFile?.name || 'unknown'} vs ${printedFile?.name || 'unknown'}`,
                                  scores: results.scores,
                                  metadata: { provider: mode.includes('API') ? provider : null, model: mode.includes('API') ? apiModel : null }
                                };
                                const res = await fetch('http://localhost:8000/api/history/save', {
                                  method: 'POST',
                                  headers: { 'Content-Type': 'application/json' },
                                  body: JSON.stringify(payload)
                                });
                                if (res.ok) alert('Results saved to Laboratory!');
                                else alert('Failed to save results.');
                              } catch(e) {
                                alert('Error saving results: ' + e.message);
                              }
                            }}
                          >
                            💾 Save Results
                          </button>
                        </motion.div>
                      )}
                    </motion.div>

                  </div>
                </motion.section>
              )}
            </motion.div>
          )}

          {activeTab === 'arena' && (
            <motion.div
              key="arena"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="glass-panel"
            >
              <h2 style={{ textAlign: 'center', marginBottom: '2rem' }}>⚔️ The Arena (Batch Benchmarking)</h2>
              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '2rem', borderRadius: '12px', textAlign: 'center' }}>
                <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
                  Arena mode allows you to benchmark extraction models against ground-truth datasets.
                </p>
                <input 
                  type="text" 
                  className="glass-input" 
                  placeholder="📂 Dataset Folder Path (e.g., C:\datasets\handwriting_benchmark)"
                  value={datasetPath}
                  onChange={(e) => setDatasetPath(e.target.value)}
                  style={{ maxWidth: '600px', margin: '0 auto 1.5rem auto' }}
                />

                <div style={{ maxWidth: '600px', margin: '0 auto 2rem auto', textAlign: 'left' }}>
                  <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)' }}>Select Strategies to Race</label>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', background: 'rgba(255,255,255,0.02)', padding: '1rem', borderRadius: '8px' }}>
                    {['OCR (Tesseract + EasyOCR)', 'API (Cloud Vision - Gemini)', 'API (Cloud Vision - Groq)'].map(strategy => (
                      <label key={strategy} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', color: 'var(--text-light)' }}>
                        <input 
                          type="checkbox" 
                          checked={arenaStrategies.includes(strategy)}
                          onChange={(e) => {
                            if (e.target.checked) setArenaStrategies([...arenaStrategies, strategy]);
                            else setArenaStrategies(arenaStrategies.filter(s => s !== strategy));
                          }}
                          style={{ accentColor: 'var(--accent-primary)', width: '18px', height: '18px' }}
                        />
                        {strategy}
                      </label>
                    ))}
                  </div>
                  {arenaStrategies.some(s => s.includes('API')) && (
                    <div style={{ marginTop: '1rem', display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                      {arenaStrategies.some(s => s.includes('Gemini')) && (
                        <div style={{ flex: 1, position: 'relative', display: 'flex', alignItems: 'center' }}>
                          <input 
                            type={showArenaGeminiKey ? 'text' : 'password'}
                            className="glass-input" 
                            style={{ width: '100%', paddingRight: '3rem' }}
                            placeholder="Gemini API Key"
                            value={apiKeys.gemini}
                            onChange={(e) => setApiKeys({...apiKeys, gemini: e.target.value})}
                          />
                          <button onClick={() => setShowArenaGeminiKey(v => !v)} style={{ position: 'absolute', right: '0.75rem', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}>
                            {showArenaGeminiKey ? <EyeOff size={16} /> : <Eye size={16} />}
                          </button>
                        </div>
                      )}
                      {arenaStrategies.some(s => s.includes('Groq')) && (
                        <div style={{ flex: 1, position: 'relative', display: 'flex', alignItems: 'center' }}>
                          <input 
                            type={showArenaGroqKey ? 'text' : 'password'}
                            className="glass-input" 
                            style={{ width: '100%', paddingRight: '3rem' }}
                            placeholder="Groq API Key"
                            value={apiKeys.groq}
                            onChange={(e) => setApiKeys({...apiKeys, groq: e.target.value})}
                          />
                          <button onClick={() => setShowArenaGroqKey(v => !v)} style={{ position: 'absolute', right: '0.75rem', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}>
                            {showArenaGroqKey ? <EyeOff size={16} /> : <Eye size={16} />}
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                <button className="btn-primary" disabled={!datasetPath || arenaStrategies.length === 0 || arenaStatus === 'processing'} onClick={runBenchmark}>
                  {arenaStatus === 'processing' ? 'Running Benchmark...' : '⚔️ Run Benchmark'}
                </button>
              </div>

              {arenaError && (
                <div style={{ padding: '1rem', background: 'rgba(255, 50, 50, 0.1)', border: '1px solid rgba(255, 50, 50, 0.3)', borderRadius: '12px', marginTop: '2rem', color: '#ff6b6b' }}>
                  ❌ {arenaError}
                </div>
              )}

              {/* Progress and Live Terminal */}
              {(arenaStatus === 'processing' || arenaStatus === 'done') && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ marginTop: '2rem' }}>
                  {arenaProgress && (
                    <div style={{ marginBottom: '1.5rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
                        <span style={{ color: 'var(--text-muted)' }}>{arenaProgress.status}</span>
                        <span>{Math.round((arenaProgress.current / arenaProgress.total) * 100)}%</span>
                      </div>
                      <div className="progress-container">
                        <motion.div 
                          className="progress-bar"
                          style={{ width: `${(arenaProgress.current / arenaProgress.total) * 100}%`, background: 'var(--accent-primary)' }}
                        />
                      </div>
                    </div>
                  )}
                  <LiveTerminal logs={arenaLogs} elapsedTime={elapsedTime} />
                </motion.div>
              )}

              {/* Arena Results */}
              {arenaStatus === 'done' && arenaResults && (
                <motion.section id="arena-results-section" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} style={{ marginTop: '4rem' }}>
                  <h3 style={{ textAlign: 'center', marginBottom: '2rem' }}>🏆 Arena Scores — Strategy Leaderboard</h3>
                  
                  <div style={{ overflowX: 'auto', marginBottom: '3rem' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', background: 'rgba(0,0,0,0.2)', borderRadius: '12px', overflow: 'hidden' }}>
                      <thead>
                        <tr style={{ background: 'rgba(255,255,255,0.05)', color: 'var(--text-muted)' }}>
                          <th style={{ padding: '1rem' }}>Strategy</th>
                          <th style={{ padding: '1rem' }}>Files</th>
                          <th style={{ padding: '1rem' }}>Arena Score (Avg Final)</th>
                          <th style={{ padding: '1rem' }}>Avg Edit</th>
                          <th style={{ padding: '1rem' }}>Avg TF-IDF</th>
                          <th style={{ padding: '1rem' }}>Avg Embedding</th>
                          <th style={{ padding: '1rem' }}>Avg Words</th>
                          <th style={{ padding: '1rem' }}>Time (s)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {arenaResults.summary.map((row, idx) => (
                          <tr key={idx} style={{ borderTop: '1px solid var(--glass-border)' }}>
                            <td style={{ padding: '1rem', fontWeight: 'bold', color: 'var(--text-light)' }}>{row.Strategy}</td>
                            <td style={{ padding: '1rem' }}>{row['Files Processed']}</td>
                            <td style={{ padding: '1rem' }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                <div style={{ width: '100px', height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px', overflow: 'hidden' }}>
                                  <div style={{ width: `${row['Arena Score (Avg Final)'] * 100}%`, height: '100%', background: 'var(--accent-primary)' }} />
                                </div>
                                <span>{row['Arena Score (Avg Final)'].toFixed(4)}</span>
                              </div>
                            </td>
                            <td style={{ padding: '1rem' }}>{row['Avg Edit Similarity'].toFixed(4)}</td>
                            <td style={{ padding: '1rem' }}>{row['Avg TF-IDF Similarity'].toFixed(4)}</td>
                            <td style={{ padding: '1rem' }}>{row['Avg Embedding Similarity'].toFixed(4)}</td>
                            <td style={{ padding: '1rem' }}>{row['Avg Word Count']}</td>
                            <td style={{ padding: '1rem' }}>{row['Total Time (s)'].toFixed(2)}s</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <div style={{ textAlign: 'center', marginTop: '2rem', marginBottom: '2rem' }}>
                    <button 
                      className="btn-primary" 
                      onClick={async () => {
                        try {
                          let successCount = 0;
                          for (const row of arenaResults.summary) {
                            const payload = {
                              view: "arena",
                              strategy: row.Strategy,
                              file_name: datasetPath,
                              scores: {
                                "Arena Score": row['Arena Score (Avg Final)'],
                                "Avg Edit": row['Avg Edit Similarity'],
                                "Avg TF-IDF": row['Avg TF-IDF Similarity'],
                                "Avg Embedding": row['Avg Embedding Similarity']
                              },
                              metadata: {
                                "files_processed": row['Files Processed'],
                                "total_time": row['Total Time (s)']
                              }
                            };
                            const res = await fetch('http://localhost:8000/api/history/save', {
                              method: 'POST',
                              headers: { 'Content-Type': 'application/json' },
                              body: JSON.stringify(payload)
                            });
                            if (res.ok) successCount++;
                          }
                          alert(`Saved ${successCount} strategies to Laboratory!`);
                        } catch(e) {
                          alert('Error saving results: ' + e.message);
                        }
                      }}
                    >
                      💾 Save Benchmark Results
                    </button>
                  </div>

                  <details style={{ background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '12px', border: '1px solid var(--glass-border)' }}>
                    <summary style={{ cursor: 'pointer', fontWeight: 'bold', outline: 'none' }}>📋 Detailed Per-File Results</summary>
                    <div style={{ marginTop: '1rem', overflowX: 'auto' }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.9rem' }}>
                        <thead>
                          <tr style={{ color: 'var(--text-muted)' }}>
                            <th style={{ padding: '0.5rem' }}>File</th>
                            <th style={{ padding: '0.5rem' }}>Strategy</th>
                            <th style={{ padding: '0.5rem' }}>Final Score</th>
                            <th style={{ padding: '0.5rem' }}>Time</th>
                          </tr>
                        </thead>
                        <tbody>
                          {arenaResults.results.map((r, idx) => (
                            <tr key={idx} style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                              <td style={{ padding: '0.5rem' }}>{r.File}</td>
                              <td style={{ padding: '0.5rem', color: 'var(--text-light)' }}>{r.Strategy}</td>
                              <td style={{ padding: '0.5rem' }}>{r['Final Similarity']?.toFixed(4) || 'Error'}</td>
                              <td style={{ padding: '0.5rem' }}>{r['Time (s)']}s</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </details>
                </motion.section>
              )}
            </motion.div>
          )}

          {activeTab === 'laboratory' && (
            <motion.div
              key="laboratory"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="glass-panel"
            >
              <h2 style={{ textAlign: 'center', marginBottom: '2rem' }}>🧪 The Laboratory (History)</h2>
              
              <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', marginBottom: '2rem' }}>
                <button 
                  className={`btn-primary`} 
                  style={{ background: historyView === 'arena' ? 'var(--accent-primary)' : 'rgba(255,255,255,0.1)' }}
                  onClick={() => setHistoryView('arena')}
                >
                  ⚔️ Arena Benchmarks
                </button>
                <button 
                  className={`btn-primary`} 
                  style={{ background: historyView === 'microscope' ? 'var(--accent-primary)' : 'rgba(255,255,255,0.1)' }}
                  onClick={() => setHistoryView('microscope')}
                >
                  🔍 Microscope Runs
                </button>
              </div>

              {isLoadingHistory ? (
                <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>Loading history...</div>
              ) : historyRuns.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>No history found for {historyView}.</div>
              ) : (
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', background: 'rgba(0,0,0,0.2)', borderRadius: '12px', overflow: 'hidden' }}>
                    <thead>
                      <tr style={{ background: 'rgba(255,255,255,0.05)', color: 'var(--text-muted)' }}>
                        <th style={{ padding: '1rem' }}>Date</th>
                        <th style={{ padding: '1rem' }}>Strategy</th>
                        <th style={{ padding: '1rem' }}>{historyView === 'arena' ? 'Dataset' : 'Files'}</th>
                        {historyView === 'arena' && <th style={{ padding: '1rem' }}>Files Processed</th>}
                        <th style={{ padding: '1rem' }}>Final Score</th>
                        <th style={{ padding: '1rem' }}>Scores Drilldown</th>
                      </tr>
                    </thead>
                    <tbody>
                      {historyRuns.map((run, idx) => (
                        <tr key={idx} style={{ borderTop: '1px solid var(--glass-border)' }}>
                          <td style={{ padding: '1rem', color: 'var(--text-muted)', fontSize: '0.85rem' }}>{new Date(run.timestamp).toLocaleString()}</td>
                          <td style={{ padding: '1rem', fontWeight: 'bold', color: 'var(--text-light)' }}>
                            {run.strategy}
                            {run.metadata?.provider && (
                              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 'normal', marginTop: '0.2rem' }}>
                                {run.metadata.provider} / {run.metadata.model}
                              </div>
                            )}
                          </td>
                          <td style={{ padding: '1rem' }}>{run.file_name}</td>
                          {historyView === 'arena' && <td style={{ padding: '1rem' }}>{run.metadata?.files_processed || 0}</td>}
                          <td style={{ padding: '1rem' }}>
                            <div className="gradient-text" style={{ fontWeight: 'bold', fontSize: '1.2rem' }}>
                              {historyView === 'arena' 
                                ? (run.scores['Arena Score'] * 100).toFixed(1)
                                : (run.scores['Final Similarity'] * 100).toFixed(1)}%
                            </div>
                          </td>
                          <td style={{ padding: '1rem', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                            {Object.entries(run.scores).map(([k, v]) => (
                              <div key={k}>{k}: {v.toFixed(4)}</div>
                            ))}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>

      </div>
    </>
  );
}

export default App;
