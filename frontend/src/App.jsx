import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Upload, FileText, CheckCircle2, ChevronDown, Activity, Settings } from 'lucide-react';
import axios from 'axios';

function App() {
  const [mode, setMode] = useState('OCR (Tesseract + EasyOCR)');
  const [apiKey, setApiKey] = useState('');
  const [handwrittenFile, setHandwrittenFile] = useState(null);
  const [printedFile, setPrintedFile] = useState(null);
  
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

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

  const analyzeDocuments = async () => {
    if (!handwrittenFile || !printedFile) {
      setError("Please upload both documents.");
      return;
    }
    
    setLoading(true);
    setError(null);
    setResults(null);
    
    const formData = new FormData();
    formData.append('handwritten_file', handwrittenFile);
    formData.append('printed_file', printedFile);
    formData.append('mode', mode);
    if (apiKey) {
      formData.append('api_key', apiKey);
    }

    try {
      // Assuming FastAPI runs on 8000
      const response = await axios.post('http://localhost:8000/api/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResults(response.data);
      
      // Scroll to results
      setTimeout(() => {
        document.getElementById('results-section')?.scrollIntoView({ behavior: 'smooth' });
      }, 300);
      
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "An error occurred during analysis.");
    } finally {
      setLoading(false);
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
        
        {/* HERO SECTION */}
        <motion.section 
          className="hero"
          initial="hidden"
          animate="visible"
          variants={staggerContainer}
        >
          <motion.h1 variants={fadeInUp}>
            Document <span className="gradient-text">Similarity</span> Analyzer
          </motion.h1>
          <motion.p variants={fadeInUp}>
            Extract text from handwritten and printed documents, then compute multi-metric similarity scores instantly.
          </motion.p>
          <motion.div variants={fadeInUp} style={{ marginTop: '2rem' }}>
            <ChevronDown size={32} color="var(--accent-primary)" style={{ animation: 'float 2s infinite ease-in-out' }} />
          </motion.div>
        </motion.section>

        {/* CONFIG SECTION */}
        <motion.section 
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          variants={fadeInUp}
          style={{ marginBottom: '4rem' }}
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
                  <option value="OCR (Tesseract + EasyOCR)">OCR (Tesseract + EasyOCR)</option>
                  <option value="API (Gemini 2.0 Flash Vision)">API (Gemini 2.0 Flash Vision)</option>
                  <option value="Local Model (TrOCR)">Local Model (TrOCR)</option>
                </select>
              </div>
              
              {mode.includes('API') && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                  <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)' }}>Gemini API Key</label>
                  <input 
                    type="password" 
                    className="glass-input" 
                    placeholder="Enter API Key"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                  />
                </motion.div>
              )}
            </div>
          </div>
        </motion.section>

        {/* UPLOAD SECTION */}
        <motion.section 
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          variants={staggerContainer}
          style={{ marginBottom: '4rem' }}
        >
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem' }}>
            
            {/* Handwritten Dropzone */}
            <motion.div variants={fadeInUp} className="glass-panel">
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
            </motion.div>

            {/* Printed Dropzone */}
            <motion.div variants={fadeInUp} className="glass-panel">
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
            </motion.div>

          </div>

          <motion.div variants={fadeInUp} style={{ textAlign: 'center', marginTop: '3rem' }}>
            {error && <p style={{ color: 'var(--accent-danger)', marginBottom: '1rem' }}>{error}</p>}
            
            <button 
              className="btn-primary" 
              onClick={analyzeDocuments} 
              disabled={!handwrittenFile || !printedFile || loading}
              style={{ padding: '1.2rem 3rem', fontSize: '1.2rem' }}
            >
              {loading ? (
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div className="spinner"></div>
                </span>
              ) : (
                'Analyze & Compare'
              )}
            </button>
          </motion.div>
        </motion.section>

        {/* RESULTS SECTION */}
        {results && (
          <motion.section 
            id="results-section"
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="glass-panel"
            style={{ marginBottom: '4rem' }}
          >
            <motion.h2 variants={fadeInUp} style={{ textAlign: 'center', marginBottom: '3rem', fontSize: '2.5rem' }}>
              Analysis <span className="gradient-text">Results</span>
            </motion.h2>

            <div className="results-grid">
              
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
                  </motion.div>
                )}
              </motion.div>

            </div>
          </motion.section>
        )}

      </div>
    </>
  );
}

export default App;
