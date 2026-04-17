"""
📄 Document Similarity Analyzer — Streamlit Web Application

OOPs Lab Project: Extracts text from handwritten and printed documents,
then computes multi-metric similarity scores.

Features:
    - 3 extraction modes (OCR, API, Local Model)
    - 3 similarity metrics (Edit Distance, TF-IDF, Embedding)
    - Normalized final similarity score (0–1)
    - Premium dark UI with glassmorphism design

Design Patterns Used:
    - Strategy Pattern  (Extraction modes)
    - Factory Pattern   (Strategy creation)
    - Template Method   (Similarity metrics)
    - Facade Pattern    (DocumentAnalyzer)

Usage:
    streamlit run app.py
"""

import streamlit as st
import time
import os

# ── Page Config (must be first Streamlit call) ──────────────
st.set_page_config(
    page_title="Document Similarity Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load custom CSS ─────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Imports ─────────────────────────────────────────────────
import config
from models.document import Document, DocType, FileType
from models.extraction.factory import StrategyFactory
from models.analyzer import DocumentAnalyzer
from utils.file_handler import FileHandler
from utils.preprocessor import TextPreprocessor


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
def render_sidebar():
    """Render the sidebar with mode selection and settings."""
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; padding: 1rem 0;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📄</div>
                <div style="font-size: 1.1rem; font-weight: 700; 
                     background: linear-gradient(135deg, #7c5cfc, #5cfcb4);
                     -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    Doc Similarity
                </div>
                <div style="font-size: 0.7rem; color: #6a6a80; margin-top: 0.3rem;
                     letter-spacing: 0.1em; text-transform: uppercase;">
                    OOPs Lab Project
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Mode Selection ──
        st.markdown("#### 🔧 Extraction Mode")
        mode = st.radio(
            "Choose how to extract text:",
            options=config.EXTRACTION_MODES,
            index=0,
            help="Select the engine used to extract text from your documents.",
            label_visibility="collapsed"
        )

        # ── Mode-specific settings ──
        api_key = None
        if "api" in mode.lower():
            st.markdown("---")
            st.markdown("#### 🔑 API Configuration")
            api_key = st.text_input(
                "Gemini API Key",
                type="password",
                placeholder="Enter your Google Gemini API key",
                help="Get your API key from https://aistudio.google.com/apikey"
            )

        if "local" in mode.lower():
            st.markdown("---")
            st.info(
                "⚠️ **First run will download ~900MB of model weights.** "
                "Subsequent runs use the cached model.",
                icon="🤖"
            )

        st.divider()

        # ── About Section ──
        st.markdown("#### 📊 Similarity Metrics")
        st.markdown("""
        <div style="font-size: 0.8rem; color: #a0a0b8; line-height: 1.8;">
            <div>📐 <strong>Edit Distance</strong> — Character-level edits</div>
            <div>📝 <strong>TF-IDF</strong> — Keyword importance overlap</div>
            <div>🧠 <strong>Embedding</strong> — Semantic meaning</div>
            <div>⚡ <strong>Final Score</strong> — Normalized average</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Design Patterns ──
        st.markdown("#### 🏗️ OOP Patterns Used")
        st.markdown("""
        <div style="font-size: 0.75rem; color: #6a6a80; line-height: 2;">
            <div><span style="color: #7c5cfc;">▸</span> Strategy Pattern</div>
            <div><span style="color: #5c9cfc;">▸</span> Factory Pattern</div>
            <div><span style="color: #5cfcb4;">▸</span> Template Method</div>
            <div><span style="color: #fcbc5c;">▸</span> Facade Pattern</div>
        </div>
        """, unsafe_allow_html=True)

    return mode, api_key


# ═══════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════
def render_header():
    """Render the app header."""
    st.markdown("""
        <div class="app-header">
            <div class="app-title">Document Similarity Analyzer</div>
            <div class="app-subtitle">
                Upload a handwritten document and a printed document to extract text 
                and compute multi-metric similarity scores.
            </div>
        </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  FILE UPLOAD SECTION
# ═══════════════════════════════════════════════════════════════
def render_upload_section():
    """Render the two-column file upload section."""
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
            <div class="glass-card">
                <div class="card-title">
                    <span class="icon">✍️</span> Handwritten Document
                </div>
            </div>
        """, unsafe_allow_html=True)
        handwritten_file = st.file_uploader(
            "Upload handwritten document",
            type=config.SUPPORTED_FILE_TYPES,
            key="handwritten_upload",
            help="Upload a JPG, JPEG, PNG, or PDF of a handwritten document",
            label_visibility="collapsed"
        )
        if handwritten_file:
            st.success(f"✅ Uploaded: **{handwritten_file.name}**", icon="📎")

    with col2:
        st.markdown("""
            <div class="glass-card">
                <div class="card-title">
                    <span class="icon">🖨️</span> Printed Document
                </div>
            </div>
        """, unsafe_allow_html=True)
        printed_file = st.file_uploader(
            "Upload printed document",
            type=config.SUPPORTED_FILE_TYPES,
            key="printed_upload",
            help="Upload a JPG, JPEG, PNG, or PDF of a printed document",
            label_visibility="collapsed"
        )
        if printed_file:
            st.success(f"✅ Uploaded: **{printed_file.name}**", icon="📎")

    return handwritten_file, printed_file


# ═══════════════════════════════════════════════════════════════
#  SCORE DISPLAY
# ═══════════════════════════════════════════════════════════════
def render_score_card(label: str, score: float, icon: str = "📊"):
    """Render a single score card with animated progress bar."""
    percentage = int(score * 100)

    # Color based on score
    if score >= 0.7:
        color = "#5cfcb4"  # Green
    elif score >= 0.4:
        color = "#fcbc5c"  # Yellow
    else:
        color = "#fc5c7c"  # Red

    st.markdown(f"""
        <div class="score-card">
            <div class="score-label">{icon} {label}</div>
            <div class="score-value">{score:.4f}</div>
            <div class="score-percentage">{percentage}%</div>
            <div class="progress-container">
                <div class="progress-bar" style="width: {percentage}%; 
                     background: linear-gradient(90deg, {color}88, {color});"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_final_score(score: float):
    """Render the large final similarity score card."""
    percentage = int(score * 100)

    st.markdown(f"""
        <div class="final-score-card">
            <div class="final-score-label">⚡ Final Similarity Score</div>
            <div class="final-score-value">{score:.4f}</div>
            <div class="score-percentage" style="font-size: 1.2rem;">{percentage}%</div>
            <div class="progress-container" style="height: 8px; margin-top: 1rem;">
                <div class="progress-bar" style="width: {percentage}%;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  RESULTS SECTION
# ═══════════════════════════════════════════════════════════════
def render_results(results: dict):
    """Render the full results section with extracted text and scores."""

    st.markdown("---")

    # ── Extracted Texts ──
    st.markdown("""
        <div style="text-align: center; margin: 1.5rem 0;">
            <span style="font-size: 1.3rem; font-weight: 700; color: #e8e8f0;">
                📝 Extracted Text
            </span>
        </div>
    """, unsafe_allow_html=True)

    text_col1, text_col2 = st.columns(2, gap="large")

    with text_col1:
        with st.expander("✍️ **Handwritten Document — Extracted Text**", expanded=True):
            st.text_area(
                "Text from handwritten document",
                value=results["text1"] or "(No text extracted)",
                height=250,
                key="text_output_1",
                label_visibility="collapsed"
            )

    with text_col2:
        with st.expander("🖨️ **Printed Document — Extracted Text**", expanded=True):
            st.text_area(
                "Text from printed document",
                value=results["text2"] or "(No text extracted)",
                height=250,
                key="text_output_2",
                label_visibility="collapsed"
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Similarity Scores ──
    st.markdown("""
        <div style="text-align: center; margin: 1.5rem 0;">
            <span style="font-size: 1.3rem; font-weight: 700; color: #e8e8f0;">
                📊 Similarity Scores
            </span>
        </div>
    """, unsafe_allow_html=True)

    scores = results["scores"]

    # Individual metric scores in 3 columns
    metric_icons = {
        "Edit Similarity": "📐",
        "TF-IDF Similarity": "📝",
        "Embedding Similarity": "🧠",
    }

    score_cols = st.columns(3, gap="medium")
    for i, (metric_name, icon) in enumerate(metric_icons.items()):
        if metric_name in scores:
            with score_cols[i]:
                render_score_card(metric_name, scores[metric_name], icon)

    st.markdown("<br>", unsafe_allow_html=True)

    # Final Score (large, centered)
    final_col1, final_col2, final_col3 = st.columns([1, 2, 1])
    with final_col2:
        render_final_score(scores.get("Final Similarity", 0.0))


# ═══════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════
def main():
    """Main application entry point."""

    # ── Sidebar ──
    mode, api_key = render_sidebar()

    # ── Header ──
    render_header()

    # ── Mode Badge ──
    st.markdown(f"""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <span class="mode-badge">⚙️ {mode}</span>
        </div>
    """, unsafe_allow_html=True)

    # ── File Upload ──
    handwritten_file, printed_file = render_upload_section()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Analyze Button ──
    col_left, col_center, col_right = st.columns([1, 1, 1])
    with col_center:
        analyze_clicked = st.button(
            "🚀  Analyze & Compare",
            use_container_width=True,
            disabled=not (handwritten_file and printed_file),
        )

    # ── Run Analysis ──
    if analyze_clicked:
        if not handwritten_file or not printed_file:
            st.error("⚠️ Please upload both documents before analyzing.", icon="❌")
            return

        # Validate API key for API mode
        if "api" in mode.lower() and not api_key:
            st.error("⚠️ Please enter your Gemini API key in the sidebar.", icon="🔑")
            return

        try:
            with st.spinner("🔄 Processing documents..."):
                progress_bar = st.progress(0, text="Initializing...")

                # ── Step 1: Load files ──
                progress_bar.progress(10, text="📁 Loading documents...")

                hw_ext = handwritten_file.name.rsplit(".", 1)[-1]
                pr_ext = printed_file.name.rsplit(".", 1)[-1]

                hw_images = FileHandler.load_images(
                    handwritten_file.read(),
                    FileType.from_extension(hw_ext)
                )
                pr_images = FileHandler.load_images(
                    printed_file.read(),
                    FileType.from_extension(pr_ext)
                )

                doc1 = Document(
                    file_name=handwritten_file.name,
                    file_type=FileType.from_extension(hw_ext),
                    doc_type=DocType.HANDWRITTEN,
                    images=hw_images,
                )
                doc2 = Document(
                    file_name=printed_file.name,
                    file_type=FileType.from_extension(pr_ext),
                    doc_type=DocType.PRINTED,
                    images=pr_images,
                )

                progress_bar.progress(25, text="📁 Documents loaded!")

                # ── Step 2: Create strategy ──
                progress_bar.progress(30, text="⚙️ Initializing extraction engine...")

                kwargs = {}
                if "api" in mode.lower():
                    kwargs["api_key"] = api_key

                strategy = StrategyFactory.create(mode, **kwargs)
                analyzer = DocumentAnalyzer(strategy)

                progress_bar.progress(40, text=f"🔍 Extracting text from handwritten doc...")

                # ── Step 3: Extract text ──
                text1 = analyzer.extract(doc1)
                progress_bar.progress(55, text="🔍 Extracting text from printed doc...")

                text2 = analyzer.extract(doc2)
                progress_bar.progress(70, text="✅ Text extraction complete!")

                # Clean OCR output
                text1 = TextPreprocessor.clean_ocr_output(text1)
                text2 = TextPreprocessor.clean_ocr_output(text2)

                # Update documents
                doc1.extracted_text = text1
                doc2.extracted_text = text2

                # ── Step 4: Compute similarity ──
                progress_bar.progress(80, text="📊 Computing similarity scores...")

                scores = analyzer.compare(text1, text2)
                progress_bar.progress(95, text="📊 Scores computed!")

                time.sleep(0.3)
                progress_bar.progress(100, text="✅ Analysis complete!")
                time.sleep(0.5)
                progress_bar.empty()

            # ── Step 5: Display results ──
            results = {
                "text1": text1,
                "text2": text2,
                "scores": scores,
            }

            # Store in session state
            st.session_state["results"] = results

        except Exception as e:
            st.error(f"❌ **Error:** {str(e)}", icon="🚨")
            import traceback
            with st.expander("🔍 Full Error Details"):
                st.code(traceback.format_exc())

    # ── Display stored results ──
    if "results" in st.session_state:
        render_results(st.session_state["results"])


# ── Entry Point ──────────────────────────────────────────────
if __name__ == "__main__":
    main()
