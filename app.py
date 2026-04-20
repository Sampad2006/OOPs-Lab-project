"""
🔬 LLM Extraction Benchmarking Arena — Streamlit Web Application

OOPs Lab Project: A comprehensive benchmarking platform for comparing
text extraction models across documents with scientific precision.

Views:
    - 🔬 The Microscope  — Single-pair document analysis
    - ⚔️ The Arena       — Batch benchmarking against ground truth
    - 🧪 The Laboratory  — Historical results browser

Design Patterns Used:
    - Strategy Pattern  (Extraction modes + BYOM plugins)
    - Factory Pattern   (Strategy creation with plugin discovery)
    - Template Method   (Similarity metrics)
    - Facade Pattern    (DocumentAnalyzer)

Usage:
    streamlit run app.py
"""

import streamlit as st
import os

# ── Page Config (must be first Streamlit call) ──────────────
st.set_page_config(
    page_title="LLM Extraction Benchmarking Arena",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load custom CSS ─────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, encoding='utf-8') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Imports ─────────────────────────────────────────────────
import config
from models.extraction.factory import StrategyFactory
from views import microscope, arena, laboratory


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
def render_sidebar():
    """Render the sidebar with mode selection and settings."""
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; padding: 1rem 0;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🔬</div>
                <div style="font-size: 1.1rem; font-weight: 700;
                     background: linear-gradient(135deg, #7c5cfc, #5cfcb4);
                     -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    Benchmarking Arena
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
        available = StrategyFactory.available_modes()
        mode = st.radio(
            "Choose how to extract text:",
            options=available,
            index=0,
            help="Select the engine used to extract text from your documents.",
            label_visibility="collapsed"
        )

        # Show plugin badge if any plugins found
        plugin_names = StrategyFactory.get_plugin_names()
        if plugin_names:
            st.markdown(
                f'<div style="font-size:0.7rem; color:#5cfcb4; margin-top:0.3rem;">'
                f'🔌 {len(plugin_names)} plugin(s) loaded</div>',
                unsafe_allow_html=True
            )

        # ── Mode-specific settings ──
        api_key = None
        if "api" in mode.lower():
            st.markdown("---")
            st.markdown("#### 🔑 API Configuration")
            api_key = st.text_input(
                "Gemini API Key", type="password",
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
            <div><span style="color: #fc5c7c;">▸</span> Plugin System (BYOM)</div>
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
            <div class="app-title">LLM Extraction Benchmarking Arena</div>
            <div class="app-subtitle">
                Benchmark, compare, and analyze text extraction models
                across documents with scientific precision.
            </div>
        </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════
def main():
    """Main application entry point."""

    # ── Sidebar ──
    mode, api_key = render_sidebar()

    # ── Header ──
    render_header()

    # ── Navigation Tabs ──
    tab_microscope, tab_arena, tab_lab = st.tabs([
        "🔬 The Microscope",
        "⚔️ The Arena",
        "🧪 The Laboratory",
    ])

    with tab_microscope:
        microscope.render(mode=mode, api_key=api_key)

    with tab_arena:
        arena.render(mode=mode, api_key=api_key)

    with tab_lab:
        laboratory.render()


# ── Entry Point ──────────────────────────────────────────────
if __name__ == "__main__":
    main()
