"""
Microscope View — Single-pair document analysis with live console,
detection visuals, and confidence heatmaps.
"""

import streamlit as st
import time

import config
from models.document import Document, DocType, FileType
from models.extraction.factory import StrategyFactory
from models.analyzer import DocumentAnalyzer
from utils.file_handler import FileHandler
from utils.preprocessor import TextPreprocessor
from utils.live_console import LiveConsole
from utils.visual_detector import VisualDetector
from utils.storage import ResultsStore


def _render_score_card(label: str, score: float, icon: str = "📊"):
    percentage = int(score * 100)
    if score >= 0.7:
        color = "#5cfcb4"
    elif score >= 0.4:
        color = "#fcbc5c"
    else:
        color = "#fc5c7c"
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


def _render_final_score(score: float):
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


def _render_results(results: dict):
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; margin: 1.5rem 0;">
            <span style="font-size: 1.3rem; font-weight: 700; color: #e8e8f0;">
                📝 Extracted Text
            </span>
        </div>
    """, unsafe_allow_html=True)

    show_heatmap = st.session_state.get("show_heatmap", False)

    text_col1, text_col2 = st.columns(2, gap="large")
    with text_col1:
        with st.expander("✍️ **Handwritten Document — Extracted Text**", expanded=True):
            if show_heatmap and results.get("text1"):
                html = VisualDetector.generate_confidence_html(
                    results["text1"], mock=True
                )
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.text_area("hw", value=results["text1"] or "(No text extracted)",
                             height=250, key="text_output_1", label_visibility="collapsed")

    with text_col2:
        with st.expander("🖨️ **Printed Document — Extracted Text**", expanded=True):
            if show_heatmap and results.get("text2"):
                html = VisualDetector.generate_confidence_html(
                    results["text2"], mock=True
                )
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.text_area("pr", value=results["text2"] or "(No text extracted)",
                             height=250, key="text_output_2", label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; margin: 1.5rem 0;">
            <span style="font-size: 1.3rem; font-weight: 700; color: #e8e8f0;">
                📊 Similarity Scores
            </span>
        </div>
    """, unsafe_allow_html=True)

    scores = results["scores"]
    metric_icons = {
        "Edit Similarity": "📐",
        "TF-IDF Similarity": "📝",
        "Embedding Similarity": "🧠",
    }
    score_cols = st.columns(3, gap="medium")
    for i, (metric_name, icon) in enumerate(metric_icons.items()):
        if metric_name in scores:
            with score_cols[i]:
                _render_score_card(metric_name, scores[metric_name], icon)

    st.markdown("<br>", unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns([1, 2, 1])
    with fc2:
        _render_final_score(scores.get("Final Similarity", 0.0))


def _render_detection_visuals(images, label, key_prefix):
    """Render bounding-box visuals with PDF page slider safety."""
    if not images:
        return
    num_pages = len(images)
    if num_pages > 1:
        page_idx = st.slider(
            f"Select page to visualize ({label})",
            min_value=1, max_value=num_pages, value=1,
            key=f"{key_prefix}_page_slider"
        ) - 1
    else:
        page_idx = 0

    selected_image = images[page_idx]
    regions = VisualDetector.detect_text_regions(selected_image)
    annotated = VisualDetector.draw_bounding_boxes(selected_image, regions)
    st.image(annotated, caption=f"{label} — Page {page_idx + 1} ({len(regions)} regions detected)",
             use_container_width=True)


def render(mode: str, api_key: str = None):
    """Main render function for the Microscope tab."""

    st.markdown(f"""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <span class="mode-badge">⚙️ {mode}</span>
        </div>
    """, unsafe_allow_html=True)

    # Upload section
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
            <div class="glass-card"><div class="card-title">
                <span class="icon">✍️</span> Handwritten Document
            </div></div>
        """, unsafe_allow_html=True)
        handwritten_file = st.file_uploader(
            "Upload handwritten document", type=config.SUPPORTED_FILE_TYPES,
            key="handwritten_upload", help="Upload JPG, PNG, or PDF",
            label_visibility="collapsed"
        )
        if handwritten_file:
            st.success(f"✅ Uploaded: **{handwritten_file.name}**", icon="📎")

    with col2:
        st.markdown("""
            <div class="glass-card"><div class="card-title">
                <span class="icon">🖨️</span> Printed Document
            </div></div>
        """, unsafe_allow_html=True)
        printed_file = st.file_uploader(
            "Upload printed document", type=config.SUPPORTED_FILE_TYPES,
            key="printed_upload", help="Upload JPG, PNG, or PDF",
            label_visibility="collapsed"
        )
        if printed_file:
            st.success(f"✅ Uploaded: **{printed_file.name}**", icon="📎")

    # Visual toggles
    toggle_col1, toggle_col2, _ = st.columns([1, 1, 2])
    with toggle_col1:
        show_visuals = st.checkbox("🔲 Show Detection Visuals", key="show_visuals")
    with toggle_col2:
        st.checkbox("🌡️ Show Confidence Heatmap", key="show_heatmap")

    st.markdown("<br>", unsafe_allow_html=True)

    # Analyze button
    bc1, bc2, bc3 = st.columns([1, 1, 1])
    with bc2:
        analyze_clicked = st.button(
            "🚀  Analyze & Compare", use_container_width=True,
            disabled=not (handwritten_file and printed_file),
        )

    if analyze_clicked:
        if not handwritten_file or not printed_file:
            st.error("⚠️ Please upload both documents.", icon="❌")
            return
        if "api" in mode.lower() and not api_key:
            st.error("⚠️ Please enter your Gemini API key in the sidebar.", icon="🔑")
            return

        try:
            # Live console
            console_container = st.empty()
            console = LiveConsole(console_container, session_key="microscope_logs")
            console.clear()

            def log_cb(level, msg):
                console.log(level, msg, delay=0.15)

            log_cb("INFO", "Loading documents...")

            hw_ext = handwritten_file.name.rsplit(".", 1)[-1]
            pr_ext = printed_file.name.rsplit(".", 1)[-1]
            hw_bytes = handwritten_file.read()
            pr_bytes = printed_file.read()

            hw_images = FileHandler.load_images(hw_bytes, FileType.from_extension(hw_ext))
            pr_images = FileHandler.load_images(pr_bytes, FileType.from_extension(pr_ext))
            log_cb("SUCCESS", f"Loaded: {len(hw_images)} + {len(pr_images)} page(s)")

            doc1 = Document(file_name=handwritten_file.name,
                            file_type=FileType.from_extension(hw_ext),
                            doc_type=DocType.HANDWRITTEN, images=hw_images)
            doc2 = Document(file_name=printed_file.name,
                            file_type=FileType.from_extension(pr_ext),
                            doc_type=DocType.PRINTED, images=pr_images)

            # Detection visuals
            if show_visuals:
                vis_c1, vis_c2 = st.columns(2)
                with vis_c1:
                    _render_detection_visuals(hw_images, "Handwritten", "hw")
                with vis_c2:
                    _render_detection_visuals(pr_images, "Printed", "pr")

            # Extract & compare
            kwargs = {}
            if "api" in mode.lower():
                kwargs["api_key"] = api_key
            strategy = StrategyFactory.create(mode, **kwargs)
            analyzer = DocumentAnalyzer(strategy)

            text1 = analyzer.extract(doc1, logger=log_cb)
            text1 = TextPreprocessor.clean_ocr_output(text1)
            doc1.extracted_text = text1

            text2 = analyzer.extract(doc2, logger=log_cb)
            text2 = TextPreprocessor.clean_ocr_output(text2)
            doc2.extracted_text = text2

            scores = analyzer.compare(text1, text2, logger=log_cb)
            log_cb("SUCCESS", "✅ Analysis complete!")

            results = {"text1": text1, "text2": text2, "scores": scores}
            st.session_state["microscope_results"] = results

            # Save to history
            try:
                store = ResultsStore()
                store.save_run(
                    view="microscope", strategy=strategy.name,
                    file_name=f"{handwritten_file.name} vs {printed_file.name}",
                    scores=scores,
                    metadata={"words_hw": len(text1.split()), "words_pr": len(text2.split())}
                )
            except Exception:
                pass

        except Exception as e:
            st.error(f"❌ **Error:** {str(e)}", icon="🚨")
            import traceback
            with st.expander("🔍 Full Error Details"):
                st.code(traceback.format_exc())

    if "microscope_results" in st.session_state:
        _render_results(st.session_state["microscope_results"])
