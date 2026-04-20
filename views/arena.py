"""
Arena View — Batch benchmarking of multiple extraction strategies
against a ground-truth dataset.
"""

import streamlit as st

import config
from models.extraction.factory import StrategyFactory
from utils.arena_runner import ArenaRunner
from utils.storage import ResultsStore


def render(mode: str, api_key: str = None):
    """Main render function for the Arena tab."""

    st.markdown("""
        <div style="text-align:center; margin-bottom:1rem;">
            <span style="font-size:1.1rem; font-weight:700; color:#e8e8f0;">
                ⚔️ Race extraction models against a ground-truth dataset
            </span>
        </div>
    """, unsafe_allow_html=True)

    # Instruction panel
    with st.expander("📖 **Benchmarking Rules & Dataset Format**", expanded=False):
        st.info(
            "**To run a benchmark, your folder must contain images and their "
            "ground-truth text files with matching names.**\n\n"
            "**Expected structure:**\n"
            "```\n"
            "dataset_folder/\n"
            "  doc1.jpg\n"
            "  doc1_gt.txt\n"
            "  doc2.png\n"
            "  doc2_gt.txt\n"
            "```\n\n"
            "- Image formats: JPG, JPEG, PNG, BMP, TIFF\n"
            "- Ground-truth files must end with `_gt.txt`\n"
            "- The base name must match (e.g., `sample` → `sample.jpg` + `sample_gt.txt`)",
            icon="📋"
        )

    # Dataset path input
    dataset_path = st.text_input(
        "📂 Dataset Folder Path",
        placeholder="e.g., C:\\datasets\\handwriting_benchmark",
        help="Enter the absolute path to your dataset folder.",
        key="arena_dataset_path"
    )

    # Strategy selection
    all_modes = StrategyFactory.available_modes()
    # Filter out API mode if no key provided
    selectable = []
    for m in all_modes:
        if "api" in m.lower() and not api_key:
            continue
        selectable.append(m)

    selected_strategies = st.multiselect(
        "🏁 Select strategies to race",
        options=selectable,
        default=[selectable[0]] if selectable else [],
        help="Choose one or more extraction models to benchmark.",
        key="arena_strategies"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Run button
    bc1, bc2, bc3 = st.columns([1, 1, 1])
    with bc2:
        run_clicked = st.button(
            "⚔️  Run Benchmark",
            use_container_width=True,
            disabled=not (dataset_path and selected_strategies),
        )

    if run_clicked:
        if not dataset_path:
            st.error("Please specify a dataset folder path.", icon="📂")
            return
        if not selected_strategies:
            st.error("Please select at least one strategy.", icon="🏁")
            return

        try:
            # Create strategy instances
            strategies = []
            for m in selected_strategies:
                kwargs = {}
                if "api" in m.lower():
                    kwargs["api_key"] = api_key
                strategies.append(StrategyFactory.create(m, **kwargs))

            runner = ArenaRunner(dataset_path=dataset_path, strategies=strategies)

            # Validate dataset
            pairs = runner.scan_dataset()
            st.success(f"Found **{len(pairs)}** image + ground-truth pairs.", icon="✅")

            # Progress bar
            progress_bar = st.progress(0, text="Starting benchmark...")
            status_text = st.empty()

            def progress_cb(current, total, status):
                progress_bar.progress(current / total, text=status)

            # Run benchmark
            results = runner.run(progress_callback=progress_cb)
            progress_bar.progress(1.0, text="✅ Benchmark complete!")

            st.session_state["arena_results"] = results
            st.session_state["arena_summary"] = ArenaRunner.compute_arena_scores(results)

            # Save to history
            try:
                store = ResultsStore()
                summary = st.session_state["arena_summary"]
                for entry in summary:
                    store.save_run(
                        view="arena",
                        strategy=entry["Strategy"],
                        file_name=dataset_path,
                        scores={
                            "Arena Score": entry["Arena Score (Avg Final)"],
                            "Avg Edit": entry["Avg Edit Similarity"],
                            "Avg TF-IDF": entry["Avg TF-IDF Similarity"],
                            "Avg Embedding": entry["Avg Embedding Similarity"],
                        },
                        metadata={
                            "files_processed": entry["Files Processed"],
                            "total_time": entry["Total Time (s)"],
                        }
                    )
            except Exception:
                pass

        except Exception as e:
            st.error(f"❌ **Error:** {str(e)}", icon="🚨")
            import traceback
            with st.expander("🔍 Full Error Details"):
                st.code(traceback.format_exc())

    # Display results
    if "arena_summary" in st.session_state:
        _render_arena_results()


def _render_arena_results():
    """Render Arena benchmark results."""
    import pandas as pd

    summary = st.session_state.get("arena_summary", [])
    results = st.session_state.get("arena_results", [])

    if not summary:
        return

    st.markdown("---")

    # Arena Scores summary
    st.markdown("""
        <div style="text-align:center; margin:1.5rem 0;">
            <span style="font-size:1.3rem; font-weight:700; color:#e8e8f0;">
                🏆 Arena Scores — Strategy Leaderboard
            </span>
        </div>
    """, unsafe_allow_html=True)

    summary_df = pd.DataFrame(summary)
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Arena Score (Avg Final)": st.column_config.ProgressColumn(
                "Arena Score", min_value=0, max_value=1, format="%.4f"
            ),
        }
    )

    # Bar chart
    st.markdown("<br>", unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("##### 📊 Arena Score Comparison")
        chart_data = summary_df[["Strategy", "Arena Score (Avg Final)"]].set_index("Strategy")
        st.bar_chart(chart_data, color="#7c5cfc")

    with chart_col2:
        st.markdown("##### ⏱️ Total Time Comparison")
        time_data = summary_df[["Strategy", "Total Time (s)"]].set_index("Strategy")
        st.bar_chart(time_data, color="#fcbc5c")

    # Detailed per-file results
    with st.expander("📋 **Detailed Per-File Results**", expanded=False):
        details_df = pd.DataFrame(results)
        display_cols = [c for c in details_df.columns if c != "Extracted Text"]
        st.dataframe(details_df[display_cols], use_container_width=True, hide_index=True)
