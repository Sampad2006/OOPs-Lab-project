"""
Laboratory View — Historical results browser with filters and charts.
"""

import streamlit as st
from datetime import datetime

from utils.storage import ResultsStore


def render():
    """Main render function for the Laboratory tab."""

    st.markdown("""
        <div style="text-align:center; margin-bottom:1rem;">
            <span style="font-size:1.1rem; font-weight:700; color:#e8e8f0;">
                🧪 Browse and analyze past benchmark runs
            </span>
        </div>
    """, unsafe_allow_html=True)

    try:
        store = ResultsStore()
    except Exception as e:
        st.error(f"Could not connect to results database: {e}")
        return

    # Filters
    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])
    with filter_col1:
        view_filter = st.selectbox(
            "Filter by View",
            options=["All", "microscope", "arena"],
            key="lab_view_filter"
        )
    with filter_col2:
        strategy_filter = st.text_input(
            "Filter by Strategy (contains)",
            placeholder="e.g., OCR",
            key="lab_strategy_filter"
        )
    with filter_col3:
        limit = st.number_input("Max results", min_value=10, max_value=500,
                                value=50, step=10, key="lab_limit")

    # Query
    view_param = None if view_filter == "All" else view_filter
    runs = store.get_runs(view=view_param, limit=limit)

    # Filter by strategy text
    if strategy_filter:
        runs = [r for r in runs if strategy_filter.lower() in r["strategy"].lower()]

    st.markdown(f"""
        <div style="margin:1rem 0; color:#a0a0b8; font-size:0.85rem;">
            Showing <strong style="color:#e8e8f0;">{len(runs)}</strong> results
        </div>
    """, unsafe_allow_html=True)

    if not runs:
        st.info("No benchmark runs found. Run an analysis in the Microscope or Arena tab first!", icon="📭")
        return

    # Results table
    import pandas as pd

    table_data = []
    for run in runs:
        ts = run.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(ts)
            ts_display = dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            ts_display = ts[:16]

        scores = run.get("scores", {})
        final = scores.get("Final Similarity", scores.get("Arena Score", 0.0))

        table_data.append({
            "ID": run["id"],
            "Date": ts_display,
            "View": run["view"].title(),
            "Strategy": run["strategy"],
            "File": run.get("file_name", "—")[:40],
            "Score": final,
        })

    df = pd.DataFrame(table_data)
    st.dataframe(
        df, use_container_width=True, hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Final Score", min_value=0, max_value=1, format="%.4f"
            ),
        }
    )

    # Expandable detail cards
    st.markdown("---")
    st.markdown("##### 🔎 Run Details")

    for run in runs[:20]:  # Limit detail cards
        ts = run.get("timestamp", "")[:16]
        scores = run.get("scores", {})
        meta = run.get("metadata", {}) or {}

        with st.expander(
            f"**#{run['id']}** — {run['strategy']} | {run['view']} | {ts}"
        ):
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown("**Scores:**")
                for k, v in scores.items():
                    pct = int(v * 100) if isinstance(v, (int, float)) else "—"
                    st.markdown(f"- {k}: **{v}** ({pct}%)")
            with sc2:
                st.markdown("**Metadata:**")
                if meta:
                    for k, v in meta.items():
                        st.markdown(f"- {k}: `{v}`")
                else:
                    st.markdown("_No metadata_")
                st.markdown(f"- File: `{run.get('file_name', '—')}`")

    # Aggregate chart
    if len(runs) >= 2:
        st.markdown("---")
        st.markdown("##### 📈 Score Distribution by Strategy")

        chart_rows = []
        for run in runs:
            scores = run.get("scores", {})
            final = scores.get("Final Similarity", scores.get("Arena Score", 0.0))
            chart_rows.append({
                "Strategy": run["strategy"],
                "Score": final,
            })

        chart_df = pd.DataFrame(chart_rows)
        avg_by_strategy = chart_df.groupby("Strategy")["Score"].mean().reset_index()
        avg_by_strategy.columns = ["Strategy", "Avg Score"]
        avg_chart = avg_by_strategy.set_index("Strategy")
        st.bar_chart(avg_chart, color="#5cfcb4")

    # Clear history
    st.markdown("---")
    clear_col1, clear_col2, _ = st.columns([1, 1, 2])
    with clear_col1:
        if st.button("🗑️ Clear All History", type="secondary", key="clear_history_btn"):
            st.session_state["confirm_clear"] = True
    with clear_col2:
        if st.session_state.get("confirm_clear", False):
            if st.button("⚠️ Confirm Delete", type="primary", key="confirm_clear_btn"):
                deleted = store.clear_history()
                st.success(f"Deleted {deleted} records.", icon="🗑️")
                st.session_state["confirm_clear"] = False
                st.rerun()
