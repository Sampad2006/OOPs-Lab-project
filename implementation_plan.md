# LLM Document Extraction Benchmarking Arena — Upgrade Plan

Refactor the existing single-view Document Similarity Analyzer into a 3-view benchmarking application with live extraction console, batch processing, result history, and a plugin system — while preserving all existing OOP design patterns.

## User Review Required

> [!IMPORTANT]
> **Scope is large (~15 files modified/created).** The plan is structured into 6 phases that can be executed incrementally. Each phase produces a working app.

> [!WARNING]
> **Breaking change: `app.py` is fully rewritten.** The existing UI code is replaced with a tabbed 3-view layout. All backend logic (models/, utils/) is preserved and extended, never replaced.

## Proposed Changes

### Phase 1 — Foundation: Config, Storage & Plugin Infrastructure

---

#### [MODIFY] [config.py](file:///d:/10501019/AdvancedOOPS/OOPs-Lab-project/config.py)

Add new configuration constants:
- `APP_TITLE` → `"🔬 LLM Extraction Benchmarking Arena"`
- `DB_PATH = "data/results.db"` — SQLite path for Laboratory
- `PLUGINS_DIR = "plugins"` — Plugin folder path
- `ARENA_SUPPORTED_EXTENSIONS` — image exts for batch scanning
- `LOG_LEVELS` dict mapping log types to colors for the live console

---

#### [NEW] [data/results.db](file:///d:/10501019/AdvancedOOPS/OOPs-Lab-project/data/) *(auto-created at runtime)*

SQLite database, created on first run. Schema:

```sql
CREATE TABLE IF NOT EXISTS benchmark_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    view        TEXT NOT NULL,          -- 'microscope' or 'arena'
    strategy    TEXT NOT NULL,
    file_name   TEXT,
    scores      TEXT NOT NULL,          -- JSON blob
    metadata    TEXT                    -- JSON blob (word count, timings, etc.)
);
```

---

#### [NEW] [utils/storage.py](file:///d:/10501019/AdvancedOOPS/OOPs-Lab-project/utils/storage.py)

New `ResultsStore` class wrapping SQLite:
- `save_run(view, strategy, file_name, scores_dict, metadata_dict)` — insert row
- `get_runs(view=None, limit=50)` → list of dicts
- `clear_history()` — delete all
- `get_run_by_id(run_id)` → single dict
- Auto-creates table + `data/` directory on init

---

#### [NEW] [plugins/\_\_init\_\_.py](file:///d:/10501019/AdvancedOOPS/OOPs-Lab-project/plugins/__init__.py)

Empty init file to make `plugins/` a package.

---

#### [NEW] [plugins/example_plugin.py](file:///d:/10501019/AdvancedOOPS/OOPs-Lab-project/plugins/example_plugin.py)

A commented-out example showing how to write a BYOM plugin:
```python
# from models.extraction.base import ExtractionStrategy
# class MyCustomStrategy(ExtractionStrategy):
#     @property
#     def name(self) -> str: return "My Custom Model"
#     def extract_text(self, images, doc_type="printed") -> str: ...
```

---

#### [MODIFY] [models/extraction/factory.py](file:///d:/10501019/AdvancedOOPS/OOPs-Lab-project/models/extraction/factory.py)

Extend `StrategyFactory` with:
- `_plugin_registry: dict` — class-level cache of discovered plugins
- `scan_plugins()` — walks `plugins/` dir, imports `.py` files, finds subclasses of `ExtractionStrategy`, populates registry
- `available_modes()` — now returns built-in modes + plugin modes
- `create()` — extended with an `elif` that checks the plugin registry before raising ValueError
- Call `scan_plugins()` lazily on first `create()` or `available_modes()` call

This preserves the existing Factory Pattern and extends it with an **Observer-like discovery** mechanism.

---

### Phase 2 — Live Extraction Console & Microscope Upgrades

---

#### [NEW] [utils/live_console.py](file:///d:/10501019/AdvancedOOPS/OOPs-Lab-project/utils/live_console.py)

`LiveConsole` class that wraps `st.empty()`:
- `__init__(container)` — takes a Streamlit container
- `log(level, message)` — appends a styled HTML log line to session_state log buffer and re-renders
- Levels: `INFO` (cyan), `SUCCESS` (green), `WARN` (amber), `ERROR` (red), `DEBUG` (gray)
- Each line timestamped in `HH:MM:SS.mmm` format
- Styled as a monospace terminal (dark bg, scan-line effect)

---

#### [NEW] [utils/visual_detector.py](file:///d:/10501019/AdvancedOOPS/OOPs-Lab-project/utils/visual_detector.py)

`VisualDetector` class for bounding-box overlays:
- `detect_text_regions(image: PIL.Image) -> List[bbox]` — uses OpenCV contour detection on the binarized image to find text regions
- `draw_bounding_boxes(image: PIL.Image, regions) -> PIL.Image` — draws green rectangles over detected regions
- `generate_confidence_html(text, word_confidences) -> str` — returns HTML with `<span>` tags colored green→yellow→red based on confidence scores
- If model doesn't provide real confidence, generates mock confidence using word length heuristics + random jitter (documented as mock)

---

#### [MODIFY] [models/analyzer.py](file:///d:/10501019/AdvancedOOPS/OOPs-Lab-project/models/analyzer.py)

Add an optional `logger` callback parameter:
- `extract(document, logger=None)` — if logger is provided, calls `logger("INFO", "Initializing model...")` etc. at key steps
- `analyze(doc1, doc2, logger=None)` — passes logger through
- The Facade still works exactly the same without a logger (backward compatible)

---

### Phase 3 — The Arena (Batch Benchmarking View)

---

#### [NEW] [utils/arena_runner.py](file:///d:/10501019/AdvancedOOPS/OOPs-Lab-project/utils/arena_runner.py)

`ArenaRunner` class that orchestrates batch benchmarking:
- `__init__(dataset_path: str, strategies: List[ExtractionStrategy], metrics: List[SimilarityMetric])`
- `scan_dataset() -> List[Tuple[image_path, gt_path]]` — finds matching `doc.jpg` + `doc_gt.txt` pairs
- `run(progress_callback=None) -> pd.DataFrame` — for each pair × each strategy:
  1. Load image
  2. Extract text with strategy
  3. Compare extracted text vs ground-truth using similarity metrics
  4. Collect per-file, per-strategy scores
- Returns a DataFrame with columns: `File`, `Strategy`, `Edit Similarity`, `TF-IDF Similarity`, `Embedding Similarity`, `Final Similarity`, `Time (s)`, `Word Count`
- `compute_arena_scores(df) -> pd.DataFrame` — aggregates per-strategy averages

---

### Phase 4 — The Laboratory (Results History View)

This view uses `utils/storage.py` (from Phase 1) to display historical runs.

Rendered directly in `app.py` — no separate module needed. The view will:
- Query `ResultsStore.get_runs()`
- Display as a styled `st.dataframe` with filters (by view, by strategy, date range)
- Show expandable detail cards for each run
- Provide a "Clear History" button
- Display aggregate charts (bar chart of average scores per strategy over time)

---

### Phase 5 — The New `app.py` (3-View UI)

---

#### [MODIFY] [app.py](file:///d:/10501019/AdvancedOOPS/OOPs-Lab-project/app.py) *(full rewrite)*

**Top-level layout:**
```
┌─────────────────────────────────────────────┐
│  🔬 LLM Extraction Benchmarking Arena       │
│  ─────────────────────────────────────────  │
│  [🔬 Microscope] [⚔️ Arena] [🧪 Laboratory] │  ← st.tabs
├─────────────────────────────────────────────┤
│                                             │
│         (Active tab content)                │
│                                             │
└─────────────────────────────────────────────┘
```

**Sidebar** (shared across all views):
- App branding (updated title/icon)
- Extraction mode selector (now includes plugin-discovered modes)
- API key input (shown conditionally)
- OOP Patterns section (preserved)
- Similarity Metrics info (preserved)

**Tab 1 — 🔬 The Microscope:**
- Two-column file upload (handwritten + printed) with enhanced drag-drop CSS
- **Checkbox**: "Show Detection Visuals" → stored in `st.session_state`
- **Checkbox**: "Show Confidence Heatmap" → stored in `st.session_state`
- "Analyze & Compare" button
- On click:
  - Live Console appears (mock terminal with scrolling logs)
  - If detection visuals enabled:
    - For PDFs: `st.slider("Select page to visualize", 1, num_pages, 1)` — renders bounding boxes for selected page only
    - For images: renders bounding boxes directly
  - Extracted text shown with confidence heatmap coloring (if enabled)
  - Similarity scores rendered (existing score cards)
  - Result auto-saved to SQLite

**Tab 2 — ⚔️ The Arena:**
- `st.text_input("Dataset folder path")` for local directory
- `st.info` instruction panel with folder structure rules
- `st.multiselect("Select strategies to race")` — populated from `StrategyFactory.available_modes()`
- "Run Benchmark" button
- On click:
  - Validates directory structure
  - Progress bar for batch processing
  - Results displayed as:
    - Detailed per-file DataFrame
    - Aggregated "Arena Score" summary table
    - Bar chart comparing strategies
  - Results auto-saved to SQLite

**Tab 3 — 🧪 The Laboratory:**
- Filter controls (view type, strategy, date range)
- Historical runs table
- Expandable detail cards
- Aggregate charts
- "Clear History" button with confirmation

---

### Phase 6 — CSS & UI Polish

---

#### [MODIFY] [assets/style.css](file:///d:/10501019/AdvancedOOPS/OOPs-Lab-project/assets/style.css)

Add new CSS classes while preserving all existing styles:

- **`.live-console`** — Terminal-look container with dark bg, monospace font, green scanline effect, fixed height + overflow scroll
- **`.console-line-info`**, **`.console-line-success`**, **`.console-line-warn`**, **`.console-line-error`** — colored log line styles
- **`.upload-zone-enhanced`** — Larger drop zones with dashed borders, animated hover, pulse effect
- **`.tab-header`** — Styled tab titles
- **`.arena-table`** — Styled dataframe for Arena results
- **`.confidence-high`** (green), **`.confidence-med`** (yellow), **`.confidence-low`** (red) — Heatmap spans
- **`.lab-card`** — Glass card variant for Laboratory entries
- **`.nav-tabs`** — Custom styling for `st.tabs` to match the scientific aesthetic

All new styles use the existing CSS custom properties (`--bg-card`, `--accent-primary`, etc.) for consistency.

---

#### [MODIFY] [requirements.txt](file:///d:/10501019/AdvancedOOPS/OOPs-Lab-project/requirements.txt)

Add:
```
pandas>=2.0.0
plotly>=5.15.0
```
(SQLite is stdlib — no new dep. OpenCV already listed.)

---

## File Change Summary

| File | Action | Phase |
|------|--------|-------|
| `config.py` | MODIFY | 1 |
| `utils/storage.py` | NEW | 1 |
| `plugins/__init__.py` | NEW | 1 |
| `plugins/example_plugin.py` | NEW | 1 |
| `models/extraction/factory.py` | MODIFY | 1 |
| `utils/live_console.py` | NEW | 2 |
| `utils/visual_detector.py` | NEW | 2 |
| `models/analyzer.py` | MODIFY | 2 |
| `utils/arena_runner.py` | NEW | 3 |
| `app.py` | REWRITE | 5 |
| `assets/style.css` | MODIFY | 6 |
| `requirements.txt` | MODIFY | 6 |

**Files NOT touched** (preserved as-is):
- `models/document.py`
- `models/extraction/base.py`
- `models/extraction/ocr_strategy.py`
- `models/extraction/api_strategy.py`
- `models/extraction/local_model_strategy.py`
- `models/similarity/base.py`
- `models/similarity/aggregator.py`
- `models/similarity/edit_distance.py`
- `models/similarity/tfidf_similarity.py`
- `models/similarity/embedding_similarity.py`
- `utils/file_handler.py`
- `utils/preprocessor.py`

## Open Questions

> [!IMPORTANT]
> **1. Plotly vs Matplotlib for charts?** The plan uses Plotly for interactive charts in the Arena/Laboratory views (hover tooltips, zoom). If you prefer static matplotlib charts, let me know.

> [!IMPORTANT]
> **2. Multi-page Streamlit vs single-file with `st.tabs`?** The plan uses `st.tabs()` in a single `app.py` for simplicity. Streamlit also supports a `pages/` directory for true multi-page apps. Which do you prefer?

> [!NOTE]
> **3. Mock confidence data.** Since Tesseract and EasyOCR can return word-level confidence, I'll use real data where available. For the Gemini API and TrOCR strategies (which don't expose per-word confidence), the heatmap will use mocked confidence with a clear "(simulated)" label.

## Verification Plan

### Automated Tests
- Run `streamlit run app.py` and verify all three tabs render without errors
- Test plugin discovery by placing a valid plugin in `plugins/`
- Test SQLite storage: save a run, query it back, clear history
- Test Arena runner against the existing `test_data/` directory (if it contains ground-truth files)

### Manual Verification
- Upload sample images in the Microscope view and verify:
  - Live console logs appear and scroll
  - Bounding boxes render on the selected page
  - Confidence heatmap colors appear on extracted text
- Run a batch benchmark in the Arena and verify the comparative DataFrame + chart
- Check the Laboratory view shows previously stored results
