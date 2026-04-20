"""
Configuration and constants for the LLM Extraction Benchmarking Arena.
Centralizes all app-wide settings, feature flags, and model names.
"""

import os

# ──────────────────────────────────────────────
# App Metadata
# ──────────────────────────────────────────────
APP_TITLE = "🔬 LLM Extraction Benchmarking Arena"
APP_DESCRIPTION = (
    "Benchmark, compare, and analyze text extraction models "
    "across documents with scientific precision."
)
PAGE_ICON = "🔬"
LAYOUT = "wide"

# ──────────────────────────────────────────────
# Supported File Types
# ──────────────────────────────────────────────
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
SUPPORTED_PDF_TYPES = ["pdf"]
SUPPORTED_FILE_TYPES = SUPPORTED_IMAGE_TYPES + SUPPORTED_PDF_TYPES

# Arena-specific: extensions to scan in dataset folders
ARENA_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# ──────────────────────────────────────────────
# Feature Flags
# ──────────────────────────────────────────────
ENABLE_LOCAL_MODEL = True  # Toggle Mode 3 (TrOCR) on/off
ENABLE_PLUGINS = True      # Toggle BYOM plugin discovery

# ──────────────────────────────────────────────
# Extraction Modes
# ──────────────────────────────────────────────
MODE_OCR = "🔍 OCR (Tesseract + EasyOCR)"
MODE_API = "🌐 API (Google Gemini Vision)"
MODE_LOCAL_MODEL = "🤖 Local Model (TrOCR)"

EXTRACTION_MODES = [MODE_OCR, MODE_API]
if ENABLE_LOCAL_MODEL:
    EXTRACTION_MODES.append(MODE_LOCAL_MODEL)

# ──────────────────────────────────────────────
# Model Names
# ──────────────────────────────────────────────
# TrOCR models (Mode 3)
TROCR_HANDWRITTEN_MODEL = "microsoft/trocr-large-handwritten"
TROCR_PRINTED_MODEL = "microsoft/trocr-large-printed"

# Gemini API (Mode 2)
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_EXTRACTION_PROMPT = (
    "Extract all text from this document image exactly as written. "
    "Return only the raw text content, no formatting, no markdown, "
    "no explanations. If multiple lines exist, preserve line breaks."
)

# Sentence Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ──────────────────────────────────────────────
# EasyOCR
# ──────────────────────────────────────────────
EASYOCR_LANGUAGES = ["en"]

# ──────────────────────────────────────────────
# Similarity Weights (equal by default)
# ──────────────────────────────────────────────
SIMILARITY_WEIGHTS = {
    "edit": 1.0,
    "tfidf": 1.0,
    "embedding": 1.0,
}

# ──────────────────────────────────────────────
# Storage (Laboratory)
# ──────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "results.db")

# ──────────────────────────────────────────────
# Plugins (BYOM)
# ──────────────────────────────────────────────
PLUGINS_DIR = os.path.join(_BASE_DIR, "plugins")

# ──────────────────────────────────────────────
# Live Console Log Levels
# ──────────────────────────────────────────────
LOG_COLORS = {
    "INFO":    "#58a6ff",   # Cyan-blue
    "SUCCESS": "#5cfcb4",   # Green
    "WARN":    "#fcbc5c",   # Amber
    "ERROR":   "#fc5c7c",   # Red
    "DEBUG":   "#6a6a80",   # Muted gray
}
